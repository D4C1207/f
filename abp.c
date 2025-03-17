/*
================================================================================
RS(15,11) 編碼／譯碼及 ABP 軟解碼整合示例（CPU 版本，參數化設計）
================================================================================
*/

// Remove any unnecessary CUDA-related header files if they existed
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//================== 可調 RS 參數設定 ===================
#define mm 8                  // 每個符號的位數
#define m ((1 << mm) - 1)       // GF(2^12) 的非零元素個數 = 4095
#define nn 255                 // RS 碼長（符號數）
#define kk 239                // RS 資料符號數 (此處 t = 60)
#define tt ((nn - kk) / 2)       // 可糾正符號數，應為 60


//================== ABP 參數設定 ===================
// 二進位展開參數
#define RS_SYM nn                          // RS 碼符號數 = nn
#define RS_DAT kk                          // RS 資料符號數 = kk
#define TOTAL_BITS (RS_SYM * mm)             // 總二進位位數 = nn * mm
#define PARITY_BITS (((RS_SYM - RS_DAT) * mm))// 校驗位數 = (nn - kk) * mm

// ABP 迭代參數與信道設定
#define MAX_ITER 10    // 最大迭代次數（本例設為 10）
#define DAMPING_START 0.8   // 初始 damping 係數
#define DAMPING_MIN 0.3    // 最小 damping 係數
double SNR_DB = 7.0;     // 信噪比, 單位 dB

// 模擬試驗次數
#define NUM_TRIALS 10

//================== 更新門檻設定 ===================
#define UPDATE_THRESHOLD 5.0  // 提高更新門檻

void adapt_H(double* L);
// 修改 ABP 參數設定
#define MAX_ITER_HIGH_SNR 3     // 高 SNR 時的最大迭代次數
#define MAX_ITER_LOW_SNR 15    // 低 SNR 時的最大迭代次數
#define SNR_THRESHOLD 8.0      // SNR 高低的分界點

//================== GF 與 RS 編碼／譯碼變數 ===================

// 不可約多項式 pp[]（此例中使用一組 11 個係數表示的多項式）
int pp[9] = { 1, 0, 1, 1, 1, 0, 0, 0, 1 };




// 查表：
// alpha_to[i] 儲存 α^i 的多項式表示；index_of[j] 為 j 對應的指數
// gg[] 儲存生成多項式 g(x) 的係數（後續轉換為指數形式）
int alpha_to[m + 1], index_of[m + 1], gg[nn - kk + 1];

// recd[] 儲存接收到的 RS 碼字（以指數形式儲存）；data[] 儲存原始資料；bb[] 儲存 RS 編碼產生的校驗符號
int recd[nn], data[kk], bb[nn - kk];

//================== ABP 二進位校驗矩陣 ===================
// HBit 的尺寸為 PARITY_BITS × TOTAL_BITS
unsigned char HBit[PARITY_BITS][TOTAL_BITS];

//================== GF(2^mm) 生成與 RS 編碼／譯碼函式 ===================

void generate_gf()
{
    int i, j;
    int mask = 1;
    alpha_to[mm] = 0;
    for (i = 0; i < mm; i++) {
        alpha_to[i] = mask;
        index_of[alpha_to[i]] = i;
        if (pp[i] != 0)
            alpha_to[mm] ^= mask;
        mask <<= 1;
    }
    index_of[alpha_to[mm]] = mm;
    mask >>= 1;
    for (i = mm + 1; i < m; i++) {
        if (alpha_to[i - 1] >= mask)
            alpha_to[i] = alpha_to[mm] ^ ((alpha_to[i - 1] ^ mask) << 1);
        else
            alpha_to[i] = alpha_to[i - 1] << 1;
        index_of[alpha_to[i]] = i;
    }
    index_of[0] = -1;
}

void gen_poly()
{
    int i, j;
    gg[0] = 2;
    gg[1] = 1;
    for (i = 2; i <= nn - kk; i++) {
        gg[i] = 1;
        for (j = i - 1; j > 0; j--) {
            if (gg[j] != 0)
                gg[j] = gg[j - 1] ^ alpha_to[(index_of[gg[j]] + i) % m];
            else
                gg[j] = gg[j - 1];
        }
        gg[0] = alpha_to[(index_of[gg[0]] + i) % m];
    }
    for (i = 0; i <= nn - kk; i++)
        gg[i] = index_of[gg[i]];
}

static unsigned char gf_mul(unsigned char a, unsigned char b) {
    if (a == 0 || b == 0)
        return 0;
    int s = index_of[a] + index_of[b];
    s %= m;
    return alpha_to[s];
}

void encode_rs()
{
    int i, j, feedback;
    for (i = 0; i < nn - kk; i++)
        bb[i] = 0;
    for (i = kk - 1; i >= 0; i--) {
        feedback = index_of[data[i] ^ bb[nn - kk - 1]];
        if (feedback != -1) {
            for (j = nn - kk - 1; j > 0; j--) {
                if (gg[j] != -1)
                    bb[j] = bb[j - 1] ^ alpha_to[(gg[j] + feedback) % m];
                else
                    bb[j] = bb[j - 1];
            }
            bb[0] = alpha_to[(gg[0] + feedback) % m];
        }
        else {
            for (j = nn - kk - 1; j > 0; j--)
                bb[j] = bb[j - 1];
            bb[0] = 0;
        }
    }
}

void decode_rs()
{
    int i, j, u, q, zz, Zout;
    int elp[nn - kk + 2][nn - kk + 1];
    int s[nn - kk + 2];
    int b_arr[nn - kk + 2][nn - kk + 1];
    int d_be[nn - kk + 3];
    int l_elp[nn - kk + 2], l_b[nn - kk + 2];
    int reg_elp[tt + 1], reg_b[tt];
    int iternum, err_num;
    int elp_odd, elp_even, b_sum;
    int syn_error = 0;
    for (i = 1; i <= nn - kk; i++) {
        s[i] = 0;
        for (j = 0; j < nn; j++) {
            if (recd[j] != -1)
                s[i] ^= alpha_to[(recd[j] + i * j) % m];
        }
        if (s[i] != 0)
            syn_error = 1;
        s[i] = index_of[s[i]];
    }
    if (syn_error) {
        for (i = 0; i <= nn - kk; i++) {
            elp[0][i] = -1;
            elp[1][i] = s[i + 1];
            b_arr[0][i] = -1;
            b_arr[1][i] = s[i + 1];
        }
        b_arr[1][nn - kk - 1] = -1;
        b_arr[1][nn - kk] = 0;
        elp[1][nn - kk] = 0;
        d_be[0] = -1;
        d_be[1] = 0;
        zz = 1;
        u = 0;
        iternum = 0;
        l_elp[0] = -1;
        l_elp[1] = 0;
        l_b[0] = -1;
        l_b[1] = 0;
        do {
            u++;
            if ((l_elp[u] <= l_b[u]) && (elp[u][0] != -1)) {
                l_elp[u + 1] = l_b[u] + 1;
                l_b[u + 1] = l_elp[u];
                d_be[u + 1] = elp[u][0];
                zz = alpha_to[(index_of[zz] + m - 1) % m];
                for (i = 0; i < nn - kk; i++) {
                    b_arr[u + 1][i] = elp[u][i + 1];
                }
                b_arr[u + 1][nn - kk] = -1;
            }
            else {
                if (l_b[u] == tt - 1) {
                    for (i = 0; i < nn - kk; i++) {
                        b_arr[u + 1][i] = b_arr[u][i + 1];
                    }
                    b_arr[u + 1][nn - kk] = -1;
                    l_b[u + 1] = l_b[u];
                }
                else {
                    for (i = 0; i <= nn - kk; i++) {
                        b_arr[u + 1][i] = b_arr[u][i];
                    }
                    l_b[u + 1] = l_b[u] + 1;
                    zz = alpha_to[(index_of[zz] + m - 1) % m];
                }
                d_be[u + 1] = d_be[u];
                l_elp[u + 1] = l_elp[u];
            }
            b_arr[u + 1][nn - kk - u - 1] = -1;
            for (i = 0; i <= nn - kk; i++)
                elp[u + 1][i] = 0;
            for (i = 0; i <= nn - kk; i++) {
                if (i != nn - kk) {
                    if (((elp[u][i + 1] == -1) || (d_be[u] == -1)) &&
                        ((elp[u][0] == -1) || (b_arr[u][i] == -1)))
                        elp[u + 1][i] = 0;
                    else if ((elp[u][0] == -1) || (b_arr[u][i] == -1))
                        elp[u + 1][i] = alpha_to[(d_be[u] + elp[u][i + 1]) % m];
                    else if ((d_be[u] == -1) || (elp[u][i + 1] == -1))
                        elp[u + 1][i] = alpha_to[(elp[u][0] + b_arr[u][i]) % m];
                    else
                        elp[u + 1][i] = alpha_to[(d_be[u] + elp[u][i + 1]) % m] ^
                        alpha_to[(elp[u][0] + b_arr[u][i]) % m];
                    elp[u + 1][i] = index_of[elp[u + 1][i]];
                }
                else {
                    if ((elp[u][0] == -1) || (b_arr[u][i] == -1))
                        elp[u + 1][i] = 0;
                    else
                        elp[u + 1][i] = alpha_to[(elp[u][0] + b_arr[u][i]) % m];
                    elp[u + 1][i] = index_of[elp[u + 1][i]];
                }
            }
            iternum++;
        } while (u < 2 * tt);
        u = u + 1;
        Zout = alpha_to[d_be[u]];
        zz = index_of[zz];
        err_num = 0;
        int err[m];
        for (i = 0; i < nn; i++) {
            err[i] = 0;
            if (recd[i] != -1)
                recd[i] = alpha_to[recd[i]];
            else
                recd[i] = 0;
        }
        if (l_elp[u - 1] <= tt) {
            int reg_elp[tt + 1], reg_b[tt];
            for (i = 0; i <= tt; i++) {
                reg_elp[i] = elp[u][i];
                if (i != tt)
                    reg_b[i] = b_arr[u][i];
            }
            for (j = 0; j < m; j++) {
                elp_odd = 0;
                elp_even = 0;
                b_sum = 0;
                if (j == 0) {
                    if ((elp[u][0] == -1) || (Zout == 0))
                        Zout = 0;
                    else
                        Zout = alpha_to[(index_of[Zout] + elp[u][0]) % m];
                }
                else {
                    if ((zz == -1) || (Zout == 0))
                        Zout = 0;
                    else
                        Zout = alpha_to[(index_of[Zout] + zz) % m];
                }
                for (i = 0; i <= tt; i++) {
                    if (reg_elp[i] != -1) {
                        if (i % 2 == 0)
                            elp_even ^= alpha_to[reg_elp[i]];
                        else
                            elp_odd ^= alpha_to[reg_elp[i]];
                    }
                    if ((reg_b[i] != -1) && (i != tt))
                        b_sum ^= alpha_to[reg_b[i]];
                    if (reg_elp[i] != -1)
                        reg_elp[i] = ((reg_elp[i] + m - i) % m);
                    if (i != tt) {
                        if (reg_b[i] != -1)
                            reg_b[i] = ((reg_b[i] + m - i) % m);
                        else
                            reg_b[i] = -1;
                    }
                }
                if (elp_even == elp_odd) {
                    if ((Zout != 0) && (b_sum != 0) && (elp_odd != 0))
                        err[j] = alpha_to[(index_of[Zout] + 2 * m - index_of[b_sum] - index_of[elp_odd]) % m];
                    err_num++;
                }
            }
            if (err_num == l_elp[u - 1]) {
                for (i = 0; i < nn; i++)
                    recd[i] ^= err[i];
            }
        }
        else {
            for (i = 0; i < nn; i++) {
                if (recd[i] != -1)
                    recd[i] = alpha_to[recd[i]];
                else
                    recd[i] = 0;
            }
        }
    }
    else {
        for (i = 0; i < nn; i++) {
            if (recd[i] != -1)
                recd[i] = alpha_to[recd[i]];
            else
                recd[i] = 0;
        }
    }
}

//================== 非二進制校驗矩陣 Hₛ 與其二進位展開 ===================

void buildNonbinaryHs(unsigned char Hs[nn - kk][nn])
{
    int i, j;
    for (i = 0; i < nn - kk; i++) {
        for (j = 0; j < nn; j++) {
            int exp = ((i + 1) * j) % m;
            if (j == 0)
                Hs[i][j] = 1;
            else
                Hs[i][j] = alpha_to[exp];
        }
    }
}

void companionMatrix(unsigned char a, unsigned char MComp[][mm])
{
    unsigned char basis[mm];
    int c, r;
    for (c = 0; c < mm; c++) {
        basis[c] = alpha_to[c];
    }
    for (c = 0; c < mm; c++) {
        unsigned char prod = gf_mul(a, basis[c]);
        for (r = 0; r < mm; r++) {
            MComp[r][c] = (prod >> r) & 1;
        }
    }
}

void buildRSBinaryParityCheckMatrix()
{
    int i, j, r, c;
    unsigned char Hs[nn - kk][nn];
    buildNonbinaryHs(Hs);
    for (i = 0; i < PARITY_BITS; i++) {
        for (j = 0; j < TOTAL_BITS; j++) {
            HBit[i][j] = 0;
        }
    }
    for (i = 0; i < nn - kk; i++) {
        for (j = 0; j < nn; j++) {
            unsigned char MComp[mm][mm];
            companionMatrix(Hs[i][j], MComp);
            for (r = 0; r < mm; r++) {
                for (c = 0; c < mm; c++) {
                    HBit[i * mm + r][j * mm + c] = MComp[r][c];
                }
            }
        }
    }
}

//================== 信道模擬與 ABP 軟解碼 ===================

static double rx[TOTAL_BITS];
static double LLR[TOTAL_BITS];

void bpsk_modulation(const unsigned char* bits, double snr_db)
{
    double snr_lin = pow(10.0, snr_db / 10.0);
    double sigma = sqrt(1.0 / (2.0 * snr_lin));
    int i;
    for (i = 0; i < TOTAL_BITS; i++) {
        double tx_val = (bits[i] == 0) ? +1.0 : -1.0;
        double u1 = (double)rand() / ((double)RAND_MAX + 1);
        double u2 = (double)rand() / ((double)RAND_MAX + 1);
        if (u1 < 1e-15) u1 = 1e-15;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        rx[i] = tx_val + sigma * z;
    }
}

void computeInitialLLR(double snr_db)
{
    double snr_lin = pow(10.0, snr_db / 10.0);
    double sigma_sq = 1.0 / (2.0 * snr_lin);
    int i;
    for (i = 0; i < TOTAL_BITS; i++) {
        LLR[i] = (2.0 * rx[i]) / sigma_sq;
    }
}

//================== ABP 軟解碼相關函式 ===================

// 這裡採用預先計算 tanh(L/2) 與 OpenMP 平行化內層迴圈優化 SPA 更新

// 修改參數設定
#define UPDATE_THRESHOLD 2.0   // 降低更新門檻
#define DAMPING_START 0.95    // 提高初始 damping 係數
#define DAMPING_MIN 0.5      // 提高最小 damping 係數

// 新增動態調整參數
#define SNR_SCALING_FACTOR 0.5  // SNR 校正因子

// Adjust ABP parameters for better performance around threshold
#define SNR_THRESHOLD 8.0     // Keep the threshold
#define SNR_MARGIN 0.5       // Add margin for smoother transition
#define MAX_ITER_HIGH_SNR 5  // Increase from 3 to 5 for better convergence
#define MAX_ITER_LOW_SNR 20  // Increase from 15 to 20 for more reliability
#define UPDATE_THRESHOLD 3.0  // Lower threshold for more updates
#define DAMPING_START 0.90   // More conservative damping
#define DAMPING_MIN 0.4      // Higher minimum damping

// 新增高速收斂的定義參數
#define MAX_ITER_HIGH_SNR 10    // 提高到10以確保收斂
#define MAX_ITER_LOW_SNR 30     // 低SNR時允許更多迭代
#define EARLY_TERMIN_THRESHOLD 0.001  // 提前結束閾值
#define FORCE_UPDATE_RATIO 0.15       // 強制更新比例
#define ALPHA_MINSUM 0.85            // min-sum 校正因子
#define AGGRESSIVE_DAMPING 0.99      // 積極的初始 damping

// Add transition zone handling
void spa_update(double* L)
{
    static double current_damping = AGGRESSIVE_DAMPING;
    static int iter_count = 0;
    iter_count++;

    // 分配外部信息陣列並初始化為零
    double* E = (double*)calloc(TOTAL_BITS, sizeof(double));
    double T[TOTAL_BITS];
    int i, r, c;

    // 前三次迭代使用極高的 damping 係數以加速收斂
    double active_damping = (iter_count <= 3) ? AGGRESSIVE_DAMPING : current_damping;

    // 根據SNR預計算最佳參數
    double llr_limit = (SNR_DB >= 8.0) ? 20.0 : 12.0;

    // 預計算 tanh 值 (高效實現)
#pragma omp parallel for
    for (i = 0; i < TOTAL_BITS; i++) {
        double l_half = L[i] / 2.0;
        if (fabs(l_half) > llr_limit) {
            l_half = (l_half > 0) ? llr_limit : -llr_limit;
        }
        T[i] = tanh(l_half);
    }

    // 改進的 SPA 更新：使用 min-sum 和並行處理
#pragma omp parallel for private(r, c) schedule(dynamic, 8)
    for (r = 0; r < PARITY_BITS; r++) {
        int idxs[TOTAL_BITS];
        int cnt = 0;

        // 收集非零位置 (校驗節點的連接位元)
        for (c = 0; c < TOTAL_BITS; c++) {
            if (HBit[r][c])
                idxs[cnt++] = c;
        }

        if (cnt == 0) continue;

        // 針對每個位元計算外部信息
        for (int idx = 0; idx < cnt; idx++) {
            int iBit = idxs[idx];

            // 使用優化的 min-sum 近似
            int sign = 1;
            double min_abs = 1e10;
            double second_min_abs = 1e10;

            for (int j = 0; j < cnt; j++) {
                if (j == idx) continue;

                double tj = T[idxs[j]];
                double abs_tj = fabs(tj);
                sign *= (tj >= 0) ? 1 : -1;

                // 追踪最小和次小絕對值
                if (abs_tj < min_abs) {
                    second_min_abs = min_abs;
                    min_abs = abs_tj;
                }
                else if (abs_tj < second_min_abs) {
                    second_min_abs = abs_tj;
                }
            }

            // 使用標準化的 min-sum 近似和校正因子
            double eVal = 2.0 * sign * (ALPHA_MINSUM * min_abs + 0.05 * second_min_abs);

            // 限制過大的更新
            if (eVal > 25.0) eVal = 25.0;
            if (eVal < -25.0) eVal = -25.0;

#pragma omp atomic
            E[iBit] += eVal;
        }
    }

    // 對 LLR 排序，找出最需更新的位元
    typedef struct { int idx; double ext_value; } ExtInfo;
    ExtInfo* ext_sorted = (ExtInfo*)malloc(TOTAL_BITS * sizeof(ExtInfo));

    for (i = 0; i < TOTAL_BITS; i++) {
        ext_sorted[i].idx = i;
        ext_sorted[i].ext_value = fabs(E[i]);
    }

    // 只排序前 N 個 (部分排序，提高效率)
    int force_update_count = (int)(TOTAL_BITS * FORCE_UPDATE_RATIO);
    for (i = 0; i < force_update_count; i++) {
        int max_idx = i;
        for (int j = i + 1; j < TOTAL_BITS; j++) {
            if (ext_sorted[j].ext_value > ext_sorted[max_idx].ext_value) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            ExtInfo temp = ext_sorted[i];
            ext_sorted[i] = ext_sorted[max_idx];
            ext_sorted[max_idx] = temp;
        }
    }

    // 自適應 LLR 更新策略
    int reliable_bits = 0;
    for (i = 0; i < TOTAL_BITS; i++) {
        double abs_llr = fabs(L[i]);
        bool force_update = false;

        // 檢查是否是強制更新的位元
        for (int j = 0; j < force_update_count; j++) {
            if (ext_sorted[j].idx == i) {
                force_update = true;
                break;
            }
        }

        // 初始迭代、不可靠位元和強制更新位元都要更新
        if (abs_llr < UPDATE_THRESHOLD || force_update || iter_count <= 3) {
            // 使用不同的更新步長
            double update_scale = (force_update || iter_count <= 3) ? 1.2 : 1.0;
            double new_llr = L[i] + active_damping * E[i] * update_scale;

            // 依 SNR 使用不同限制
            double llr_bound = (SNR_DB >= 8.0) ? 30.0 : 15.0;
            if (new_llr > llr_bound) new_llr = llr_bound;
            if (new_llr < -llr_bound) new_llr = -llr_bound;

            L[i] = new_llr;
        }
        else {
            reliable_bits++;
        }
    }

    free(ext_sorted);

    // 動態調整 damping 係數
    double reliability_ratio = (double)reliable_bits / TOTAL_BITS;

    // 前期保持高 damping，後期根據收斂情況調整
    if (iter_count < 4) {
        current_damping = AGGRESSIVE_DAMPING;
    }
    else {
        // 低信噪比時使用更積極的 damping 策略
        if (SNR_DB < 7.0) {
            current_damping = AGGRESSIVE_DAMPING - (AGGRESSIVE_DAMPING - 0.7) * reliability_ratio * 0.5;
        }
        else {
            current_damping = AGGRESSIVE_DAMPING - (AGGRESSIVE_DAMPING - 0.6) * reliability_ratio * 0.8;
        }
    }

    free(E);
}

// 改進 abp_decode 函數，實現更有效的停止條件
void abp_decode()
{
    int iter;
    int max_iter = (SNR_DB >= SNR_THRESHOLD) ? MAX_ITER_HIGH_SNR : MAX_ITER_LOW_SNR;
    double prev_error = 1e9;
    double prev_parity_fails = PARITY_BITS;
    int stall_counter = 0;

    // 初始一次 adapt_H 可以提高初始解碼效率
    adapt_H(LLR);

    for (iter = 0; iter < max_iter; iter++) {
        // 記錄更新前狀態
        double current_error = 0;
        for (int i = 0; i < TOTAL_BITS; i++) {
            current_error += fabs(LLR[i]);
        }

        // 執行核心 SPA 更新
        spa_update(LLR);

        // 每隔幾次迭代再做 adapt_H (避免過度消元)
        if (iter % 2 == 0) {
            adapt_H(LLR);
        }

        // 計算目前奇偶校驗失敗數
        int parity_fails = 0;
        for (int r = 0; r < PARITY_BITS; r++) {
            int sum = 0;
            for (int c = 0; c < TOTAL_BITS; c++) {
                if (HBit[r][c]) {
                    int dec = (LLR[c] >= 0) ? 0 : 1;
                    sum ^= dec;
                }
            }
            if (sum != 0)
                parity_fails++;
        }

        // 檢查全部校驗方程是否滿足
        if (parity_fails == 0) {
            printf("ABP 在第 %d 次迭代成功收斂 (所有校驗方程皆滿足)\n", iter + 1);
            break;
        }

        // 檢查解碼進度
        double error_change = fabs(current_error - prev_error) / (current_error + 1e-10);
        double parity_change = fabs(parity_fails - prev_parity_fails) / (PARITY_BITS + 1e-10);

        // 更靈活的收斂判定條件
        if ((error_change < EARLY_TERMIN_THRESHOLD && parity_change < 0.01) && iter > 3) {
            stall_counter++;
        }
        else {
            stall_counter = 0;
        }

        // 連續多次無明顯進展則提前結束
        if (stall_counter >= 3) {
            printf("ABP 在第 %d 次迭代停止 (解碼停滯，還有 %d 個校驗失敗)\n",
                iter + 1, parity_fails);
            break;
        }

        prev_error = current_error;
        prev_parity_fails = parity_fails;
    }

    // 標準輸出結果
    int s, i;
    printf("ABP Decoded codeword (我們的方法):\n");
    for (s = 0; s < RS_SYM; s++) {
        int val = 0;
        for (i = 0; i < mm; i++) {
            int bitIdx = s * mm + i;
            int dec = (LLR[bitIdx] >= 0) ? 0 : 1;
            val |= (dec << i);
        }
        printf("%03x ", val & ((1 << mm) - 1));
    }
    printf("\n");
}

typedef struct {
    int idx;
    double rel;
} BitRel;

int cmpRel(const void* a, const void* b)
{
    double ra = ((BitRel*)a)->rel;
    double rb = ((BitRel*)b)->rel;
    if (ra < rb) return -1;
    if (ra > rb) return 1;
    return 0;
}

// adapt_H() 對低可靠性位所在的 HBit 區域進行局部高斯消元，並平行化內層 XOR 消元
void adapt_H(double* L)
{
    BitRel arr[TOTAL_BITS];
    int i;

    // 改進可靠性計算
    for (i = 0; i < TOTAL_BITS; i++) {
        arr[i].idx = i;
        double abs_llr = fabs(L[i]);
        // 使用sigmoid函數調整可靠性度量
        arr[i].rel = 1.0 / (1.0 + exp(-abs_llr));
        if (abs_llr > UPDATE_THRESHOLD) {
            arr[i].rel += 1.0;  // 給予高信賴度比特更大權重
        }
    }

    qsort(arr, TOTAL_BITS, sizeof(BitRel), cmpRel);

    // 優化局部高斯消元
    int processed[TOTAL_BITS] = { 0 };
    int k;
    for (k = 0; k < PARITY_BITS; k++) {
        int best_col = -1;
        int best_pivot = -1;
        double best_rel = -1.0;

        // 尋找最佳樞紐元素
        for (i = 0; i < TOTAL_BITS; i++) {
            if (processed[arr[i].idx]) continue;
            int col = arr[i].idx;
            int r;
            for (r = k; r < PARITY_BITS; r++) {
                if (HBit[r][col] && arr[i].rel > best_rel) {
                    best_col = col;
                    best_pivot = r;
                    best_rel = arr[i].rel;
                }
            }
            if (best_col >= 0) break;
        }

        if (best_col < 0) continue;

        processed[best_col] = 1;

        // 行交換
        if (best_pivot != k) {
            for (i = 0; i < TOTAL_BITS; i++) {
                int temp = HBit[k][i];
                HBit[k][i] = HBit[best_pivot][i];
                HBit[best_pivot][i] = temp;
            }
        }

        // 消元
#pragma omp parallel for schedule(static)
        for (int rr = k + 1; rr < PARITY_BITS; rr++) {
            if (HBit[rr][best_col]) {
                for (int cc = 0; cc < TOTAL_BITS; cc++) {
                    HBit[rr][cc] ^= HBit[k][cc];
                }
            }
        }
    }
}

bool check_parity(const double* L)
{
    int r, c;
    for (r = 0; r < PARITY_BITS; r++) {
        int sum = 0;
        for (c = 0; c < TOTAL_BITS; c++) {
            if (HBit[r][c]) {
                int dec = (L[c] >= 0) ? 0 : 1;
                sum ^= dec;
            }
        }
        if (sum != 0)
            return false;
    }
    return true;
}

//================== 主函式 ===================

int main()
{
    int trial;
    int abp_success_count = 0;          // ABP 譯碼完全正確的試驗數
    int abp_total_symbol_err = 0;       // ABP 總符號錯誤數
    int abp_total_bit_err = 0;          // ABP 總位元錯誤數

    // 新增 RS 代數譯碼（學長版）錯誤統計變數
    int prof_success_count = 0;
    int prof_total_symbol_err = 0;
    int prof_total_bit_err = 0;

    srand((unsigned)time(NULL));

    int NUM_TRIALS_local = NUM_TRIALS;

    for (trial = 0; trial < NUM_TRIALS_local; trial++) {
        // 每次試驗前重建 GF 與生成 RS 生成多項式
        generate_gf();
        gen_poly();

        int i;
        for (i = 0; i < kk; i++) {
            data[i] = rand() % (1 << mm);
        }

        // 使用 unsigned short 以存放 mm 位元符號（例如 mm=10）
        unsigned short rs_codeword[RS_SYM];
        encode_rs();
        for (i = 0; i < nn - kk; i++) {
            rs_codeword[i] = bb[i];
        }
        for (i = 0; i < kk; i++) {
            rs_codeword[i + nn - kk] = data[i];
        }

        printf("Trial %d:\n", trial + 1);
        printf("原始 RS 碼字 (學長編碼):\n");
        for (i = 0; i < RS_SYM; i++) {
            printf("%03x ", rs_codeword[i] & ((1 << mm) - 1));
        }
        printf("\n");

        // 將 RS 碼字展開為二進位陣列 bits[]（每符號展開為 mm 位）
        unsigned char bits[TOTAL_BITS];
        int s, j;
        for (s = 0; s < RS_SYM; s++) {
            unsigned short sym = rs_codeword[s];
            for (j = 0; j < mm; j++) {
                bits[s * mm + j] = (sym >> j) & 1;
            }
        }

        // 信道模擬：BPSK 調變後加入 AWGN 噪聲
        bpsk_modulation(bits, SNR_DB);
        computeInitialLLR(SNR_DB);

        // RS 代數譯碼（學長版）：先進行硬判決，再組成符號
        unsigned char hard_bits[TOTAL_BITS];
        for (i = 0; i < TOTAL_BITS; i++) {
            hard_bits[i] = (rx[i] >= 0) ? 0 : 1;
        }
        unsigned short recd_sym[RS_SYM];
        for (s = 0; s < RS_SYM; s++) {
            int sym = 0;
            for (j = 0; j < mm; j++) {
                sym |= (hard_bits[s * mm + j] << j);
            }
            recd_sym[s] = sym;
        }
        for (i = 0; i < nn; i++) {
            recd[i] = index_of[recd_sym[i]];
        }
        decode_rs();
        unsigned short rs_decoded[RS_SYM];
        for (i = 0; i < nn; i++) {
            rs_decoded[i] = recd[i] & ((1 << mm) - 1);
        }
        printf("RS 代數譯碼 (學長版):\n");
        for (i = 0; i < RS_SYM; i++) {
            printf("%03x ", rs_decoded[i]);
        }
        printf("\n");

        // 新增：計算 RS 代數譯碼（學長版）的 FER 與 BER
        int prof_symbol_err = 0, prof_bit_err = 0;
        for (s = 0; s < RS_SYM; s++) {
            int diff_sym = rs_decoded[s] ^ rs_codeword[s];
            if (diff_sym != 0)
                prof_symbol_err++;
            while (diff_sym) {
                prof_bit_err += (diff_sym & 1);
                diff_sym >>= 1;
            }
        }
        if (prof_symbol_err == 0)
            prof_success_count++;
        prof_total_symbol_err += prof_symbol_err;
        prof_total_bit_err += prof_bit_err;

        // 構造二進位校驗矩陣 HBit
        buildRSBinaryParityCheckMatrix();

        // 呼叫 ABP 軟解碼
        abp_decode();

        // 計算 ABP 解碼結果與原始 RS 碼字之間的錯誤數
        int abp_symbol_err = 0, abp_bit_err = 0;
        unsigned short abp_decoded[RS_SYM];
        for (s = 0; s < RS_SYM; s++) {
            int val = 0;
            for (j = 0; j < mm; j++) {
                int bitIdx = s * mm + j;
                int dec = (LLR[bitIdx] >= 0) ? 0 : 1;
                val |= (dec << j);
            }
            abp_decoded[s] = val & ((1 << mm) - 1);
            int diff_sym = abp_decoded[s] ^ rs_codeword[s];
            if (diff_sym != 0)
                abp_symbol_err++;
            while (diff_sym) {
                abp_bit_err += (diff_sym & 1);
                diff_sym >>= 1;
            }
        }
        if (abp_symbol_err == 0)
            abp_success_count++;
        abp_total_symbol_err += abp_symbol_err;
        abp_total_bit_err += abp_bit_err;
        printf("\n-----------------------------------------\n");
    }

    double prof_FER = (double)(NUM_TRIALS_local - prof_success_count) / NUM_TRIALS_local;
    double prof_BER = (double)prof_total_bit_err / (NUM_TRIALS_local * TOTAL_BITS);
    double abp_FER = (double)(NUM_TRIALS_local - abp_success_count) / NUM_TRIALS_local;
    double abp_BER = (double)abp_total_bit_err / (NUM_TRIALS_local * TOTAL_BITS);

    printf("\n=== 模擬結果 ===\n");
    printf("試驗總數: %d\n", NUM_TRIALS_local);

    // 輸出 RS 代數譯碼 (學長版) 的結果
    printf("\n-- RS 代數譯碼 (學長版) --\n");
    printf("修正成功率: %.2f%%\n", (double)prof_success_count / NUM_TRIALS_local * 100.0);
    printf("總錯誤符號數: %d\n", prof_total_symbol_err);
    printf("平均每次試驗錯誤符號數: %.2f\n", (double)prof_total_symbol_err / NUM_TRIALS_local);
    printf("平均幀錯誤率 (FER): %.6f\n", prof_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", prof_BER);

    // 輸出 ABP 軟解碼 (我們的方法) 的結果
    printf("\n-- ABP 軟解碼 (我們的方法) --\n");
    printf("修正成功率: %.2f%%\n", (double)abp_success_count / NUM_TRIALS_local * 100.0);
    printf("總錯誤符號數: %d\n", abp_total_symbol_err);
    printf("平均每次試驗錯誤符號數: %.2f\n", (double)abp_total_symbol_err / NUM_TRIALS_local);
    printf("平均幀錯誤率 (FER): %.6f\n", abp_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", abp_BER);

    return 0;
}
