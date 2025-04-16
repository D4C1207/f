//================== 標準函式庫 ===================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//================== 可調 RS 參數設定 ===================
#define mm 6                  // 每個符號的位數
#define m ((1 << mm) - 1)     // GF(2^mm) 的非零元素個數
#define nn 63                 // RS 碼長（符號數）
#define kk 55                 // RS 資料符號數
#define tt ((nn - kk) / 2)    // 可糾正符號數

//================== ADP 參數設定 ===================
#define RS_SYM nn                           // RS 碼符號數
#define RS_DAT kk                           // RS 資料符號數
#define TOTAL_BITS (RS_SYM * mm)              // 總二進位位數
#define PARITY_BITS ((RS_SYM - RS_DAT) * mm)   // 校驗位數
#define MAX_ITER 20      // 最大迭代次數
#define DAMPING  0.15  // 阻尼係數
double SNR_DB = 5.0;     // 信噪比 (dB)
#define NUM_TRIALS 1// 模擬試驗次數
// 用於partial更新中更新的不可靠位元數量（可根據實驗調整）
#define PARTIAL_UPDATE_COUNT 240
// 分組次數（grouping）－論文中建議多次試驗
#define NUM_GROUPS 10

//================== GF 與 RS 編碼／譯碼全域變數 ===================
int pp[mm + 1] = { 1, 1, 0, 0, 0, 0, 1 }; // 生成多項式：x^6 + x + 1
int alpha_to[m + 1], index_of[m + 1], gg[nn - kk + 1];
int recd[nn], data[kk], bb[nn - kk];
unsigned short rs_codeword[RS_SYM]; // 正確RS碼字（學長版）

// HBit 為二進位校驗矩陣，共 PARITY_BITS x TOTAL_BITS
unsigned char HBit[PARITY_BITS][TOTAL_BITS];

// 接收信號與 LLR 向量
static double rx[TOTAL_BITS];
static double LLR[TOTAL_BITS];

//----- 用於位元不可靠性排序的結構 -----
typedef struct {
    int idx;
    double absLLR;
} BitInfo;

// 提前宣告函式
void decode_rs();
unsigned char gf_mul(unsigned char a, unsigned char b);
bool check_parity(const double* L);
void abp_decode();
void hdd_baseline(unsigned char* bits, unsigned short* decoded);

//================== GF 生成 ===================
void generate_gf() {
    int i;
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

//================== RS 生成多項式 ===================
void gen_poly() {
    int i, j;
    gg[0] = 2; // 從 α^2 開始
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

//================== GF 乘法 ===================
unsigned char gf_mul(unsigned char a, unsigned char b) {
    if (a == 0 || b == 0) return 0;
    int sum = index_of[a] + index_of[b];
    return alpha_to[sum % m];
}

//================== Companion Matrix 展開 ===================
void companionMatrix(unsigned char a, unsigned char MComp[][mm]) {
    for (int c = 0; c < mm; c++) {
        unsigned char product = gf_mul(a, alpha_to[c]);
        for (int r = 0; r < mm; r++) {
            MComp[r][c] = (product >> r) & 1;
        }
    }
}

//================== 非二進位 Hs 生成 ===================
void buildNonbinaryHs(unsigned char Hs[nn - kk][nn]) {
    for (int i = 0; i < nn - kk; i++) {
        for (int j = 0; j < nn; j++) {
            if (i == 0)
                Hs[i][j] = 1;
            else {
                int exp = (i * j) % m;
                Hs[i][j] = alpha_to[exp];
            }
        }
    }
}

//================== 傳統 RS Binary Parity Check 矩陣建構 ===================
void buildRSBinaryParityCheckMatrix() {
    unsigned char Hs[nn - kk][nn];
    buildNonbinaryHs(Hs);
    for (int i = 0; i < PARITY_BITS; i++)
        for (int j = 0; j < TOTAL_BITS; j++)
            HBit[i][j] = 0;
    for (int i = 0; i < nn - kk; i++) {
        for (int j = 0; j < nn; j++) {
            unsigned char MComp[mm][mm];
            companionMatrix(Hs[i][j], MComp);
            for (int r = 0; r < mm; r++)
                for (int c = 0; c < mm; c++)
                    HBit[i * mm + r][j * mm + c] = MComp[r][c];
        }
    }
}

// ========== 論文版 baseline HDD（bit-wise 判斷 + symbol 重建）==========
void hdd_baseline(unsigned char* bits, unsigned short* decoded) {
    for (int s = 0; s < RS_SYM; s++) {
        int val = 0;
        for (int j = 0; j < mm; j++) {
            val |= (bits[s * mm + j] << j);
        }
        decoded[s] = val & ((1 << mm) - 1);
    }
}

//================== 符號層級適應的 HBit 建構（含 grouping） ===================
void buildGroupedSymbolLevelParityCheckMatrix(double* LLR, int group) {
    double symRel[RS_SYM];
    int symIdx[RS_SYM];
    for (int i = 0; i < RS_SYM; i++) {
        double minLLR = 1e9;
        for (int j = 0; j < mm; j++) {
            double absLLR = fabs(LLR[i * mm + j]);
            if (absLLR < minLLR) minLLR = absLLR;
        }
        symRel[i] = minLLR;
        symIdx[i] = i;
    }
    // bubble sort: 升序排列（不可靠性高者在前）
    for (int i = 0; i < RS_SYM - 1; i++) {
        for (int j = i + 1; j < RS_SYM; j++) {
            if (symRel[symIdx[i]] > symRel[symIdx[j]]) {
                int tmp = symIdx[i];
                symIdx[i] = symIdx[j];
                symIdx[j] = tmp;
            }
        }
    }
    // 若 group > 0，隨機交換部分邊界符號（大約10%）
    if (group > 0) {
        int numSwap = (nn - kk) / 10;
        if (numSwap < 1) numSwap = 1;
        for (int i = 0; i < numSwap; i++) {
            int idx1 = rand() % (nn - kk);
            int idx2 = (nn - kk) + (rand() % (RS_SYM - (nn - kk)));
            int tmp = symIdx[idx1];
            symIdx[idx1] = symIdx[idx2];
            symIdx[idx2] = tmp;
        }
    }
    // 標記不可靠符號
    bool inSL[RS_SYM] = { false };
    for (int i = 0; i < (nn - kk); i++) {
        inSL[symIdx[i]] = true;
    }
    // 建立非二進位符號層級校驗矩陣 Hs_sym
    unsigned char Hs_sym[nn - kk][nn];
    for (int i = 0; i < nn - kk; i++) {
        for (int j = 0; j < nn; j++) {
            if (inSL[j]) {
                int pos = -1;
                for (int k = 0; k < (nn - kk); k++) {
                    if (symIdx[k] == j) { pos = k; break; }
                }
                Hs_sym[i][j] = (i == pos) ? 1 : 0;
            }
            else {
                Hs_sym[i][j] = (i == 0) ? 1 : alpha_to[((i * j) % m)];
            }
        }
    }
    // 展開成二進位校驗矩陣 HBit
    for (int i = 0; i < PARITY_BITS; i++)
        for (int j = 0; j < TOTAL_BITS; j++)
            HBit[i][j] = 0;
    for (int i = 0; i < nn - kk; i++) {
        for (int j = 0; j < nn; j++) {
            unsigned char MComp[mm][mm];
            companionMatrix(Hs_sym[i][j], MComp);
            for (int r = 0; r < mm; r++)
                for (int c = 0; c < mm; c++)
                    HBit[i * mm + r][j * mm + c] = MComp[r][c];
        }
    }
    // 印出 HBit 矩陣前5列前20欄（供檢查用）
    printf("\n== 檢查 HBit 矩陣前 5 列前 20 欄 (Group %d)==\n", group);
    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < 20; c++)
            printf("%d", HBit[r][c]);
        printf(" ...\n");
    }
}

//================== RS 編碼 ===================
void encode_rs() {
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

//================== RS 代數譯碼 (Berlekamp–Massey) ===================
void decode_rs() {
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

//================== BPSK 調變 ===================
void bpsk_modulation(const unsigned char* bits, double snr_db) {
    double snr_lin = pow(10.0, snr_db / 10.0);
    double sigma = sqrt(1.0 / (2.0 * snr_lin));
    for (int i = 0; i < TOTAL_BITS; i++) {
        double tx_val = bits[i] ? -1.0 : 1.0;
        // 高斯雜訊 (Box-Muller 方法)
        double u1 = ((double)rand() + 1) / ((double)RAND_MAX + 1);
        double u2 = ((double)rand() + 1) / ((double)RAND_MAX + 1);
        double z = sigma * sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
        rx[i] = tx_val + z;
    }
}

//================== 初始 LLR 計算 ===================
void computeInitialLLR(double snr_db) {
    double snr_lin = pow(10.0, snr_db / 10.0);
    double sigma_sq = 1.0 / (2.0 * snr_lin);
    for (int i = 0; i < TOTAL_BITS; i++)
        LLR[i] = 2.0 * rx[i] / sigma_sq;
}

//================== SPA 部分更新 ===================
int cmpBitInfo(const void* a, const void* b) {
    double d = ((BitInfo*)a)->absLLR - ((BitInfo*)b)->absLLR;
    return (d < 0) ? -1 : (d > 0);
}

void partial_spa_update(double* L) {
    double* E = (double*)calloc(TOTAL_BITS, sizeof(double));
    double T[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++)
        T[i] = tanh(L[i] / 2.0);
    // 對每一個校驗檢查行進行 SPA 更新
    for (int r = 0; r < PARITY_BITS; r++) {
        int idxs[mm * nn], cnt = 0;
        for (int c = 0; c < TOTAL_BITS; c++) {
            if (HBit[r][c])
                idxs[cnt++] = c;
        }
        for (int i = 0; i < cnt; i++) {
            double prod = 1.0;
            for (int j = 0; j < cnt; j++) {
                if (i != j)
                    prod *= T[idxs[j]];
            }
            if (prod > 0.999999) prod = 0.999999;
            if (prod < -0.999999) prod = -0.999999;
            double extr = 2.0 * atanh(prod);
            E[idxs[i]] += extr;
        }
    }
    // 僅更新 PARTIAL_UPDATE_COUNT 個較不可靠的位元
    BitInfo bits[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        bits[i].idx = i;
        bits[i].absLLR = fabs(L[i]);
    }
    qsort(bits, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    for (int i = 0; i < PARTIAL_UPDATE_COUNT; i++) {
        int idx = bits[i].idx;
        L[idx] += DAMPING * E[idx];
    }
    free(E);
}

//================== 局部高斯消去調整 HBit ===================
void adapt_H(double* L) {
    BitInfo unreliables[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        unreliables[i].idx = i;
        unreliables[i].absLLR = fabs(L[i]);
    }
    qsort(unreliables, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    for (int k = 0; k < PARITY_BITS; k++) {
        int col = unreliables[k].idx;
        int pivot = -1;
        for (int r = 0; r < PARITY_BITS; r++) {
            if (HBit[r][col]) { pivot = r; break; }
        }
        if (pivot < 0) continue;
        for (int r2 = 0; r2 < PARITY_BITS; r2++) {
            if (r2 == pivot) continue;
            if (HBit[r2][col]) {
                for (int c = 0; c < TOTAL_BITS; c++)
                    HBit[r2][c] ^= HBit[pivot][c];
            }
        }
    }
}

//================== 2-Degree Random Connection ===================
void degree2_random_connection(unsigned char H[][TOTAL_BITS], int rows) {
    std::vector<int> perm(rows);
    for (int i = 0; i < rows; i++) perm[i] = i;
    for (int i = rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (int i = 0; i < rows - 1; i++) {
        for (int c = 0; c < TOTAL_BITS; c++)
            H[perm[i]][c] ^= H[perm[i + 1]][c];
    }
}

//================== 校驗檢查 ===================
bool check_parity(const double* L) {
    for (int r = 0; r < PARITY_BITS; r++) {
        int sum = 0;
        for (int c = 0; c < TOTAL_BITS; c++) {
            if (HBit[r][c])
                sum ^= (L[c] < 0);
        }
        if (sum) return false;
    }
    return true;
}

//================== 翻轉位元嘗試（若迭代未收斂） ===================
void try_flipping_bits(double* L) {
    BitInfo bitInfos[TOTAL_BITS];
    bool hard_decision[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        bitInfos[i].idx = i;
        bitInfos[i].absLLR = fabs(L[i]);
        hard_decision[i] = (L[i] < 0);
    }
    qsort(bitInfos, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    int flip_candidates[10], count = 0;
    for (int i = 0; i < TOTAL_BITS && count < 5; i++) {
        int idx = bitInfos[i].idx;
        int hard = hard_decision[idx];
        if ((L[idx] > 0 && hard == 1) || (L[idx] < 0 && hard == 0))
            flip_candidates[count++] = idx;
    }
    for (int i = 0; i < TOTAL_BITS && count < 5; i++) {
        int idx = bitInfos[i].idx;
        bool ok = true;
        for (int j = 0; j < count; j++) {
            if (flip_candidates[j] == idx) { ok = false; break; }
        }
        if (ok) flip_candidates[count++] = idx;
    }
    for (int i = 0; i < count; i++) {
        for (int j = i; j < count; j++) {
            int idx1 = flip_candidates[i];
            int idx2 = flip_candidates[j];
            double orig1 = L[idx1], orig2 = L[idx2];
            L[idx1] *= -1;
            if (i != j) L[idx2] *= -1;
            if (check_parity(L)) {
                printf("\n>>> Flipping 成功！翻轉 idx=%d %d\n", idx1, (i != j) ? idx2 : -1);
                return;
            }
            L[idx1] = orig1;
            if (i != j) L[idx2] = orig2;
        }
    }
    printf(">>> Flipping 嘗試失敗\n");
}

//================== ABP 解碼主程序 ===================
void abp_decode() {
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        partial_spa_update(LLR);
        adapt_H(LLR);
        degree2_random_connection(HBit, PARITY_BITS);
        double avg_abs = 0.0;
        for (int i = 0; i < TOTAL_BITS; i++)
            avg_abs += fabs(LLR[i]);
        avg_abs /= TOTAL_BITS;
        printf("Parity check %s at iteration %d\n", check_parity(LLR) ? "passed" : "failed", iter);
        printf("Iter %d: Avg |LLR| = %.4f\n", iter, avg_abs);
        if (check_parity(LLR)) {
            printf("ABP 收斂於迭代 %d\n", iter);
            break;
        }
    }
    if (iter == MAX_ITER) {
        printf("ABP 迭代達到最大次數但未完全收斂\n");
        try_flipping_bits(LLR);
    }
    printf("ADP(20,3) 最終選擇結果 (bit errors 計算於此次結果):\n");
    unsigned short abp_decoded[RS_SYM];
    for (int s = 0; s < RS_SYM; s++) {
        int val = 0;
        for (int i = 0; i < mm; i++) {
            int bitIdx = s * mm + i;
            int dec = (LLR[bitIdx] >= 0) ? 0 : 1;
            val |= (dec << i);
        }
        abp_decoded[s] = val & ((1 << mm) - 1);
        printf("%03x ", abp_decoded[s]);
    }
    printf("\n");
}

//================== 主函式 ===================
int main() {
    int trial;
    int prof_success_count = 0, abp_success_count = 0;
    int prof_total_symbol_err = 0, prof_total_bit_err = 0;
    int abp_total_symbol_err = 0, abp_total_bit_err = 0;
    int hdd_success_count = 0, hdd_total_symbol_err = 0, hdd_total_bit_err = 0;
    srand((unsigned)time(NULL));
    int NUM_TRIALS_local = NUM_TRIALS;

    for (trial = 0; trial < NUM_TRIALS_local; trial++) {
        printf("\n================ Trial %d ================\n", trial + 1);
        generate_gf();
        gen_poly();
        // 產生隨機 RS 資料
        for (int i = 0; i < kk; i++)
            data[i] = rand() % (1 << mm);
        encode_rs();
        // 組成完整 RS 碼字：前 (nn-kk) 個為校驗位，後 kk 個為資料位
        for (int i = 0; i < nn - kk; i++) rs_codeword[i] = bb[i];
        for (int i = 0; i < kk; i++) rs_codeword[i + nn - kk] = data[i];
        printf("原始 RS 碼字 (學長編碼):\n");
        for (int i = 0; i < RS_SYM; i++)
            printf("%03x ", rs_codeword[i] & ((1 << mm) - 1));
        printf("\n");
        // 將 RS 碼字轉成 bit 串
        unsigned char bits[TOTAL_BITS];
        for (int s = 0; s < RS_SYM; s++) {
            for (int j = 0; j < mm; j++) {
                bits[s * mm + j] = (rs_codeword[s] >> j) & 1;
            }
        }
        // BPSK 調變與 AWGN 通道
        bpsk_modulation(bits, SNR_DB);
        computeInitialLLR(SNR_DB);
        // baseline HDD: 先產生硬決定結果
        unsigned char hard_bits[TOTAL_BITS];
        for (int i = 0; i < TOTAL_BITS; i++)
            hard_bits[i] = (rx[i] >= 0) ? 0 : 1;
        for (int s = 0; s < RS_SYM; s++) {
            int sym = 0;
            for (int j = 0; j < mm; j++) {
                sym |= (hard_bits[s * mm + j] << j);
            }
            recd[s] = index_of[sym];
        }
        decode_rs();
        unsigned short rs_decoded[RS_SYM];
        for (int i = 0; i < RS_SYM; i++) {
            rs_decoded[i] = recd[i] & ((1 << mm) - 1);
        }
        printf("RS 代數譯碼 (學長版):\n");
        for (int i = 0; i < RS_SYM; i++)
            printf("%03x ", rs_decoded[i]);
        printf("\n");
        // 統計 RS 代數譯碼錯誤
        int prof_symbol_err = 0, prof_bit_err = 0;
        for (int s = 0; s < RS_SYM; s++) {
            int diff = rs_decoded[s] ^ rs_codeword[s];
            if (diff != 0) prof_symbol_err++;
            while (diff) {
                prof_bit_err += (diff & 1);
                diff >>= 1;
            }
        }
        if (prof_symbol_err == 0) prof_success_count++;
        prof_total_symbol_err += prof_symbol_err;
        prof_total_bit_err += prof_bit_err;

        // ADP 軟解碼：進行 NUM_GROUPS 次 grouping，選最佳結果
        unsigned short best_adp[RS_SYM];
        int best_bit_err = TOTAL_BITS + 1;
        int best_group = -1;
        for (int grp = 0; grp < NUM_GROUPS; grp++) {
            double LLR_backup[TOTAL_BITS];
            for (int i = 0; i < TOTAL_BITS; i++) {
                LLR_backup[i] = LLR[i];
            }
            buildGroupedSymbolLevelParityCheckMatrix(LLR, grp);
            abp_decode();
            unsigned short adp_decoded[RS_SYM];
            for (int s = 0; s < RS_SYM; s++) {
                int val = 0;
                for (int j = 0; j < mm; j++) {
                    int bitIdx = s * mm + j;
                    int dec = (LLR[bitIdx] >= 0) ? 0 : 1;
                    val |= (dec << j);
                }
                adp_decoded[s] = val & ((1 << mm) - 1);
            }
            int curr_symbol_err = 0, curr_bit_err = 0;
            for (int s = 0; s < RS_SYM; s++) {
                int diff = adp_decoded[s] ^ rs_codeword[s];
                if (diff != 0) curr_symbol_err++;
                while (diff) { curr_bit_err += (diff & 1); diff >>= 1; }
            }
            printf("Group %d: bit errors = %d\n", grp, curr_bit_err);
            if (curr_bit_err < best_bit_err) {
                best_bit_err = curr_bit_err;
                best_group = grp;
                for (int s = 0; s < RS_SYM; s++)
                    best_adp[s] = adp_decoded[s];
            }
            for (int i = 0; i < TOTAL_BITS; i++) {
                LLR[i] = LLR_backup[i];
            }
        }
        printf("ADP(20,3) 最終選擇結果 (group %d, bit errors = %d):\n", best_group, best_bit_err);
        for (int s = 0; s < RS_SYM; s++) {
            printf("%03x ", best_adp[s]);
        }
        printf("\n");
        int adp_symbol_err = 0, adp_bit_err = 0;
        for (int s = 0; s < RS_SYM; s++) {
            int diff = best_adp[s] ^ rs_codeword[s];
            if (diff != 0) adp_symbol_err++;
            while (diff) { adp_bit_err += (diff & 1); diff >>= 1; }
        }
        if (adp_symbol_err == 0) abp_success_count++;
        abp_total_symbol_err += adp_symbol_err;
        abp_total_bit_err += adp_bit_err;

        // baseline HDD 結果統計
        unsigned short hdd_decoded[RS_SYM];
        hdd_baseline(hard_bits, hdd_decoded);
        printf("Baseline HDD 解碼結果:\n");
        for (int i = 0; i < RS_SYM; i++) {
            printf("%03x ", hdd_decoded[i]);
        }
        printf("\n");
        int curr_hdd_symbol_err = 0, curr_hdd_bit_err = 0;
        for (int s = 0; s < RS_SYM; s++) {
            int diff = hdd_decoded[s] ^ rs_codeword[s];
            if (diff != 0) curr_hdd_symbol_err++;
            while (diff) { curr_hdd_bit_err += (diff & 1); diff >>= 1; }
        }
        if (curr_hdd_symbol_err == 0) hdd_success_count++;
        hdd_total_symbol_err += curr_hdd_symbol_err;
        hdd_total_bit_err += curr_hdd_bit_err;

        printf("\n-----------------------------------------\n");
    }

    double prof_FER = (double)(NUM_TRIALS_local - prof_success_count) / NUM_TRIALS_local;
    double prof_BER = (double)prof_total_bit_err / (NUM_TRIALS_local * TOTAL_BITS);
    double abp_FER = (double)(NUM_TRIALS_local - abp_success_count) / NUM_TRIALS_local;
    double abp_BER = (double)abp_total_bit_err / (NUM_TRIALS_local * TOTAL_BITS);
    double hdd_FER = (double)(NUM_TRIALS_local - hdd_success_count) / NUM_TRIALS_local;
    double hdd_BER = (double)hdd_total_bit_err / (NUM_TRIALS_local * TOTAL_BITS);

    printf("\n-- Baseline HDD (論文) --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * hdd_success_count / NUM_TRIALS_local);
    printf("平均幀錯誤率 (FER): %.6f\n", hdd_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", hdd_BER);

    printf("\n-- RS 代數譯碼 (學長版) --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * prof_success_count / NUM_TRIALS_local);
    printf("平均幀錯誤率 (FER): %.6f\n", prof_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", prof_BER);

    printf("\n-- ADP(20,3) 軟解碼 --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * abp_success_count / NUM_TRIALS_local);
    printf("平均幀錯誤率 (FER): %.6f\n", abp_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", abp_BER);

    return 0;
}
