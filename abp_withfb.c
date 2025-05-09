//================== 標準函式庫 ===================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
double snr = 3.0;
//================== 可調 RS 參數設定 ===================
#define mm 4                  // 每個符號的位數
#define m ((1 << mm) - 1)     // GF(2^mm) 的非零元素個數
#define nn 63                 // RS 碼長（符號數）
#define kk 55                // RS 資料符號數
#define tt ((nn - kk) / 2)    // 可糾正符號數
double R = (double)kk / nn;  // RS(63,55) 的码率 55/63 ≈ 0.873

//================== ADP + OSD(1) 參數設定 ===================
#define RS_SYM nn                           // RS 碼符號數
#define RS_DAT kk                           // RS 資料符號數
#define TOTAL_BITS (RS_SYM * mm)            // 總二進位位數
#define PARITY_BITS ((RS_SYM - RS_DAT) * mm) // 校驗位數
#define MAX_ITER 20                         // 最大迭代次數
#define DAMPING  0.05                    // 阻尼係數
#define NUM_TRIALS 1000            // 模擬試驗次數
#define PARTIAL_UPDATE_COUNT TOTAL_BITS           // 部分更新位數
#define NUM_GROUPS 3                       // 分組數量
#define OSD_PERIOD 6                        // 初始機制周期
#define OSD_A 0.5                        // OSD(1) 反馈消息量
#define FLIP_F 1                         // FLIP(F) 翻轉位數

//================== GF 與 RS 編碼／譯碼全域變數 ===================
int pp[mm + 1] = { 1, 1, 0, 0, 0, 0, 1 }; // 生成多項式：x^6 + x + 1
//int pp[mm + 1] = { 1, 0, 0, 1, 1 };  // 生成多項式：x^4 + x^3 + 1

int alpha_to[m + 1], index_of[m + 1], gg[nn - kk + 1];
int recd[nn], data[kk], bb[nn - kk];
unsigned short rs_codeword[RS_SYM]; // 正確RS碼字

unsigned char HBit[PARITY_BITS][TOTAL_BITS]; // 二進位校驗矩陣
static double rx[TOTAL_BITS];    // 通道輸出
static double LLR[TOTAL_BITS];   // LLR 向量

typedef struct { int idx; double absLLR; } BitInfo;

// 函式宣告
void generate_gf();
void gen_poly();
unsigned char gf_mul(unsigned char a, unsigned char b);
void companionMatrix(unsigned char a, unsigned char MComp[][mm]);
void buildGroupedSymbolLevelParityCheckMatrix(double* LLR_in, int group);
void encode_rs();
void decode_rs();
void bpsk_modulation(const unsigned char* bits, double snr_db);
void computeInitialLLR(double snr_db);
int cmpBitInfo(const void* a, const void* b);
void partial_spa_update(double* L);
void adapt_H(double* L);
void degree2_random_connection(unsigned char H[][TOTAL_BITS], int rows);
bool check_parity(const double* L);
void try_flipping_bits(double* L);
void osd1_feedback(double* LLR);
void flip_F_bits(double* L, int F);
void hdd_baseline(unsigned char* bits, unsigned short* decoded);

//================== GF 生成 ===================
void generate_gf() {
    int i, mask = 1;
    alpha_to[mm] = 0;
    for (i = 0; i < mm; i++) {
        alpha_to[i] = mask;
        index_of[mask] = i;
        if (pp[i]) alpha_to[mm] ^= mask;
        mask <<= 1;
    }
    index_of[alpha_to[mm]] = mm;
    mask >>= 1;
    for (i = mm + 1; i < m; i++) {
        if (alpha_to[i - 1] & mask)
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
    gg[0] = 2; gg[1] = 1;
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
    if (!a || !b) return 0;
    int sum = index_of[a] + index_of[b];
    return alpha_to[sum % m];
}

//================== Companion Matrix 展開 ===================
void companionMatrix(unsigned char a, unsigned char MComp[][mm]) {
    for (int c = 0; c < mm; c++) {
        unsigned char prod = gf_mul(a, alpha_to[c]);
        for (int r = 0; r < mm; r++)
            MComp[r][c] = (prod >> r) & 1;
    }
}

//================== 符號層級適應 HBit 建構 ===================
void buildGroupedSymbolLevelParityCheckMatrix(double* LLR_in, int group) {
    double symRel[RS_SYM];
    int symIdx[RS_SYM];
    for (int i = 0; i < RS_SYM; i++) {
        double mn = 1e9;
        for (int j = 0; j < mm; j++)
            mn = fmin(mn, fabs(LLR_in[i * mm + j]));
        symRel[i] = mn; symIdx[i] = i;
    }
    for (int i = 0; i < RS_SYM - 1; i++)
        for (int j = i + 1; j < RS_SYM; j++)
            if (symRel[symIdx[i]] > symRel[symIdx[j]]) {
                int t = symIdx[i]; symIdx[i] = symIdx[j]; symIdx[j] = t;
            }
    if (group > 0) {
        int numSwap = (nn - kk) / 10; if (numSwap < 1) numSwap = 1;
        for (int i = 0; i < numSwap; i++) {
            int a = rand() % (nn - kk), b = (nn - kk) + rand() % (RS_SYM - (nn - kk));
            int t = symIdx[a]; symIdx[a] = symIdx[b]; symIdx[b] = t;
        }
    }
    bool inSL[RS_SYM] = { false };
    for (int i = 0; i < nn - kk; i++)
        inSL[symIdx[i]] = true;
    unsigned char Hs_sym[nn - kk][nn];
    for (int i = 0; i < nn - kk; i++)
        for (int j = 0; j < nn; j++)
            Hs_sym[i][j] = inSL[j] ? ((symIdx[i] == j) ? 1 : 0)
            : ((i == 0) ? 1 : alpha_to[(i * j) % m]);
    memset(HBit, 0, sizeof(HBit));
    for (int i = 0; i < nn - kk; i++)
        for (int j = 0; j < nn; j++) {
            unsigned char MComp[mm][mm];
            companionMatrix(Hs_sym[i][j], MComp);
            for (int r = 0; r < mm; r++)
                for (int c = 0; c < mm; c++)
                    HBit[i * mm + r][j * mm + c] = MComp[r][c];
        }
}

//================== Baseline HDD ===================
void hdd_baseline(unsigned char* bits, unsigned short* decoded) {
    for (int s = 0; s < RS_SYM; s++) {
        int v = 0;
        for (int j = 0; j < mm; j++)
            v |= bits[s * mm + j] << j;
        decoded[s] = v & ((1 << mm) - 1);
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
    double EbN0_lin = pow(10.0, snr_db / 10.0);   // now Eb/N0
    double EsN0_lin = R * EbN0_lin;             // Es/N0 = R · Eb/N0
    double sigma = sqrt(1.0 / (2.0 * EsN0_lin));
    for (int i = 0; i < TOTAL_BITS; i++) {
        double tx = bits[i] ? -1.0 : 1.0;
        double u1 = ((double)rand() + 1) / (RAND_MAX + 1);
        double u2 = ((double)rand() + 1) / (RAND_MAX + 1);
        rx[i] = tx + sigma * sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    }
}

//================== 初始 LLR 計算 ===================
void computeInitialLLR(double snr_db) {
    double EbN0_lin = pow(10.0, snr_db / 10.0);
    double EsN0_lin = R * EbN0_lin;
    double sigma_sq = 1.0 / (2.0 * EsN0_lin);
    for (int i = 0; i < TOTAL_BITS; i++)
        LLR[i] = 2.0 * rx[i] / sigma_sq;
}

//================== BitInfo 排序 cmp ===================
int cmpBitInfo(const void* a, const void* b) {
    double d = ((BitInfo*)a)->absLLR - ((BitInfo*)b)->absLLR;
    return (d < 0) ? -1 : (d > 0) ? 1 : 0;
}

//================== 部分 SPA 更新 ===================
void partial_spa_update(double* L) {
    double* E = (double*)calloc(TOTAL_BITS, sizeof(double));
    double T[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) T[i] = tanh(L[i] / 2.0);
    for (int r = 0; r < PARITY_BITS; r++) {
        int idxs[TOTAL_BITS], cnt = 0;
        for (int c = 0; c < TOTAL_BITS; c++) if (HBit[r][c]) idxs[cnt++] = c;
        for (int i = 0; i < cnt; i++) {
            double prod = 1.0;
            for (int j = 0; j < cnt; j++) if (i != j) prod *= T[idxs[j]];
            prod = fmin(fmax(prod, -0.999999), 0.999999);
            double extr = 2.0 * atanh(prod);
            E[idxs[i]] += extr;
        }
    }
    BitInfo bits[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        bits[i].idx = i; bits[i].absLLR = fabs(L[i]);
    }
    qsort(bits, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    for (int i = 0; i < PARTIAL_UPDATE_COUNT; i++) {
        int idx = bits[i].idx;
        L[idx] += DAMPING * E[idx];
    }
    free(E);
}

//================== 自適應高斯消去 ===================
void adapt_H(double* L) {
    BitInfo b[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        b[i].idx = i; b[i].absLLR = fabs(L[i]);
    }
    qsort(b, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    for (int k = 0; k < PARITY_BITS; k++) {
        int col = b[k].idx, pivot = -1;
        for (int r = 0; r < PARITY_BITS; r++) if (HBit[r][col]) { pivot = r; break; }
        if (pivot < 0) continue;
        for (int r2 = 0; r2 < PARITY_BITS; r2++) if (r2 != pivot && HBit[r2][col]) {
            for (int c = 0; c < TOTAL_BITS; c++) HBit[r2][c] ^= HBit[pivot][c];
        }
    }
}

//================== 2‑Degree 隨機連接 ===================
void degree2_random_connection(unsigned char H[][TOTAL_BITS], int rows) {
    std::vector<int> perm(rows);
    for (int i = 0; i < rows; i++) perm[i] = i;
    for (int i = rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(perm[i], perm[j]);
    }
    for (int i = 0; i < rows - 1; i++)
        for (int c = 0; c < TOTAL_BITS; c++)
            H[perm[i]][c] ^= H[perm[i + 1]][c];
}

//================== 校驗檢查 ===================
bool check_parity(const double* L) {
    for (int r = 0; r < PARITY_BITS; r++) {
        int sum = 0;
        for (int c = 0; c < TOTAL_BITS; c++)
            if (HBit[r][c] && (L[c] < 0)) sum ^= 1;
        if (sum) return false;
    }
    return true;
}

//================== 翻轉補救 ===================
void try_flipping_bits(double* L) {
    BitInfo b[TOTAL_BITS];
    bool hd[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        b[i].idx = i; b[i].absLLR = fabs(L[i]);
        hd[i] = (L[i] < 0);
    }
    qsort(b, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    int cand[10], cnt = 0;
    for (int i = 0; i < TOTAL_BITS && cnt < 5; i++) {
        int idx = b[i].idx;
        if ((L[idx] > 0 && hd[idx]) || (L[idx] < 0 && !hd[idx]))
            cand[cnt++] = idx;
    }
    for (int i = 0; i < TOTAL_BITS && cnt < 5; i++) {
        int idx = b[i].idx; bool ok = true;
        for (int j = 0; j < cnt; j++) if (cand[j] == idx) ok = false;
        if (ok) cand[cnt++] = idx;
    }
    for (int i = 0; i < cnt; i++) for (int j = i; j < cnt; j++) {
        int i1 = cand[i], i2 = (i == j ? -1 : cand[j]);
        double o1 = L[i1], o2 = (i2 < 0 ? 0 : L[i2]);
        L[i1] *= -1; if (i2 >= 0)L[i2] *= -1;
        if (check_parity(L)) { printf("Flipping 성공 %d %d\n", i1, i2); return; }
        L[i1] = o1; if (i2 >= 0)L[i2] = o2;
    }
    printf("Flipping 실패\n");
}

//================== OSD(1) 反馈机制 ===================
void osd1_feedback(double* L) {
    unsigned char hard[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) hard[i] = (L[i] < 0) ? 1 : 0;
    for (int s = 0; s < RS_SYM; s++) {
        int v = 0;
        for (int j = 0; j < mm; j++) v |= hard[s * mm + j] << j;
        recd[s] = index_of[v];
    }
    decode_rs();
    for (int s = 0; s < RS_SYM; s++) {
        unsigned short sym = recd[s] & ((1 << mm) - 1);
        for (int j = 0; j < mm; j++) {
            int idx = s * mm + j, bit = (sym >> j) & 1;
            L[idx] += (bit == 0 ? +OSD_A : -OSD_A);
        }
    }
}

//================== FLIP(F) 预翻转机制 ===================
void flip_F_bits(double* L, int F) {
    BitInfo b[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        b[i].idx = i; b[i].absLLR = fabs(L[i]);
    }
    qsort(b, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    for (int i = 0; i < F && i < TOTAL_BITS; i++) {
        int idx = b[i].idx; L[idx] *= -1;
    }
}

//================== ABP‑OSD 解碼主程序 ===================
void abp_decode() {
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        if (iter > 0 && iter % OSD_PERIOD == 0) {
            osd1_feedback(LLR);
            printf("[OSD-init] at iter %d\n", iter);
        }
        partial_spa_update(LLR);
        adapt_H(LLR);
        degree2_random_connection(HBit, PARITY_BITS);
        osd1_feedback(LLR);
        double avg = 0;
        for (int i = 0; i < TOTAL_BITS; i++) avg += fabs(LLR[i]);
        avg /= TOTAL_BITS;
        printf("Iter %2d: Avg|LLR|=%.4f, parity %s\n",
            iter, avg, check_parity(LLR) ? "OK" : "FAIL");
        if (check_parity(LLR)) {
            printf("Converged at %d\n", iter);
            break;
        }
    }
    if (iter == MAX_ITER) {
        printf("MaxIter reached, try flipping...\n");
        try_flipping_bits(LLR);
    }
}

//================== 主函式 ===================
int main() {
    srand(time(NULL));
    int prof_success = 0, hdd_success = 0, abp_success = 0;
    long prof_bit_err = 0, hdd_bit_err = 0, abp_bit_err = 0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        printf("\n=== Trial %d ===\n", trial + 1);
        generate_gf(); gen_poly();
        for (int i = 0; i < kk; i++) data[i] = rand() % (1 << mm);
        encode_rs();
        for (int i = 0; i < nn - kk; i++) rs_codeword[i] = bb[i];
        for (int i = 0; i < kk; i++) rs_codeword[i + nn - kk] = data[i];
        printf("Original: ");
        for (int s = 0; s < RS_SYM; s++) printf("%03x ", rs_codeword[s]);
        printf("\n");

        unsigned char bits[TOTAL_BITS];
        for (int s = 0; s < RS_SYM; s++)
            for (int j = 0; j < mm; j++)
                bits[s * mm + j] = (rs_codeword[s] >> j) & 1;
        bpsk_modulation(bits, snr);
        computeInitialLLR(snr);

        // Prof decode
        
        unsigned char prof_hard[TOTAL_BITS];
        for (int i = 0; i < TOTAL_BITS; i++)
            prof_hard[i] = (rx[i] >= 0 ? 0 : 1);

        // 2) 將每 mm 個 bit 組成一個符號，存到 recd[]
        for (int s = 0; s < RS_SYM; s++) {
            int v = 0;
            for (int j = 0; j < mm; j++) {
                v |= (prof_hard[s * mm + j] << j);
            }
            recd[s] = index_of[v];
        }

        decode_rs();
        unsigned short prof_dec[RS_SYM];
        int prof_err = 0;
        for (int s = 0; s < RS_SYM; s++) {
            prof_dec[s] = recd[s] & ((1 << mm) - 1);
            int diff = prof_dec[s] ^ rs_codeword[s];
            while (diff) { prof_err += (diff & 1); diff >>= 1; }
        }
        if (prof_err == 0) prof_success++;
        prof_bit_err += prof_err;
        printf("Prof: ");
        for (int s = 0; s < RS_SYM; s++) printf("%03x ", prof_dec[s]);
        printf("  err=%d\n", prof_err);

        // HDD baseline
        unsigned char hard_bits[TOTAL_BITS];
        for (int i = 0; i < TOTAL_BITS; i++) hard_bits[i] = (rx[i] >= 0 ? 0 : 1);
        unsigned short hdd_dec[RS_SYM];
        hdd_baseline(hard_bits, hdd_dec);
        int hdd_err = 0;
        for (int s = 0; s < RS_SYM; s++) {
            int diff = hdd_dec[s] ^ rs_codeword[s];
            while (diff) { hdd_err += (diff & 1); diff >>= 1; }
        }
        if (hdd_err == 0) hdd_success++;
        hdd_bit_err += hdd_err;
        printf("HDD:  ");
        for (int s = 0; s < RS_SYM; s++) printf("%03x ", hdd_dec[s]);
        printf("  err=%d\n", hdd_err);

        // ABP-OSD list decode
        double LLR_bk[TOTAL_BITS];
        memcpy(LLR_bk, LLR, sizeof(LLR));
        int best_err = TOTAL_BITS + 1, best_grp = 0;
        unsigned short best_dec[RS_SYM];
        for (int grp = 0; grp < NUM_GROUPS; grp++) {
            memcpy(LLR, LLR_bk, sizeof(LLR));
            buildGroupedSymbolLevelParityCheckMatrix(LLR, grp);
            flip_F_bits(LLR, FLIP_F);
            printf("[Grp %d] FLIP(%d)\n", grp, FLIP_F);
            abp_decode();
            unsigned short abp_dec[RS_SYM];
            int abp_err = 0;
            for (int s = 0; s < RS_SYM; s++) {
                int v = 0;
                for (int j = 0; j < mm; j++) {
                    int bit = (LLR[s * mm + j] < 0 ? 1 : 0);
                    v |= bit << j;
                }
                abp_dec[s] = v & ((1 << mm) - 1);
                int diff = abp_dec[s] ^ rs_codeword[s];
                while (diff) { abp_err += (diff & 1); diff >>= 1; }
            }
            printf("Grp %d err=%d\n", grp, abp_err);
            if (abp_err < best_err) {
                best_err = abp_err; best_grp = grp;
                memcpy(best_dec, abp_dec, sizeof(abp_dec));
            }
        }
        if (best_err == 0) abp_success++;
        abp_bit_err += best_err;
        printf("Best(g%d): ", best_grp);
        for (int s = 0; s < RS_SYM; s++) printf("%03x ", best_dec[s]);
        printf("  err=%d\n", best_err);
    }

    // 计算 FER 和 BER
    double hdd_FER = (double)(NUM_TRIALS - hdd_success) / NUM_TRIALS;
    double hdd_BER = (double)hdd_bit_err / (NUM_TRIALS * TOTAL_BITS);

    double prof_FER = (double)(NUM_TRIALS - prof_success) / NUM_TRIALS;
    double prof_BER = (double)prof_bit_err / (NUM_TRIALS * TOTAL_BITS);

    double abp_FER = (double)(NUM_TRIALS - abp_success) / NUM_TRIALS;
    double abp_BER = (double)abp_bit_err / (NUM_TRIALS * TOTAL_BITS);

    printf("\nsnr:%.2f\n",snr);
    printf("\n-- Baseline HDD (論文) --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * hdd_success / NUM_TRIALS);
    printf("平均幀錯誤率 (FER): %.6f\n", hdd_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", hdd_BER);

    printf("\n-- RS 代數譯碼 (學長版) --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * prof_success / NUM_TRIALS);
    printf("平均幀錯誤率 (FER): %.6f\n", prof_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", prof_BER);

    printf("\n-- ADP(20,3) 軟解碼 --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * abp_success / NUM_TRIALS);
    printf("平均幀錯誤率 (FER): %.6f\n", abp_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", abp_BER);
    return 0;
}

//尚未完成osd純軟體實作
