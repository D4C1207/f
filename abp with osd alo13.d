// === Segment 1: Includes, Macros, Globals ===

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// 基本参数
#define mm            6
#define m             ((1<<mm)-1)
#define nn            63
#define kk            55
#define tt            ((nn-kk)/2)
#define RS_SYM        nn
#define RS_DAT        kk
#define TOTAL_BITS    (RS_SYM*mm)
#define PARITY_BITS   ((RS_SYM-RS_DAT)*mm)

// 算法参数
#define MAX_ITER      20
#define DAMPING       0.8
#define OSD_PERIOD    6
#define OSD_A         0.5
#define OSD_K         (RS_DAT*mm)
#define HYB_DELTA     0.1
#define HYB_GROUPS    (PARITY_BITS/2)

// 仿真参数
#define NUM_TRIALS    100
#define SNR_DB        4.0
//snr=4 d=0.8
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GF 与 RS 多项式全局
int pp[mm + 1] = { 1, 1, 0, 0, 0, 0, 1 }; // 生成多項式：x^6 + x + 1
//int pp[mm + 1] = { 1,0,0,1,1 };               // x^4+x^3+1
int alpha_to[m + 1], index_of[m + 1], gg[nn - kk + 1];

// RS 缓冲
int data[kk], bb[nn - kk], recd[nn];

// 通道与 LLR
static double rx[TOTAL_BITS];
static double LLR[TOTAL_BITS];
static double LLR_init[TOTAL_BITS];

// Bit-level 矩阵
static unsigned char HBit[PARITY_BITS][TOTAL_BITS];
static unsigned char GBit[OSD_K][TOTAL_BITS];

// 混合列表行分组
static int hyb_row_perm[PARITY_BITS];
static bool hyb_row_perm_init = false;
// ==== OSD/Hybrid 用全域陣列（多用於大 for 迴圈與 memcpy）====
static unsigned char g_hard[TOTAL_BITS];
static unsigned char g_hard0[TOTAL_BITS];
static unsigned char g_cand[TOTAL_BITS];
static unsigned char g_best_hard[TOTAL_BITS];
static unsigned char g_code_sym[RS_SYM];
static unsigned char g_Gtmp[OSD_K][TOTAL_BITS];
static int g_MRIP[OSD_K];
static int g_vote0[TOTAL_BITS], g_vote1[TOTAL_BITS];

// === Segment 2: GF(2^m) Generation & RS Poly ===
// 在 #include 之后或全局变量声明之后
void encode_rs(void);
void gen_poly(void);

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

void gen_poly() {
    int i, j;
    gg[0] = 2; gg[1] = 1;
    for (i = 2; i <= nn - kk; i++) {
        gg[i] = 1;
        for (j = i - 1; j > 0; j--) {
            if (gg[j] != -1)
                gg[j] = gg[j - 1] ^ alpha_to[(index_of[gg[j]] + i) % m];
            else
                gg[j] = gg[j - 1];
        }
        gg[0] = alpha_to[(index_of[gg[0]] + i) % m];
    }
    for (i = 0; i <= nn - kk; i++)
        gg[i] = index_of[gg[i]];
}

unsigned char gf_mul(unsigned char a, unsigned char b) {
    if (!a || !b) return 0;
    int sum = index_of[a] + index_of[b];
    return alpha_to[sum % m];
}
// === Segment 3: Companion Matrix ===

void companionMatrix(unsigned char a, unsigned char MComp[][mm]) {
    for (int c = 0; c < mm; c++) {
        unsigned char prod = gf_mul(a, alpha_to[c]);
        for (int r = 0; r < mm; r++) {
            MComp[r][c] = (prod >> r) & 1;
        }
    }
}
// === Segment 4: Build HBit & GBit ===

// 4.1 HBit: 符号级 Hs -> bit-level HBit
void buildHBit() {
    unsigned char Hs[nn - kk][nn];
    for (int r = 0; r < nn - kk; r++) {
        for (int j = 0; j < nn; j++) {
            Hs[r][j] = (j == 0) ? 1 : alpha_to[((r + 1) * j) % m];
        }
    }
    unsigned char MComp[mm][mm];
    for (int r = 0; r < nn - kk; r++) {
        for (int j = 0; j < nn; j++) {
            companionMatrix(Hs[r][j], MComp);
            for (int br = 0; br < mm; br++)
                for (int bc = 0; bc < mm; bc++)
                    HBit[r * mm + br][j * mm + bc] = MComp[br][bc];
        }
    }
}

// 4.2 GBit: 系统化 G -> bit-level GBit
void buildGBit() {
    unsigned char code_sym[RS_SYM];
    unsigned char MComp[mm][mm];
    for (int i = 0; i < RS_DAT; i++) {
        for (int s = 0; s < RS_DAT; s++)
            data[s] = (s == i) ? 1 : 0;
        encode_rs();
        for (int j = 0; j < nn - kk; j++)
            code_sym[j] = bb[j];
        for (int j = 0; j < kk; j++)
            code_sym[j + nn - kk] = data[j];
        for (int j = 0; j < RS_SYM; j++) {
            unsigned char sym = code_sym[j] & ((1 << mm) - 1);
            companionMatrix(sym, MComp);
            for (int br = 0; br < mm; br++)
                for (int bc = 0; bc < mm; bc++)
                    GBit[i * mm + br][j * mm + bc] = MComp[br][bc];
        }
    }
}
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
void bpsk_modulation(const unsigned char* bits, double snr_db) {
    double EbN0 = pow(10.0, snr_db / 10.0);
    double EsN0 = (double)kk / nn * EbN0;
    double sigma = sqrt(1.0 / (2 * EsN0));
    for (int i = 0; i < TOTAL_BITS; i++) {
        double tx = bits[i] ? -1.0 : 1.0;
        double u1 = ((double)rand() + 1) / (RAND_MAX + 1);
        double u2 = ((double)rand() + 1) / (RAND_MAX + 1);
        rx[i] = tx + sigma * sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    }
}

void computeInitialLLR(double snr_db) {
    double EbN0 = pow(10.0, snr_db / 10.0);
    double EsN0 = (double)kk / nn * EbN0;
    double var = 1.0 / (2 * EsN0);
    for (int i = 0; i < TOTAL_BITS; i++)
        LLR[i] = 2.0 * rx[i] / var;
}
// === Segment 7: Sort, Adapt H, SPA update, Parity Check ===

typedef struct { int idx; double absLLR; } BitInfo;
int cmpBitInfo(const void* a, const void* b) {
    double da = ((BitInfo*)a)->absLLR, db = ((BitInfo*)b)->absLLR;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

void adapt_H(double* LLR) {
    BitInfo b[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        b[i].idx = i; b[i].absLLR = fabs(LLR[i]);
    }
    qsort(b, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    for (int p = 0; p < PARITY_BITS; p++) {
        int col = b[p].idx, pivot = -1;
        for (int r = 0; r < PARITY_BITS; r++)
            if (HBit[r][col]) { pivot = r; break; }
        if (pivot < 0) continue;
        if (pivot != p)
            for (int c = 0; c < TOTAL_BITS; c++) {
                // 在循环里用手动交换
                unsigned char tmp = HBit[p][c];
                HBit[p][c] = HBit[pivot][c];
                HBit[pivot][c] = tmp;

            }
        for (int r = 0; r < PARITY_BITS; r++) {
            if (r != p && HBit[r][col])
                for (int c = 0; c < TOTAL_BITS; c++)
                    HBit[r][c] ^= HBit[p][c];
        }
    }
}

void partial_spa_update(double* LLR) {
    // 改成栈上数组，无需 malloc/void*：
    double E[TOTAL_BITS] = { 0 };

    double Tval[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++)
        Tval[i] = tanh(LLR[i] / 2.0);
    for (int r = 0; r < PARITY_BITS; r++) {
        int idxs[TOTAL_BITS], cnt = 0;
        for (int c = 0; c < TOTAL_BITS; c++)
            if (HBit[r][c]) idxs[cnt++] = c;
        for (int j = 0; j < cnt; j++) {
            double prod = 1;
            for (int k = 0; k < cnt; k++) if (k != j) prod *= Tval[idxs[k]];
            prod = fmin(fmax(prod, -0.999999), 0.999999);
            double mmsg = 2.0 * atanh(prod);
            E[idxs[j]] += mmsg;
        }
    }
    double Lold[TOTAL_BITS];
    memcpy(Lold, LLR, sizeof(Lold));
    for (int i = 0; i < TOTAL_BITS; i++)
        LLR[i] = Lold[i] + DAMPING * E[i];

}

bool check_parity(const double* LLR) {
    for (int r = 0; r < PARITY_BITS; r++) {
        int sum = 0;
        for (int c = 0; c < TOTAL_BITS; c++)
            if (HBit[r][c] && LLR[c] < 0) sum ^= 1;
        if (sum) return false;
    }
    return true;
}
// === Segment 8: OSD(1) Reprocess & Feedback ===

void osd1_reprocess(const double* LLR, unsigned char* best_hard) {
    typedef struct { int idx; double absLLR; } BitInfo;
    BitInfo bits[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        bits[i].idx = i; bits[i].absLLR = fabs(LLR[i]);
    }
    qsort(bits, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);
    memcpy(g_Gtmp, GBit, sizeof(GBit));
    int piv = 0;
    for (int id = TOTAL_BITS - 1; id >= 0 && piv < OSD_K; id--) {
        int col = bits[id].idx, row;
        for (row = piv; row < OSD_K; row++)
            if (g_Gtmp[row][col]) break;
        if (row == OSD_K) continue;
        for (int c = col; c < TOTAL_BITS; c++) {
            unsigned char tmp = g_Gtmp[piv][c];
            g_Gtmp[piv][c] = g_Gtmp[row][c];
            g_Gtmp[row][c] = tmp;
        }
        for (int r = 0; r < OSD_K; r++) {
            if (r != piv && g_Gtmp[r][col])
                for (int c = col; c < TOTAL_BITS; c++)
                    g_Gtmp[r][c] ^= g_Gtmp[piv][c];
        }
        g_MRIP[piv++] = col;
    }

    for (int i = 0; i < TOTAL_BITS; i++)
        g_hard0[i] = (LLR[i] < 0);

    double best_corr = -INFINITY;
    for (int t = 0; t <= OSD_K; t++) {
        memcpy(g_hard, g_hard0, sizeof(unsigned char) * TOTAL_BITS);
        if (t > 0) g_hard[g_MRIP[t - 1]] ^= 1;

        for (int s = 0; s < RS_DAT; s++) {
            int v = 0;
            for (int b = 0; b < mm; b++)
                v |= (g_hard[s * mm + b] << b);
            data[s] = v;
        }
        encode_rs();
        for (int i = 0; i < nn - kk; i++) g_code_sym[i] = bb[i];
        for (int i = 0; i < kk; i++) g_code_sym[i + nn - kk] = data[i];
        for (int s = 0; s < RS_SYM; s++)
            for (int b = 0; b < mm; b++)
                g_cand[s * mm + b] = (g_code_sym[s] >> b) & 1;
        double corr = 0;
        for (int i = 0; i < TOTAL_BITS; i++)
            corr += LLR[i] * (1 - 2 * g_cand[i]);
        if (corr > best_corr) {
            best_corr = corr;
            memcpy(best_hard, g_cand, sizeof(unsigned char) * TOTAL_BITS);
        }
    }
}


void osd1_feedback(double* LLR) {
    unsigned char best_hard[TOTAL_BITS];
    osd1_reprocess(LLR, best_hard);

    BitInfo bits[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) {
        bits[i].idx = i; bits[i].absLLR = fabs(LLR[i]);
    }
    qsort(bits, TOTAL_BITS, sizeof(BitInfo), cmpBitInfo);

    bool isLRIP[TOTAL_BITS];
    for (int i = 0; i < TOTAL_BITS; i++) isLRIP[i] = true;
    for (int j = 0; j < OSD_K; j++) {
        isLRIP[bits[TOTAL_BITS - 1 - j].idx] = false;
    }
    for (int i = 0; i < TOTAL_BITS; i++) {
        if (!isLRIP[i]) continue;
        LLR[i] += (best_hard[i] == 0 ? +OSD_A : -OSD_A);
    }
}

void initial_mechanism(double* LLR) {
    memcpy(LLR, LLR_init, sizeof(LLR_init));
    osd1_feedback(LLR);
}
// === Segment 9: Hybrid List Decoding ===

void init_hybrid_rows() {
    for (int i = 0;i < PARITY_BITS;i++) hyb_row_perm[i] = i;
    for (int i = PARITY_BITS - 1;i > 0;i--) {
        int j = rand() % (i + 1);
        int t = hyb_row_perm[i];
        hyb_row_perm[i] = hyb_row_perm[j];
        hyb_row_perm[j] = t;
    }
    hyb_row_perm_init = true;
}

void hybrid_list_decode(double* LLR) {
    if (!hyb_row_perm_init) init_hybrid_rows();
    int vote0[TOTAL_BITS] = { 0 }, vote1[TOTAL_BITS] = { 0 };

    for (int g = 0; g < HYB_GROUPS; g++) {
        int r1 = hyb_row_perm[2 * g], r2 = hyb_row_perm[2 * g + 1];
        int Ci[TOTAL_BITS], M = 0;
        for (int c = 0;c < TOTAL_BITS;c++) {
            if (HBit[r1][c] || HBit[r2][c]) Ci[M++] = c;
        }
        if (M < 2) continue;
        // 排序 Ci by |LLR|
        for (int i = 0;i < M;i++) {
            for (int j = i + 1;j < M;j++) {
                if (fabs(LLR[Ci[i]]) > fabs(LLR[Ci[j]])) {
                    int t = Ci[i]; Ci[i] = Ci[j]; Ci[j] = t;
                }
            }
        }
        unsigned char hard0[TOTAL_BITS];
        for (int i = 0;i < TOTAL_BITS;i++) hard0[i] = (LLR[i] < 0);
        double best_corr = -INFINITY;
        unsigned char best_hard[TOTAL_BITS];

        for (int t = 0;t <= M;t++) {
            unsigned char hard[TOTAL_BITS];
            memcpy(hard, hard0, TOTAL_BITS);
            if (t > 0) hard[Ci[t - 1]] ^= 1;

            for (int s = 0;s < RS_DAT;s++) {
                int v = 0;
                for (int b = 0;b < mm;b++)
                    v |= (hard[s * mm + b] << b);
                data[s] = v;
            }
            encode_rs();
            unsigned char code_sym[RS_SYM];
            for (int i = 0;i < nn - kk;i++) code_sym[i] = bb[i];
            for (int i = 0;i < kk;i++)   code_sym[i + nn - kk] = data[i];
            unsigned char cand[TOTAL_BITS];
            for (int s = 0;s < RS_SYM;s++)
                for (int b = 0;b < mm;b++)
                    cand[s * mm + b] = (code_sym[s] >> b) & 1;

            double corr = 0;
            for (int i = 0;i < TOTAL_BITS;i++)
                corr += LLR[i] * (1 - 2 * cand[i]);
            if (corr > best_corr) {
                best_corr = corr;
                memcpy(best_hard, cand, TOTAL_BITS);
            }
        }
        for (int i = 0;i < TOTAL_BITS;i++) {
            if (best_hard[i] == 0) vote0[i]++;
            else                vote1[i]++;
        }
    }
    for (int i = 0;i < TOTAL_BITS;i++) {
        if (vote0[i] > vote1[i])      LLR[i] += +HYB_DELTA;
        else if (vote1[i] > vote0[i]) LLR[i] += -HYB_DELTA;
    }
}
// === Segment 10: ABP Decode Loop & Main ===

void try_flipping_bits(double* LLR) {
    // TODO: 实现多位翻转救援
}

void abp_decode() {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        adapt_H(LLR);
        partial_spa_update(LLR);
        if (check_parity(LLR)) return;
        if (iter > 0 && iter % OSD_PERIOD == 0)
            initial_mechanism(LLR);
        osd1_feedback(LLR);
        hybrid_list_decode(LLR);
    }
    try_flipping_bits(LLR);
}

int main() {
    srand(time(NULL));
    generate_gf(); gen_poly();
    buildHBit(); buildGBit();

    int prof_success = 0, hdd_success = 0, abp_success = 0;
    long prof_bit_err = 0, hdd_bit_err = 0, abp_bit_err = 0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        printf("\n=== Trial %d ===\n", trial + 1);

        // 產生隨機資料與碼字
        for (int i = 0; i < kk; i++) data[i] = rand() % (1 << mm);
        encode_rs();
        unsigned char rs_codeword[RS_SYM];
        for (int i = 0; i < nn - kk; i++) rs_codeword[i] = bb[i];
        for (int i = 0; i < kk; i++)   rs_codeword[i + nn - kk] = data[i];

        printf("Original: ");
        for (int s = 0; s < RS_SYM; s++) printf("%03x ", rs_codeword[s]);
        printf("\n");

        // bit 展開
        unsigned char bits[TOTAL_BITS];
        for (int s = 0; s < RS_SYM; s++)
            for (int j = 0; j < mm; j++)
                bits[s * mm + j] = (rs_codeword[s] >> j) & 1;

        // 通道
        bpsk_modulation(bits, SNR_DB);
        computeInitialLLR(SNR_DB);
        memcpy(LLR_init, LLR, sizeof(LLR_init));

        // 1. Prof (代數) 解碼
        unsigned char prof_hard[TOTAL_BITS];
        for (int i = 0; i < TOTAL_BITS; i++)
            prof_hard[i] = (rx[i] >= 0 ? 0 : 1);

        for (int s = 0; s < RS_SYM; s++) {
            int v = 0;
            for (int j = 0; j < mm; j++) v |= (prof_hard[s * mm + j] << j);
            recd[s] = index_of[v];  // index_of[v] 是 GF log domain
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

        // 2. HDD baseline
        unsigned char hard_bits[TOTAL_BITS];
        for (int i = 0; i < TOTAL_BITS; i++) hard_bits[i] = (rx[i] >= 0 ? 0 : 1);

        // HDD 解碼流程（複製 Prof 解碼步驟即可）
        unsigned short hdd_dec[RS_SYM];
        for (int s = 0; s < RS_SYM; s++) {
            int v = 0;
            for (int j = 0; j < mm; j++) v |= (hard_bits[s * mm + j] << j);
            hdd_dec[s] = v & ((1 << mm) - 1);
        }
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

        // 3. ABP-OSD list decode
        memcpy(LLR, LLR_init, sizeof(LLR_init));
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
        if (abp_err == 0) abp_success++;
        abp_bit_err += abp_err;
        printf("ABP:  ");
        for (int s = 0; s < RS_SYM; s++) printf("%03x ", abp_dec[s]);
        printf("  err=%d\n", abp_err);
    }

    // FER/BER 統計
    double hdd_FER = (double)(NUM_TRIALS - hdd_success) / NUM_TRIALS;
    double hdd_BER = (double)hdd_bit_err / (NUM_TRIALS * TOTAL_BITS);

    double prof_FER = (double)(NUM_TRIALS - prof_success) / NUM_TRIALS;
    double prof_BER = (double)prof_bit_err / (NUM_TRIALS * TOTAL_BITS);

    double abp_FER = (double)(NUM_TRIALS - abp_success) / NUM_TRIALS;
    double abp_BER = (double)abp_bit_err / (NUM_TRIALS * TOTAL_BITS);

    printf("\nSNR: %.2f dB\n", SNR_DB);

    printf("\n-- Baseline HDD (論文) --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * hdd_success / NUM_TRIALS);
    printf("平均幀錯誤率 (FER): %.6f\n", hdd_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", hdd_BER);

    printf("\n-- RS 代數譯碼 (學長版) --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * prof_success / NUM_TRIALS);
    printf("平均幀錯誤率 (FER): %.6f\n", prof_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", prof_BER);

    printf("\n-- ABP-OSD/ADP(20,3) 軟解碼 --\n");
    printf("修正成功率: %.2f%%\n", 100.0 * abp_success / NUM_TRIALS);
    printf("平均幀錯誤率 (FER): %.6f\n", abp_FER);
    printf("平均位元錯誤率 (BER): %.6f\n", abp_BER);

    return 0;
}

