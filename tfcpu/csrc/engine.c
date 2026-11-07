/* NativeCPU: a C inference engine for the v2 Transformer CPU.
 *
 * It runs the same weights with the same operations as FastCPU (tfcpu/fast.py):
 * sparse matrix-vector products over the non-zero weights of each Linear layer,
 * hardmax attention, ReLU and residual adds, evaluated only on the rows that can
 * change (the CPU row and the store-target MEM row), with the keys/values of the
 * other rows cached. Everything is an integer (or half-integer) below 2^24, so
 * float32 results are exact whatever the summation order. No -ffast-math.
 *
 * Hardmax ties on a CPU query, or a store-port tie, would mean a broken v2
 * invariant: the engine stops and returns a negative code instead of guessing.
 */
#include <stdint.h>
#include <string.h>

typedef struct {
    int32_t rows, cols;
    const int32_t *indptr, *indices;
    const float *data;
} CSR;

typedef struct {
    int32_t T, d, cpu, mem0, nmem, halted, out, out_flag;   /* field indices on the CPU row */
    /* block 1 */
    int32_t dk1, dv1;
    CSR q1, o1;                 /* q1: dk1 x d, o1: d x dv1 */
    const float *K1, *V1;       /* cached fetch keys / values: T x dk1, T x dv1 */
    CSR m1a, m1b;               /* MLP fc1: h1 x d, fc2: d x h1 */
    const float *c1a, *c1b;     /* their biases */
    /* block 2 */
    int32_t H, dk2, dv2;
    CSR q2, k2, v2, o2;         /* q2,k2: H*dk2 x d, v2: H*dv2 x d, o2: d x H*dv2 */
    const float *K2;            /* load-head keys: T x dk2 */
    float *V2;                  /* load-head values: T x dv2 (updated on stores) */
    const float *Qs;            /* store-head queries of MEM rows: nmem x (H-1) x dk2 */
    const float *best;          /* best non-CPU score per MEM query: nmem x (H-1) */
    CSR m2a, m2b;
    const float *c2a, *c2b;
} Model;

enum { ERR_FETCH_TIE = -1, ERR_LOAD_TIE = -2, ERR_STORE_TIE = -3 };

static void mv(const CSR *A, const float *x, float *y) {          /* y = A x */
    for (int32_t r = 0; r < A->rows; r++) {
        float s = 0.0f;
        for (int32_t k = A->indptr[r]; k < A->indptr[r + 1]; k++) s += A->data[k] * x[A->indices[k]];
        y[r] = s;
    }
}

static void mv_add(const CSR *A, const float *x, float *y) {      /* y += A x */
    for (int32_t r = 0; r < A->rows; r++) {
        int32_t k0 = A->indptr[r], k1 = A->indptr[r + 1];
        if (k0 == k1) continue;
        float s = 0.0f;
        for (int32_t k = k0; k < k1; k++) s += A->data[k] * x[A->indices[k]];
        y[r] += s;
    }
}

/* y = x + fc2(relu(fc1 x + b1)) + b2, in place on y (y may alias x) */
static void mlp_residual(const CSR *fc1, const float *b1, const CSR *fc2, const float *b2,
                         float *row, float *h, float *tmp, int32_t d) {
    mv(fc1, row, h);
    for (int32_t j = 0; j < fc1->rows; j++) { float v = h[j] + b1[j]; h[j] = v > 0.0f ? v : 0.0f; }
    mv(fc2, h, tmp);
    for (int32_t i = 0; i < d; i++) row[i] += tmp[i] + b2[i];
}

/* unique-argmax hardmax of q against T keys; returns the winning row or -1 on a tie */
static int32_t argmax_unique(const float *q, const float *K, int32_t T, int32_t dk) {
    int32_t best = 0, ties = 0;
    float bs = 0.0f;
    for (int32_t t = 0; t < T; t++) {
        float s = 0.0f;
        for (int32_t k = 0; k < dk; k++) s += q[k] * K[t * dk + k];
        if (t == 0 || s > bs) { bs = s; best = t; ties = 1; }
        else if (s == bs) ties++;
    }
    return ties == 1 ? best : -1;
}

static int step(const Model *m, float *x) {
    const int32_t d = m->d, H = m->H, dk2 = m->dk2, dv2 = m->dv2, S = m->H - 1;
    float c[d], c1[d], tmp[d], r[d];
    float h[m->m1a.rows > m->m2a.rows ? m->m1a.rows : m->m2a.rows];
    float q[m->dk1 > H * dk2 ? m->dk1 : H * dk2];
    float kc[H * dk2], vc[H * dv2], hv[H * dv2];

    /* ---- block 1: fetch (hardmax over cached PROG keys), then speculative execute */
    memcpy(c, x + m->cpu * d, sizeof c);
    mv(&m->q1, c, q);
    int32_t t = argmax_unique(q, m->K1, m->T, m->dk1);
    if (t < 0) return ERR_FETCH_TIE;
    mv_add(&m->o1, m->V1 + t * m->dv1, c);
    mlp_residual(&m->m1a, m->c1a, &m->m1b, m->c1b, c, h, tmp, d);
    memcpy(c1, c, sizeof c);

    /* ---- block 2, CPU row: load head; the store heads' output on the CPU row is gated off */
    mv(&m->q2, c1, q);                                   /* rows 0..dk2-1 are the load head */
    t = argmax_unique(q, m->K2, m->T, dk2);
    if (t < 0) return ERR_LOAD_TIE;
    memset(hv, 0, sizeof hv);
    memcpy(hv, m->V2 + t * dv2, dv2 * sizeof(float));
    mv_add(&m->o2, hv, c);
    mlp_residual(&m->m2a, m->c2a, &m->m2b, m->c2b, c, h, tmp, d);

    /* ---- block 2, store heads: a MEM row is written iff the CPU's key beats its best other key in every store head */
    mv(&m->k2, c1, kc);
    mv(&m->v2, c1, vc);
    for (int32_t i = 0; i < m->nmem; i++) {
        int win = 1;
        for (int32_t s = 0; s < S; s++) {
            const float *qs = m->Qs + ((size_t)i * S + s) * dk2, *kk = kc + (s + 1) * dk2;
            float sc = 0.0f;
            for (int32_t k = 0; k < dk2; k++) sc += qs[k] * kk[k];
            float b = m->best[i * S + s];
            if (sc == b) return ERR_STORE_TIE;
            if (sc < b) win = 0;
        }
        if (!win) continue;
        float *row = x + (size_t)(m->mem0 + i) * d;
        memcpy(r, row, sizeof r);
        memset(hv, 0, dv2 * sizeof(float));                  /* load head: not needed on a MEM row */
        memcpy(hv + dv2, vc + dv2, (size_t)S * dv2 * sizeof(float));
        mv_add(&m->o2, hv, r);
        mlp_residual(&m->m2a, m->c2a, &m->m2b, m->c2b, r, h, tmp, d);
        memcpy(row, r, sizeof r);
        float v[H * dv2];
        mv(&m->v2, r, v);                                    /* refresh this row's cached load value */
        memcpy(m->V2 + (size_t)(m->mem0 + i) * dv2, v, dv2 * sizeof(float));
    }
    memcpy(x + m->cpu * d, c, sizeof c);
    return 0;
}

/* Run up to max_steps steps on one machine; stops after the step that halts.
 * After every step whose out_flag is set, the CPU's `out` value is appended to
 * outbuf (up to cap values; *nout counts all of them).
 * Returns the number of steps executed, or a negative error code. */
int64_t tf_run(const Model *m, float *x, int64_t max_steps, float *outbuf, int64_t cap, int64_t *nout) {
    int64_t n = 0;
    const float *cpu = x + (size_t)m->cpu * m->d;
    while (n < max_steps && cpu[m->halted] != 1.0f) {
        int e = step(m, x);
        if (e) return e;
        n++;
        if (cpu[m->out_flag] == 1.0f) {
            if (*nout < cap) outbuf[*nout] = cpu[m->out];
            (*nout)++;
        }
    }
    return n;
}
