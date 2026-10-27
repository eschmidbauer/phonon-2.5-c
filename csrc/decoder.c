/*  decoder.c – LSTM prediction network, joint network, greedy TDT decoding.
 *
 *  The loop is transformers' ParakeetTDTGenerationMixin: the decoder starts from zero state fed the blank token
 *  (whose embedding row is all zeros), runs only when a non-blank token is emitted, and the frame pointer advances by
 *  the predicted duration – at least 1 on a blank. One addition from NeMo: after P25_MAX_SYMBOLS tokens at a single
 *  frame the pointer moves on by one, where transformers only caps the total step count.
 */

#include "phonon25.h"
#include "blas.h"
#include "threadpool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const int DURATIONS[P25_N_DURATIONS] = {0, 1, 2, 3, 4};

#define sigmoid_f p25_sigmoid   /* the clamped one from phonon25.h */

/* LSTM pre-activations for gate rows [r0, r1): gates[r] = W_ih[r]·x + W_hh[r]·h + b_ih[r] + b_hh[r].
 * One pass over both matrices per layer, spread over the pool: the two 6.5 MB matrices per layer are the bulk of
 * what the decoder streams, and reading them on one thread was most of the decode time. */
struct lstm_gates_args {
    const float *x; int input_size;
    const float *W_ih, *W_hh, *b_ih, *b_hh, *h;
    float *gates;
};

static void lstm_gates_worker(int tid, int nt, void *aux)
{
    struct lstm_gates_args *a = (struct lstm_gates_args *)aux;
    enum { H = P25_LSTM_HIDDEN };
    int r0 = (4 * H * tid) / nt, r1 = (4 * H * (tid + 1)) / nt;
    for (int r = r0; r < r1; r++)
        a->gates[r] = p25_dot_f32(a->W_ih + (size_t)r * a->input_size, a->x, a->input_size)
                    + p25_dot_f32(a->W_hh + (size_t)r * H, a->h, H)
                    + a->b_ih[r] + a->b_hh[r];
}

/* One nn.LSTM step in PyTorch's layout, gate rows ordered i, f, g, o. */
static void lstm_step(const float *x, int input_size,
                      const float *W_ih, const float *W_hh, const float *b_ih, const float *b_hh,
                      float *h, float *c)
{
    enum { H = P25_LSTM_HIDDEN };
    float gates[4 * H];
    struct lstm_gates_args a = { x, input_size, W_ih, W_hh, b_ih, b_hh, h, gates };
    p25_parallel(p25_pool(), lstm_gates_worker, &a);

    for (int j = 0; j < H; j++) {
        float gi = sigmoid_f(gates[j]);
        float gf = sigmoid_f(gates[H + j]);
        float gg = tanhf(gates[2 * H + j]);
        float go = sigmoid_f(gates[3 * H + j]);
        c[j] = gf * c[j] + gi * gg;
        h[j] = go * tanhf(c[j]);
    }
}

/* ── Decoder step: embedding → 2-layer LSTM → decoder_projector ─────────── */

void p25_decoder_step(const P25Model *m, int token_id, LSTMState *state, float *dec_out)
{
    const DecoderWeights *d = &m->dec;
    const float *emb = d->embed_w + (size_t)token_id * P25_EMBED_DIM;

    lstm_step(emb, P25_EMBED_DIM, d->lstm_w_ih[0], d->lstm_w_hh[0], d->lstm_b_ih[0], d->lstm_b_hh[0],
              state->h[0], state->c[0]);
    lstm_step(state->h[0], P25_LSTM_HIDDEN, d->lstm_w_ih[1], d->lstm_w_hh[1], d->lstm_b_ih[1], d->lstm_b_hh[1],
              state->h[1], state->c[1]);

    p25_gemv_rows(d->proj_w, state->h[1], dec_out, P25_JOINT_DIM, P25_LSTM_HIDDEN);
    p25_bias_add(dec_out, d->proj_b, P25_JOINT_DIM);
}

/* ── Joint network ───────────────────────────────────────────────────────── */

void p25_joint(const P25Model *m, const float *enc_t, const float *dec_out, float *logits)
{
    const DecoderWeights *d = &m->dec;
    float hidden[P25_JOINT_DIM];
    for (int i = 0; i < P25_JOINT_DIM; i++) {
        float v = enc_t[i] + dec_out[i];
        hidden[i] = v > 0.0f ? v : 0.0f;
    }
    p25_gemv_rows(d->head_w, hidden, logits, P25_N_LOGITS, P25_JOINT_DIM);
    p25_bias_add(logits, d->head_b, P25_N_LOGITS);
}

/* ── Greedy TDT decoding ─────────────────────────────────────────────────── */

static inline int argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

int p25_greedy_decode(const P25Model *m, const float *enc_out, int T, P25Token **out_tokens)
{
    LSTMState state;
    memset(&state, 0, sizeof(state));

    int cap = T + 16;
    P25Token *tokens = (P25Token *)malloc(cap * sizeof(P25Token));
    int n_tokens = 0;

    float dec_out[P25_JOINT_DIM];
    float logits[P25_N_LOGITS];

    p25_decoder_step(m, P25_BLANK, &state, dec_out);

    int t = 0, symbols = 0;
    while (t < T) {
        p25_joint(m, enc_out + (size_t)t * P25_JOINT_DIM, dec_out, logits);
        int token = argmax(logits, P25_VOCAB);
        int duration = DURATIONS[argmax(logits + P25_VOCAB, P25_N_DURATIONS)];

        if (token == P25_BLANK) {
            if (duration == 0) duration = 1;
        } else {
            if (n_tokens >= cap) {
                cap *= 2;
                tokens = (P25Token *)realloc(tokens, cap * sizeof(P25Token));
            }
            tokens[n_tokens].id = token;
            tokens[n_tokens].frame = t;
            tokens[n_tokens].duration = duration;
            n_tokens++;
            p25_decoder_step(m, token, &state, dec_out);
            symbols++;
            if (duration == 0 && symbols >= P25_MAX_SYMBOLS) duration = 1;
        }
        if (duration) symbols = 0;
        t += duration;
    }

    *out_tokens = tokens;
    return n_tokens;
}
