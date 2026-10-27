/*  tokens.c – Detokeniser over tokens.txt.
 *
 *  tokens.txt holds one SentencePiece piece per line; the line number is the token id (8192 lines of vocabulary,
 *  then "<blank>"). Decoding is the SentencePiece convention: concatenate the pieces, turn U+2581 (▁, UTF-8 E2 96 81)
 *  into a space, strip the leading space. The tokenizer's special ids (<unk>, <pad>, <|endoftext|>) and blank are
 *  dropped, as transformers' decode(skip_special_tokens=True) does.
 */

#include "phonon25.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct P25Tokens {
    char **pieces;
    int    n;
};

/* The detokeniser over the text of tokens.txt (`len` bytes, need not be NUL-terminated): one piece per line, line
 * ends (LF, CRLF) stripped, exactly P25_VOCAB lines. `label` names the source in the message when the count is off. */
static P25Tokens *tokens_from(const char *text, size_t len, const char *label)
{
    int cap = 8192, n = 0;
    char **pieces = (char **)malloc(cap * sizeof(char *));
    size_t pos = 0;
    while (pos < len) {
        size_t end = pos;
        while (end < len && text[end] != '\n') end++;
        size_t plen = end - pos;
        while (plen > 0 && text[pos + plen - 1] == '\r') plen--;
        if (n >= cap) {
            cap *= 2;
            pieces = (char **)realloc(pieces, cap * sizeof(char *));
        }
        pieces[n] = (char *)malloc(plen + 1);
        memcpy(pieces[n], text + pos, plen);
        pieces[n][plen] = '\0';
        n++;
        pos = end + 1;
    }

    if (n != P25_VOCAB) {
        fprintf(stderr, "%s: %d lines, %d tokens expected\n", label, n, P25_VOCAB);
        for (int i = 0; i < n; i++) free(pieces[i]);
        free(pieces);
        return NULL;
    }

    P25Tokens *t = (P25Tokens *)malloc(sizeof(*t));
    t->pieces = pieces;
    t->n = n;
    return t;
}

P25Tokens *p25_tokens_parse(const char *text, size_t len)
{
    return tokens_from(text, len, "tokens.txt");
}

P25Tokens *p25_tokens_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) { perror(path); fclose(f); return NULL; }
    char *text = (char *)malloc((size_t)len + 1);
    if (!text) { fclose(f); return NULL; }
    if (fread(text, 1, (size_t)len, f) != (size_t)len) { perror(path); fclose(f); free(text); return NULL; }
    fclose(f);
    P25Tokens *t = tokens_from(text, (size_t)len, path);
    free(text);
    return t;
}

void p25_tokens_free(P25Tokens *t)
{
    if (!t) return;
    for (int i = 0; i < t->n; i++) free(t->pieces[i]);
    free(t->pieces);
    free(t);
}

const char *p25_tokens_piece(const P25Tokens *t, int id)
{
    if (!t || id < 0 || id >= t->n) return NULL;
    return t->pieces[id];
}

static int is_dropped(int id)
{
    return id == P25_BLANK || id == P25_UNK_ID || id == P25_PAD_ID || id == P25_EOS_ID;
}

char *p25_tokens_text(const P25Tokens *t, const P25Token *tokens, int n)
{
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        const char *p = p25_tokens_piece(t, tokens[i].id);
        if (p && !is_dropped(tokens[i].id)) total += strlen(p);
    }

    char *out = (char *)malloc(total + 1);
    size_t wp = 0;
    for (int i = 0; i < n; i++) {
        const char *p = p25_tokens_piece(t, tokens[i].id);
        if (!p || is_dropped(tokens[i].id)) continue;
        for (size_t rp = 0; p[rp]; ) {
            if ((unsigned char)p[rp] == 0xE2 && (unsigned char)p[rp + 1] == 0x96 && (unsigned char)p[rp + 2] == 0x81) {
                out[wp++] = ' ';
                rp += 3;
            } else {
                out[wp++] = p[rp++];
            }
        }
    }
    out[wp] = '\0';

    if (wp > 0 && out[0] == ' ')
        memmove(out, out + 1, wp);   /* includes the terminator */
    return out;
}

static int starts_word(const char *piece)
{
    return (unsigned char)piece[0] == 0xE2 && (unsigned char)piece[1] == 0x96 && (unsigned char)piece[2] == 0x81;
}

static void str_append(char **s, size_t *len, size_t *cap, const char *add)
{
    size_t n = strlen(add);
    if (*len + n + 1 > *cap) {
        *cap = (*len + n + 1) * 2;
        *s = (char *)realloc(*s, *cap);
    }
    memcpy(*s + *len, add, n + 1);
    *len += n;
}

int p25_tokens_words(const P25Tokens *t, const P25Token *tokens, int n, P25Word **out)
{
    P25Word *words = NULL;
    int n_words = 0, cap = 0;
    char *cur = NULL;
    size_t cur_len = 0, cur_cap = 0;
    float start = 0.0f, end = 0.0f;
    int have = 0;

    for (int i = 0; i <= n; i++) {
        const char *piece = i < n ? p25_tokens_piece(t, tokens[i].id) : NULL;
        int boundary = i == n || !piece || starts_word(piece);
        if (boundary && have && cur_len > 0) {
            if (n_words == cap) {
                cap = cap ? 2 * cap : 64;
                words = (P25Word *)realloc(words, (size_t)cap * sizeof(*words));
            }
            words[n_words].text  = cur;
            words[n_words].start = start;
            words[n_words].end   = end;
            n_words++;
            cur = NULL;
            cur_len = cur_cap = 0;
            have = 0;
        }
        if (i == n || !piece || is_dropped(tokens[i].id)) continue;
        float t0 = tokens[i].frame * P25_FRAME_SECONDS;
        float t1 = (tokens[i].frame + (tokens[i].duration > 0 ? tokens[i].duration : 1)) * P25_FRAME_SECONDS;
        if (!have) { start = t0; end = t1; have = 1; }
        if (t1 > end) end = t1;
        str_append(&cur, &cur_len, &cur_cap, boundary ? piece + 3 : piece);
    }
    free(cur);   /* a trailing ▁ with nothing after it */
    *out = words;
    return n_words;
}

void p25_words_free(P25Word *words, int n)
{
    if (!words) return;
    for (int i = 0; i < n; i++) free(words[i].text);
    free(words);
}
