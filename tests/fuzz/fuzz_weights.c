/*  fuzz_weights.c – p25_load_weights on a (mutated) weights.json / weights.bin pair, and p25_tokens_load on a
 *  tokens.txt: each must accept or reject cleanly. Built with the sanitizers by `make -C csrc fuzz`. */
#include "phonon25.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: fuzz_weights <weights.bin> <weights.json> [tokens.txt]\n"); return 2; }
    p25_verbose = 0;
    P25Model m;
    int rc = p25_load_weights(&m, argv[1], argv[2]);
    printf("weights: %s\n", rc == 0 ? "accepted" : "rejected");
    if (rc == 0) p25_free_weights(&m);
    if (argc > 3) {
        P25Tokens *t = p25_tokens_load(argv[3]);
        printf("tokens: %s\n", t ? "accepted" : "rejected");
        p25_tokens_free(t);
    }
    return 0;
}
