# phonon-2.5-c

A pure-C inference engine for [Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5), Fermion Research's
five-value, 25-language student of NVIDIA's Parakeet-TDT 0.6B v3 — by its model card 11.71 % average word error on
the 25 FLEURS test sets against 13.26 % for the 2.5 GB full-precision model it derives from, and 5.06 % on the Open
ASR Leaderboard's seven English sets. Version 0.1.0.

The engine keeps the model's own weight format: every encoder linear weight is one of five values per row,
`{0, ±lo, ±hi}`, stored as a 2-bit sign plane and a 1-bit plane. The GEMMs run on NEON int8 dot products straight
from those planes, or — in exact mode — on fp32 after expanding a 16-column panel at a time. The whole model runs
from **255 MB of weights**; the exact mode reproduces the `transformers` reference **token for token**, the default
mode transcribes an 8.4-minute chapter in **3.7 s** (137× realtime), a 256 s phone call in **2.8 s** and a 3 s
utterance in **0.04 s** on an M5 Max.

```bash
$ ./csrc/phonon25 c_weights recording.wav
Okay. Um I'm looking to see if you guys have any Mercedes bins in the S series …

$ ./csrc/phonon25 --json c_weights recording.wav
{"text":"Okay. Um I'm …","words":[{"text":"Okay.","start":12.800,"end":13.200}, …
```

No dependencies beyond libc, libm and pthreads — every GEMM, the mel front end and the resampler are in the tree.

## Features

- **FastConformer encoder** (24 blocks, d=1024, 8-head relative-position attention), **2-layer LSTM prediction
  network**, **TDT joint network** with a 5-way duration head, **greedy token-and-duration decoding** — all in C
- **Five-value GEMM**: the 216 quantised encoder weights stay quantised on disk and in memory (229 MB). Default: the
  activations are quantised per row to int8 and the weights go through NEON `SDOT` as int8 codes — 2.3 TMAC/s on
  18 cores, 4× the fp32 path. `--exact` keeps everything fp32 on the original weights.
- **Verified against transformers' `ParakeetForTDT`**: in exact mode, identical token ids on a 10 s clip and on a
  256 s call given the same features; in the default mode, the same WER on LibriSpeech
- **Log-mel front end matching transformers' `ParakeetFeatureExtractor`** (pre-emphasis, zero-padded STFT, slaney mel,
  per-feature normalisation)
- **Any WAV in**: 8/16/24/32-bit PCM or 32-bit float, any sample rate, any channel count — mixed down and resampled
  to 16 kHz by the engine's own polyphase resampler, with no change in WER
- **A C library** behind the CLI: `p25_engine_open` / `p25_transcribe` / `p25_result_free` give text, word times and
  tokens from samples at any rate; a **stream** (`p25_stream_feed` / `p25_stream_next` / `p25_stream_partial`) for
  audio that arrives over time; every stage is exported too; CMake installs it with a `phonon25::phonon25` target
- **Python bindings** in one dependency-free file (`python/phonon25.py`, `ctypes` over the shared library)
- **Word timestamps** from the decoder's emission frames (`--json`)
- **Long audio cut at pauses** into 25–35 s windows (the model's regime; the same windows as Fermion's runtime),
  each decoded on its own, with a re-decode of any window the model went silent in — 1.65 % / 2.76 % WER on
  5–10 minute LibriSpeech recordings (full-length attention: 1.65 / 2.94; a chunked encoder: 10.2 % on test-clean)
- **Thread pool** over every GEMM, norm, softmax and layout shuffle; **NEON** and **AVX2** inner loops with scalar fallbacks
- **Concurrent transcription** of several files over one loaded model

## Quick start

```bash
# 1. build
make -C csrc                     # ./csrc/phonon25, ./csrc/libphonon25.a and the shared ./csrc/libphonon25.dylib|.so

# 2. weights: downloads the converted weights (183 MB) from huggingface.co/eschmidbauer/phonon-2.5-c, checks the
#    pinned sha256 and unpacks them into c_weights/ (255 MB); `make -C csrc weights` is the same thing
scripts/get_weights.sh

# 3. run (any WAV: PCM or float, any rate, mono or multi-channel)
./csrc/phonon25 c_weights recording.wav
```

CMake works too: `cmake -S . -B build && cmake --build build`; `cmake --install build` puts `libphonon25.a`, the
headers (`include/phonon25/`), the exported `phonon25::phonon25` target and the binary under the prefix.
`FetchContent` on this repository gives the same target.

The weights can also be built from Phonon-2.5's own release instead of downloaded: `pip install -r requirements.txt`
(numpy, zstandard, huggingface_hub) and `python convert_weights.py` fetch `phonon-2.5.bps.tar.zst` (163 MB) from
FermionResearch/Phonon-2.5 and write the same `c_weights/` byte for byte; see "How the weights are converted".

The int8 GEMMs need AArch64 with the dot-product extension (any Apple silicon, Neoverse, Cortex-A75+); every other
CPU — x86-64 with or without AVX2, older ARM — builds and runs the exact fp32 path, at roughly the `exact` speeds
below scaled by the machine. The x86-64 AVX2 and SSE2 builds, the generic armv8 build and the scalar-only build
(`-DP25_SCALAR`) all give the same tokens as the native exact mode on the test clips (x86 run under Rosetta 2).

### Options

```
phonon25 [options] <weights_dir> <audio.wav> [audio2.wav ...]
  --json          text plus one {text, start, end} per word
  --tokens        token ids instead of text
  --mel-in FILE   read log-mel features (raw float32 [frames × 128]) instead of computing them
  --mel-out FILE  write the computed features in that layout
  --threads N     thread-pool size (default: online CPUs; also P25_THREADS)
  --exact         fp32 arithmetic on the five-value weights instead of int8 dot products (also P25_EXACT=1)
  --cache-codes   pre-pack the int8 weight codes once (604 MB of RAM): a 3 s utterance ~15 % faster (also P25_CACHE_CODES=1)
  --repeat N      run the pipeline N times on one file and report the fastest
  --quiet         no encoder progress on stderr
  --stream        no audio files: 16-bit mono PCM on stdin, one line per finished 25–35 s window
  --rate R        with --stream: the PCM's sample rate (default 16000; anything else is resampled on the way)
  --partial S     with --stream: a provisional transcript of the pending audio on stderr every S seconds
```

`P25_GEMM=exact|int8|int8w` picks the GEMM mode (default `int8w`; `int8` keeps the weights exact and quantises only
the activations, at half the speed). Audio over 35 s is cut at pauses into 25–35 s windows (see *Long audio*);
`P25_SEGMENT=0` hands the whole file to the encoder instead, where `P25_CHUNK=0` means one full-length attention
pass (quadratic memory: ~2 GB at 256 s) and `P25_CHUNK=1` the chunked encoder (`P25_CHUNK_SIZE` /
`P25_CHUNK_OVERLAP` in 80 ms frames, default 400 / 8) — both kept for comparison, neither a good idea on long files.

### Input

Any RIFF WAV: PCM 8, 16, 24 or 32-bit or IEEE float, plain or `WAVE_FORMAT_EXTENSIBLE`, any sample rate, any number
of channels. Channels are averaged to mono, and anything not at 16 kHz goes through the engine's windowed-sinc
polyphase resampler (Kaiser β = 9, cut off at 0.46 of the lower rate; 66–106 dB against analytic test signals,
0.03 s for 256 s of 44.1 or 48 kHz audio). The library takes mono float32 at any rate and does the same inside
`p25_transcribe`.

## Library

The CLI is a consumer of the library in `csrc/`; `make -C csrc` builds `libphonon25.a` next to it, and the whole API
is `csrc/phonon25.h`:

```c
#include "phonon25.h"

P25Engine *engine = p25_engine_open("c_weights");               /* weights.bin, weights.json, tokens.txt */
float *audio; int n, rate, channels;
p25_wav_read("recording.wav", &audio, &n, &rate, &channels);    /* any PCM / float WAV, mixed to mono */
P25Result r;
p25_transcribe(engine, audio, n, rate, &r);                     /* resample → mel → encoder → TDT → text */
printf("%s\n", r.text);
for (int i = 0; i < r.n_words; i++)
    printf("%.2f %.2f %s\n", r.words[i].start, r.words[i].end, r.words[i].text);
p25_result_free(&r); free(audio); p25_engine_close(engine);
```

`examples/transcribe.c` is that program (`make -C csrc example`, or the `transcribe` target of the CMake build).
`p25_transcribe` may be called from any number of threads over one engine.

For audio that arrives over time there is a stream: `p25_stream_open(engine, rate)`, `p25_stream_feed` (mono float32
at that rate, any chunk size — resampled on the way by a stateful `P25Resampler` whose output is bit for bit the
whole-file resampler's), `p25_stream_next` for each window as it completes, `p25_stream_partial` for a provisional
transcript of the audio pending since the last cut, `p25_stream_finish`, `p25_stream_result`. Windows are the model's
regime, so final text arrives at window granularity — up to 35 s after the speech — while partials (a decode of the
pending audio, ≤ 35 s, about 0.2 s of work) give provisional text as often as you like; a stream fed a whole file
emits exactly the windows `p25_transcribe` gives for it (`make check` holds it to that).
`phonon25 --stream [--rate R] [--partial S] [--json] c_weights < audio.pcm` is the CLI form: 16-bit little-endian
mono PCM in (16 kHz unless `--rate` says otherwise), one line per finished window out, with `--json` a segment
object carrying that window's words. `p25_pcm16le_to_float` is the library's converter for such raw PCM.

### Python

`python/phonon25.py` is the same API from Python, over the shared library `make -C csrc` builds next to the static
one (`csrc/libphonon25.dylib` / `.so`; found through `PHONON25_LIB` or its default location). One file, `ctypes`
only; numpy arrays are accepted when numpy is present but nothing is required:

```python
import phonon25
engine = phonon25.Engine("c_weights")                 # cache_codes=True pre-packs the int8 codes
result = engine.transcribe_file("recording.wav")     # or engine.transcribe(samples, rate)
print(result.text)
for w in result.words:
    print(f"{w.start:7.2f} {w.end:7.2f}  {w.text}")

with engine.stream(rate=48000) as s:                 # audio arriving over time, any rate
    for chunk in chunks:
        for seg in s.feed(chunk):
            print(seg.start, seg.end, seg.text)
    for seg in s.finish():
        print(seg.start, seg.end, seg.text)
```

`Result` carries `text`, `words`, `tokens`, `segments` (each with its own words and tokens), `encoder_frames`,
`duration_seconds` and the stage times; `phonon25.read_wav`, `phonon25.resample` and `phonon25.pcm16le_to_float` expose
the audio utilities. `python3 python/phonon25.py c_weights recording.wav` runs it as a script. `make check`
exercises the bindings against the CLI's output.

The stages behind it — `p25_resample`,
`p25_mel_spectrogram`, `p25_encoder`, `p25_greedy_decode`, `p25_tokens_text`, `p25_tokens_words` — are exported for
callers that want them one at a time, and `p25_transcribe_mel` starts from features. Thread count, GEMM mode and
chunking are read from `P25_THREADS`, `P25_GEMM`, `P25_CHUNK…` on first use; a server can instead give each of its
threads a pool of its own (`p25_pool_create`, `p25_set_pool` in `threadpool.h`). The library is compiled with
`-march=native`: build it on the machine that runs it.

## Speed

Apple M5 Max (6 performance + 12 efficiency cores), 18 threads, best of several runs (`--repeat`), measured on
2026-10-27 with the machine otherwise quiet (≈ 13 % busy); one 256 s phone call, 3 s / 10 s clips cut from it, and
an 8.4-minute LibriSpeech chapter:

| Mode | 3 s utterance | 10 s clip | 256 s call, 25–35 s windows (default) | 256 s call, `P25_REDECODE=0` | 256 s, chunked encoder | 256 s, full-length attention | 8.4 min chapter (default) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `int8w` (default) | **0.041 s · 74×** | **0.083 s · 120×** | **2.76 s · 93× realtime** | **2.07 s · 123×** | 1.94 s · 132× | 3.69 s · 69× | **3.7 s · 137×** |
| `int8w` + `--cache-codes` | 0.035 s · 86× | 0.083 s · 121× | | 1.89 s · 135× | | | |
| `exact` | 0.100 s · 30× | 0.244 s · 41× | 8.4 s · 30× | | 5.7 s · 45× | 7.4 s · 35× | |

Of which the decoder is about 0.5 s on the long files. The default's 2.76 s on the call against 2.07 s without the
re-decode is its 5 re-decodes (far-end speech and line noise the energy check flags, see *Long audio*); read speech
pays nothing for them — the chapter is 505 s of audio in 3.7 s with no window flagged. `--cache-codes` pre-packs
the int8 weight codes once (604 MB) so that each pass skips expanding the sign and flag planes — about 5 ms per
pass on the 3 s clip, within noise on the 10 s one and nothing on long audio.

The int8 path is sensitive to background load in a way exact mode is not — its many short fork-joins each wait for
whichever worker was descheduled — so measure on a quiet machine: with the same engine and the same model size,
phonon-2-c recorded Phonon-2's 256 s call at 1.67 s on an idle machine and 2.7 s with Docker's VM holding ~1.6 of
the 18 cores, and the figures above carry some background load too. An optional Accelerate / OpenBLAS build
existed in phonon-2-c until 2026-10-06: it matched the hand-rolled code in the default mode and was faster only in
exact mode; it was removed so that the library has one code path and no dependency.

The thread pool spins briefly between dispatches (`P25_SPIN`, default 8000 iterations ≈ 100 µs; `0` parks at once).
`--repeat N` runs the pipeline N times on one file and reports the fastest, for latency work.

## Accuracy and verification

### Word error rate

`scripts/eval_librispeech.py` fetches a LibriSpeech split from OpenSLR, runs the engine over it and scores with the
Whisper English normaliser (what the Open ASR Leaderboard uses):

```bash
python scripts/eval_librispeech.py c_weights                     # test-clean
python scripts/eval_librispeech.py c_weights --split test-other
```

| Set | `int8w` (default) | `exact` | Model card (leaderboard protocol) | Phonon-2, same engine and protocol |
| --- | --: | --: | --: | --: |
| test-clean (2,620 utterances, 5.4 h) | **2.03 %** | 2.01 % | 1.60 % | 2.16 % |
| test-other (2,939 utterances, 5.3 h) | **3.95 %** | 3.87 % | 3.41 % | 4.34 % |

The int8 default costs 12 edits in 53,000 words on test-clean and 42 in 52,900 on test-other against exact mode.
Exact mode and torch `ParakeetForTDT`, each through its own mel front end, give the identical transcript on every
utterance of a random 300-utterance subset (`--limit 300 --seed 0`: 300 of 300), so the gap to the card is the
leaderboard's scoring protocol (its dataset variant, reference text and normaliser), not the engine. The errors are
the model's: proper nouns ("Osh Popham" → "Popum", "La Vallière" → "Levalier"), dialect spellings the normaliser
cannot bridge ("yearnin", "allers"), "missus" written as "misses".

A 300-utterance subset of test-clean (`--limit 300 --seed 0`; 2.02 % as 16 kHz float) fed as 8, 22.05, 44.1 or
48 kHz, stereo 24-bit or 8-bit WAVs (`--rate`, `--channels`, `--subtype`) scores 1.99–2.09 %, within noise of the
baseline; unlike Phonon-2, which lost 0.2 points to the missing 4–8 kHz band, this model scores 1.99 % on 8 kHz
input.

### Parity with transformers

`scripts/compare_reference.py` runs the C engine and `transformers` on the same recording:

```bash
python scripts/compare_reference.py c_weights recording.wav --phonon25-repo /path/to/FermionResearch/Phonon-2.5
```

- **Graph parity** (both sides fed the same log-mel features, exact mode, full-length attention): **identical token
  ids** — 50 of 50 on the 10 s clip, 965 of 965 on the 256 s call. This is the pass/fail criterion for the graph.
- **Default int8 mode**: 34 token edits in 965 against the exact run on the 256 s call, none in 50 on the clip — the
  same scale as Fermion's bf16 and int8 engines show against torch. Its accuracy is the WER above, not parity.
- **End to end** with the C mel front end: identical to torch on both recordings. Two feature extractors never agree
  exactly in silent mel bins, where per-feature normalisation amplifies float32 rounding to O(1) — median difference
  1e-5 here, 99th percentile 0.19, max 1.3, all in the dead bins above the phone band — and Phonon-2 moved 20 tokens
  on this call for it; Phonon-2.5 moves none.

### Long audio

Phonon-2.5 is a ≤ 35 s model. Its predecessor had a sharp failure mode — given a window that begins mid-speech it
could emit nothing for 10–30 s of clear speech and resume later, torch included, with the outcome flipping on a
0.5 s shift of the cut — and the engine's long-audio design comes from that. `scripts/eval_longform.py` measures it
on 5–10 minute recordings built by concatenating LibriSpeech chapters (exact references):

| Long-form strategy | test-clean (15 chapters, 116 min) | test-other (18 chapters, 121 min) |
| --- | --: | --: |
| the original utterances one by one | 1.69 % | 2.97 % |
| one full-length attention pass per recording | 1.65 % | 2.94 % |
| **pause-aligned 25–35 s windows with the dropout re-decode (the engine's default)** | **1.65 %** | **2.76 %** |
| the same windows without the re-decode (Fermion's runtime; `P25_REDECODE=0`) | 1.65 % | 2.76 % |
| cut every 32 s regardless of content, pieces decoded alone | 2.31 % | |
| chunked encoder, 32 s chunks with 0.6 s of context (`P25_SEGMENT=0 P25_CHUNK=1`) | 10.2 % | |

Phonon-2.5 is far less prone to the mid-speech silence than Phonon-2 (which scored 26 % on the fixed cuts and 40 %
with the chunked encoder) but still pays for every cut inside speech, so the engine keeps cutting audio over 35 s
into 25–35 s windows ending in the longest pause of each band — the same windows Fermion's runtime uses — and
decoding each on its own; that scores best on both sets. Every decoded window is still checked for a stretch of
≥ 2 s without a token over speech energy and re-decoded as two halves split at a pause when it has one; on these
237 minutes of read speech the check never fired, so the re-decode costs nothing there. `--json` lists the windows
under `"segments"`. The check is energy-based, so on telephone audio with line noise and far-end speech it flags
the odd window — 5 on the 256 s call, which takes 2.76 s instead of 2.07 s (`P25_REDECODE=0`).

## Tests

`make -C csrc check` runs the built engine over three LibriSpeech test-clean utterances in `tests/data/` (CC BY 4.0,
~1 MB): exact-mode token ids must match `tests/expected/` byte for byte — they do on Apple silicon, x86-64 (AVX2 and
SSE2), generic ARM and the scalar-only build — and the default mode, `--json`, batched invocation, a 22.05 kHz
input and `examples/transcribe` must agree with the reference text and each other. Needs Python 3 and the converted
weights, nothing else. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs it on macOS (Apple silicon),
x86-64 Linux and ARM Linux on every push, with the CMake build and install and the scalar-only build.

`make -C csrc fuzz` rebuilds the library with AddressSanitizer and UBSan and throws a few thousand mutated WAV
files, weights manifests, truncated weight binaries and token files at the parsers (`tests/fuzz/`); CI runs a short
round on Linux. Non-finite samples (a NaN or infinity in a float WAV or a live feed) are zeroed on the way in — one
would otherwise silence a whole window through the mel normalisation.

## How the weights are converted

`convert_weights.py` reads the `model.fermion` container with Fermion's reference reader (`fermion_container.py`,
vendored unchanged) and writes `weights.bin` + `weights.json` + `tokens.txt`:

| Tensors | Count | Stored as |
| --- | --: | --- |
| Encoder linears (FF, Q/K/V fused, position, output, pointwise convs) | 216 | five-value: 2-bit signs + 1-bit lo/hi flags, `[in, out]`, fp32 `lo`/`hi` per output — 229 MB |
| Subsampling linear, encoder projector | 2 | int8 with one fp32 scale per output column (exact from the container's int6), used as stored |
| Embedding, LSTM, decoder projector, joint head | 7 | int8 with one fp32 scale per row (exact), expanded to fp32 when loaded — 17 MB on disk |
| Norms, biases, subsampling convs, depthwise conv (BatchNorm folded in) | 354 | fp32 |

Everything is exact except the BatchNorm fold (fp32 arithmetic). The vocabulary comes from the tarball's own
config, so no other download is needed.

`--archive` also packs the three files into `phonon-2.5-c-weights-v1.tar.gz`, built reproducibly (ustar, fixed
order, zero timestamps and owners, no gzip header name) so every run gives the same bytes. That file is what
[huggingface.co/eschmidbauer/phonon-2.5-c](https://huggingface.co/eschmidbauer/phonon-2.5-c) publishes, with its
sha256 and the three files' in `SHA256SUMS.txt` there. That repo carries verbatim copies of `convert_weights.py`
and `fermion_container.py` and a `build.sh` that rebuilds and checks the archive; a change to the converter here
is a change there too. `weights.json` records the layout's name under `_meta` (`phonon-2.5-c-weights-v1`); the loader
refuses any other, so a stale download or an old conversion fails with a message instead of running on garbage.

## Project layout

| File | Lines | Purpose |
| --- | --: | --- |
| `csrc/phonon25.h` | 425 | Model constants, `Weight` kinds, weight structs, the whole API (engine, stream, resampler, components) |
| `csrc/engine.c` | 474 | `P25Engine`, `p25_transcribe`: windows → mel → encoder → decoder → text, words, segments; the dropout re-decode; `P25Stream` |
| `csrc/segment.c` | 184 | Pause-aligned 25–35 s windows for long audio (Fermion's rule), pause and speech measures |
| `csrc/blas.c` | 1392 | fp32 sgemm (NEON/AVX2, packed, blocked) and the five-value / int8 GEMMs — no BLAS library |
| `csrc/encoder.c` | 762 | Subsampling, conformer blocks, chunking, projector |
| `csrc/decoder.c` | 141 | LSTM, joint, greedy TDT |
| `csrc/mel.c` | 217 | Log-mel features |
| `csrc/tensor_ops.c` | 649 | LayerNorm, softmax, SiLU, depthwise conv, row GEMV, producer-side int8 quantisation |
| `csrc/weights.c` | 527 | Manifest parser and strict loader, the int8 code cache |
| `csrc/wav.c`, `resample.c` | 346 | WAV reader (any PCM / float format, mixdown), stateful polyphase resampler |
| `csrc/main.c`, `threadpool.c`, `tokens.c` | 864 | CLI (files, JSON, `--stream`), spinning fork-join pool, detokeniser and word times |
| `examples/transcribe.c` | 47 | The library API end to end |
| `python/phonon25.py` | 400 | Python bindings (`ctypes` over `libphonon25`): `Engine`, `Stream`, the audio utilities |
| `csrc/bench_gemm.c` | 58 | GEMM microbenchmark (`make -C csrc bench_gemm`) |
| `scripts/get_weights.sh` | 37 | Downloads the published weights, checks the pinned sha256, unpacks them |
| `convert_weights.py` | 277 | Container → C weights, `--archive` packs them for the Hub |
| `scripts/compare_reference.py` | 177 | Parity check against transformers |
| `scripts/eval_librispeech.py` | 205 | WER on a LibriSpeech split, optionally through resampled / multi-channel / PCM variants |
| `scripts/eval_longform.py` | 247 | Long-form WER on concatenated chapters: windows, chunking and cutting strategies side by side |
| `tests/check.py`, `tests/fuzz/` | 533 | The regression check behind `make check`; the sanitizer fuzz harness behind `make fuzz` |

Derived from [phonon-2-c](https://github.com/eschmidbauer/phonon-2-c), the engine for Phonon-2: Phonon-2.5 ships
in the same container with the same 723 records and shapes, so the graph is unchanged and only the model, the
weights format name and the names moved (`phonon2`, `p2_`, `P2_` → `phonon25`, `p25_`, `P25_`, so the two engines
install side by side). That engine was in turn derived from
[parakeet-unified-en-0.6b-c](https://github.com/eschmidbauer/parakeet-unified-en-0.6b-c), which has the same
encoder family; the differences that matter for Phonon-2.5 (no linear biases, BatchNorm after the depthwise conv, no
input scaling, PyTorch LSTM gate order, TDT decoding, an 8193-token vocabulary) are all here.

## Licence

Engine code: MIT. `fermion_container.py`: Apache-2.0, Fermion Research. The weights the converter downloads are
CC-BY-4.0 (Fermion Research's derivation of NVIDIA's parakeet-tdt-0.6b-v3); see `NOTICE`.
