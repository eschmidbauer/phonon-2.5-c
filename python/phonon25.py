"""phonon25 – Python bindings for the phonon-2.5-c engine (ctypes over libphonon25; no dependencies, numpy optional).

    import phonon25
    engine = phonon25.Engine("c_weights")                      # cache_codes=True pre-packs the int8 codes (604 MB)
    result = engine.transcribe_file("recording.wav")          # any WAV: PCM or float, any rate, any channel count
    result = engine.transcribe(samples, rate=48000)           # mono float32 samples at any rate
    print(result.text)
    for w in result.words: print(f"{w.start:7.2f} {w.end:7.2f} {w.text}")

    with engine.stream(rate=16000) as s:                      # audio that arrives over time
        for chunk in chunks:
            for seg in s.feed(chunk): print(seg.start, seg.end, seg.text)
        print(s.partial())                                    # provisional text of the pending audio
        for seg in s.finish(): print(seg.start, seg.end, seg.text)

`samples` may be a numpy float32 array, an array.array('f'), the raw bytes of float32 samples, or any sequence of
floats. The shared library is found through the PHONON25_LIB environment variable, next to this file's parent
directory (csrc/libphonon25.dylib or .so, as `make -C csrc` leaves it), or on the dynamic loader's path.
Environment variables the engine reads (P25_THREADS, P25_GEMM, P25_CACHE_CODES, P25_SEGMENT, …) apply as in the CLI.
"""
from __future__ import annotations

import array
import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

__all__ = ["Engine", "Stream", "Result", "Word", "Token", "Segment", "resample", "read_wav", "pcm16le_to_float",
           "set_verbose", "load_library"]

_c = None   # the loaded library


# ── Library loading and C structures ─────────────────────────────────────────
# The Structures mirror P25Word / P25Token / P25Segment / P25Result in csrc/phonon25.h field for field.

class _P25Word(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("start", ctypes.c_float), ("end", ctypes.c_float)]


class _P25Token(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("frame", ctypes.c_int), ("duration", ctypes.c_int)]


class _P25Segment(ctypes.Structure):
    _fields_ = [("start", ctypes.c_float), ("end", ctypes.c_float), ("text", ctypes.c_char_p),
                ("first_word", ctypes.c_int), ("n_words", ctypes.c_int),
                ("first_token", ctypes.c_int), ("n_tokens", ctypes.c_int)]


class _P25Result(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p),
                ("words", ctypes.POINTER(_P25Word)), ("n_words", ctypes.c_int),
                ("tokens", ctypes.POINTER(_P25Token)), ("n_tokens", ctypes.c_int),
                ("segments", ctypes.POINTER(_P25Segment)), ("n_segments", ctypes.c_int),
                ("encoder_frames", ctypes.c_int), ("duration_seconds", ctypes.c_float),
                ("t_mel", ctypes.c_double), ("t_encoder", ctypes.c_double), ("t_decoder", ctypes.c_double)]


def _candidates() -> Iterator[str]:
    env = os.environ.get("PHONON25_LIB")
    if env:
        yield env
    here = Path(__file__).resolve().parent
    for name in ("libphonon25.dylib", "libphonon25.so"):
        yield str(here.parent / "csrc" / name)
        yield str(here / name)
    found = ctypes.util.find_library("phonon25")
    if found:
        yield found


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """Load libphonon25 (once) and declare its signatures. Call with a path to override the search."""
    global _c
    if _c is not None and path is None:
        return _c
    tried = []
    for cand in ([path] if path else _candidates()):
        tried.append(cand)
        if os.sep in cand and not os.path.exists(cand):
            continue   # a path that is not there; bare names go to the loader
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    else:
        raise OSError(f"libphonon25 not found; set PHONON25_LIB or run `make -C csrc`. Tried: {tried}")

    fp = ctypes.POINTER(ctypes.c_float)
    lib.p25_engine_open.restype = ctypes.c_void_p
    lib.p25_engine_open.argtypes = [ctypes.c_char_p]
    lib.p25_engine_close.argtypes = [ctypes.c_void_p]
    lib.p25_engine_model.restype = ctypes.c_void_p
    lib.p25_engine_model.argtypes = [ctypes.c_void_p]
    lib.p25_cache_codes.restype = ctypes.c_size_t
    lib.p25_cache_codes.argtypes = [ctypes.c_void_p]
    lib.p25_transcribe.restype = ctypes.c_int
    lib.p25_transcribe.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(_P25Result)]
    lib.p25_result_free.argtypes = [ctypes.POINTER(_P25Result)]
    lib.p25_wav_read.restype = ctypes.c_int
    lib.p25_wav_read.argtypes = [ctypes.c_char_p, ctypes.POINTER(fp), ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.p25_resample.restype = ctypes.c_int
    lib.p25_resample.argtypes = [fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(fp),
                                ctypes.POINTER(ctypes.c_int)]
    lib.p25_pcm16le_to_float.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, fp]
    lib.p25_stream_open.restype = ctypes.c_void_p
    lib.p25_stream_open.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.p25_stream_close.argtypes = [ctypes.c_void_p]
    lib.p25_stream_feed.restype = ctypes.c_int
    lib.p25_stream_feed.argtypes = [ctypes.c_void_p, fp, ctypes.c_int]
    lib.p25_stream_next.restype = ctypes.POINTER(_P25Segment)
    lib.p25_stream_next.argtypes = [ctypes.c_void_p]
    lib.p25_stream_partial.restype = ctypes.c_void_p   # char * we must free
    lib.p25_stream_partial.argtypes = [ctypes.c_void_p]
    lib.p25_stream_pending_seconds.restype = ctypes.c_double
    lib.p25_stream_pending_seconds.argtypes = [ctypes.c_void_p]
    lib.p25_stream_finish.restype = ctypes.c_int
    lib.p25_stream_finish.argtypes = [ctypes.c_void_p]
    lib.p25_stream_result.restype = ctypes.POINTER(_P25Result)
    lib.p25_stream_result.argtypes = [ctypes.c_void_p]
    _c = lib
    return lib


def _libc_free(ptr) -> None:
    """free() for buffers the engine malloc'd and handed over (same C runtime as the library on these platforms)."""
    libc = ctypes.CDLL(None)
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free(ptr)


def set_verbose(on: bool) -> None:
    """The engine's progress lines on stderr (p25_verbose). Off by default from Python."""
    ctypes.c_int.in_dll(load_library(), "p25_verbose").value = 1 if on else 0


# ── Samples in and out ────────────────────────────────────────────────────────

Samples = Union["array.array[float]", bytes, bytearray, memoryview, Sequence[float], "numpy.ndarray"]


class _Floats:
    """A contiguous float32 view of whatever the caller passed, kept alive for the duration of a call."""

    def __init__(self, samples: Samples):
        self.keep = None
        np = sys.modules.get("numpy")
        if np is not None and isinstance(samples, np.ndarray):
            arr = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
            self.keep = arr
            self.ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self.n = int(arr.size)
            return
        if isinstance(samples, array.array) and samples.typecode == "f":
            buf = samples
        elif isinstance(samples, (bytes, bytearray, memoryview)):
            buf = array.array("f")
            buf.frombytes(bytes(samples))
        else:
            buf = array.array("f", samples)
        self.keep = buf
        addr, n = buf.buffer_info()
        self.ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_float))
        self.n = n


def _floats_to_python(ptr, n: int):
    """Engine-allocated float32 samples → numpy array if numpy is loaded, else array('f'); frees the buffer."""
    np = sys.modules.get("numpy")
    if np is not None:
        out = np.ctypeslib.as_array(ptr, shape=(n,)).copy()
    else:
        out = array.array("f", bytes(ctypes.string_at(ptr, n * 4)))
    _libc_free(ctypes.cast(ptr, ctypes.c_void_p))
    return out


# ── Audio utilities ───────────────────────────────────────────────────────────

def read_wav(path: Union[str, os.PathLike]):
    """A WAV file → (mono float32 samples, rate). numpy array when numpy is loaded, else array('f')."""
    c = load_library()
    ptr = ctypes.POINTER(ctypes.c_float)()
    n, rate, channels = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if c.p25_wav_read(os.fsencode(os.fspath(path)), ctypes.byref(ptr), ctypes.byref(n), ctypes.byref(rate),
                     ctypes.byref(channels)) != 0:
        raise RuntimeError(f"cannot read {path}")
    return _floats_to_python(ptr, n.value), rate.value


def resample(samples: Samples, rate_in: int, rate_out: int):
    """Windowed-sinc polyphase resampling (the engine's own)."""
    c = load_library()
    f = _Floats(samples)
    ptr = ctypes.POINTER(ctypes.c_float)()
    n = ctypes.c_int()
    if c.p25_resample(f.ptr, f.n, int(rate_in), int(rate_out), ctypes.byref(ptr), ctypes.byref(n)) != 0:
        raise RuntimeError(f"cannot resample {rate_in} → {rate_out} Hz")
    return _floats_to_python(ptr, n.value)


def pcm16le_to_float(data: bytes, channels: int = 1):
    """Raw 16-bit little-endian PCM → mono float32 samples."""
    c = load_library()
    n = len(data) // (2 * channels)
    out = array.array("f", bytes(4 * n))
    addr, _ = out.buffer_info()
    c.p25_pcm16le_to_float(data, n, channels, ctypes.cast(addr, ctypes.POINTER(ctypes.c_float)))
    np = sys.modules.get("numpy")
    return np.frombuffer(out, dtype=np.float32).copy() if np is not None else out


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class Word:
    text: str
    start: float   # seconds
    end: float


@dataclass
class Token:
    id: int
    frame: int     # encoder frame (80 ms) it was emitted at
    duration: int  # frames the decoder then advanced, 0..4


@dataclass
class Segment:
    start: float   # seconds
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)


@dataclass
class Result:
    text: str
    words: List[Word]
    tokens: List[Token]
    segments: List[Segment]
    encoder_frames: int
    duration_seconds: float
    t_mel: float
    t_encoder: float
    t_decoder: float

    @property
    def seconds(self) -> float:
        """Wall time spent in the engine."""
        return self.t_mel + self.t_encoder + self.t_decoder


def _words(c: _P25Result, first: int, count: int) -> List[Word]:
    return [Word(c.words[i].text.decode("utf-8", "replace"), c.words[i].start, c.words[i].end)
            for i in range(first, first + count)]


def _tokens(c: _P25Result, first: int, count: int) -> List[Token]:
    return [Token(c.tokens[i].id, c.tokens[i].frame, c.tokens[i].duration) for i in range(first, first + count)]


def _segment(c: _P25Result, s: _P25Segment) -> Segment:
    return Segment(s.start, s.end, s.text.decode("utf-8", "replace") if s.text else "",
                   _words(c, s.first_word, s.n_words), _tokens(c, s.first_token, s.n_tokens))


def _result(c: _P25Result) -> Result:
    return Result(c.text.decode("utf-8", "replace") if c.text else "", _words(c, 0, c.n_words),
                  _tokens(c, 0, c.n_tokens), [_segment(c, c.segments[i]) for i in range(c.n_segments)],
                  c.encoder_frames, c.duration_seconds, c.t_mel, c.t_encoder, c.t_decoder)


# ── The stream ────────────────────────────────────────────────────────────────

class Stream:
    """Live transcription at window granularity: feed samples as they arrive, get each 25–35 s window as it
    completes. Made by Engine.stream()."""

    def __init__(self, engine: "Engine", rate: int = 16000):
        self._c = engine._c
        self._engine = engine   # keeps the engine alive
        self._h = self._c.p25_stream_open(engine._h, int(rate))
        if not self._h:
            raise RuntimeError(f"cannot open a stream at {rate} Hz")

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._c.p25_stream_close(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _drain(self) -> List[Segment]:
        res = self._c.p25_stream_result(self._h).contents
        out = []
        while True:
            seg = self._c.p25_stream_next(self._h)
            if not seg:
                return out
            out.append(_segment(res, seg.contents))

    def feed(self, samples: Samples) -> List[Segment]:
        """Append samples; returns the windows completed by this call (usually none)."""
        f = _Floats(samples)
        if self._c.p25_stream_feed(self._h, f.ptr, f.n) < 0:
            raise RuntimeError("stream decoding failed")
        return self._drain()

    @property
    def pending_seconds(self) -> float:
        return self._c.p25_stream_pending_seconds(self._h)

    def partial(self) -> str:
        """Provisional transcript of the audio pending since the last window cut."""
        p = self._c.p25_stream_partial(self._h)
        if not p:
            raise RuntimeError("stream decoding failed")
        try:
            return ctypes.string_at(p).decode("utf-8", "replace")
        finally:
            _libc_free(p)

    def finish(self) -> List[Segment]:
        """Decode the pending audio as the last window; returns the windows that completed. Ends the stream."""
        if self._c.p25_stream_finish(self._h) < 0:
            raise RuntimeError("stream decoding failed")
        return self._drain()

    def result(self) -> Result:
        """Everything so far (text complete after finish())."""
        return _result(self._c.p25_stream_result(self._h).contents)


# ── The engine ────────────────────────────────────────────────────────────────

class Engine:
    """A loaded model. Thread-safe: transcribe() may be called from several threads at once."""

    def __init__(self, weights_dir: Union[str, os.PathLike], cache_codes: bool = False, verbose: bool = False):
        self._c = load_library()
        set_verbose(verbose)
        self._h = self._c.p25_engine_open(os.fsencode(os.fspath(weights_dir)))
        if not self._h:
            raise RuntimeError(f"cannot load the engine from {weights_dir}")
        self.cached_bytes = self._c.p25_cache_codes(self._c.p25_engine_model(self._h)) if cache_codes else 0

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._c.p25_engine_close(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def transcribe(self, samples: Samples, rate: int = 16000) -> Result:
        """Mono float32 samples at `rate` → Result (text, words with times, tokens, windows)."""
        f = _Floats(samples)
        res = _P25Result()
        if self._c.p25_transcribe(self._h, f.ptr, f.n, int(rate), ctypes.byref(res)) != 0:
            raise RuntimeError("transcription failed")
        try:
            return _result(res)
        finally:
            self._c.p25_result_free(ctypes.byref(res))

    def transcribe_file(self, path: Union[str, os.PathLike]) -> Result:
        """Any WAV file (PCM 8/16/24/32-bit or float, any rate, channels averaged to mono)."""
        samples, rate = read_wav(path)
        return self.transcribe(samples, rate)

    def stream(self, rate: int = 16000) -> Stream:
        return Stream(self, rate)


if __name__ == "__main__":   # python3 phonon25.py c_weights recording.wav
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <weights_dir> <audio.wav>")
    with Engine(sys.argv[1]) as e:
        r = e.transcribe_file(sys.argv[2])
    print(r.text)
    for w in r.words:
        print(f"{w.start:7.2f} {w.end:7.2f}  {w.text}")
    print(f"{r.duration_seconds:.1f} s of audio in {r.seconds:.2f} s", file=sys.stderr)
