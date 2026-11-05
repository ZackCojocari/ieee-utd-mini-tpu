"""Writes every test-vector set under vectors/ from the golden model. See README.md.

The image layout is encoded here per Contract 2; generation does not depend on
whether model.pack has been implemented yet. Numerical answers only use golden.
"""
from pathlib import Path

import numpy as np

from model.golden import layer, tile


VECTORS = Path("vectors")
EXPORT = Path("model/export/mlp_int8.npz")


def _bytes_word(values):
    """Pack eight signed bytes, with element zero in the least significant byte."""
    return sum((int(v) & 0xFF) << (8 * i) for i, v in enumerate(values))


def _a_image(A):
    M, K = A.shape
    return [_bytes_word(A[8 * mt:8 * mt + 8, k])
            for mt in range(M // 8) for k in range(K)]


def _w_image(W, bias, M0, s):
    K, N = W.shape
    words = []
    for nt in range(N // 8):
        for c in range(8):
            n = 8 * nt + c
            words.append((int(bias[n]) & 0xFFFFFFFF) |
                         (int(M0[n]) << 32) | (int(s[n]) << 48))
        words.extend(_bytes_word(W[k, 8 * nt:8 * nt + 8]) for k in range(K))
    return words


def _o_image(y):
    M, N = y.shape
    return [_bytes_word(y[8 * mt:8 * mt + 8, 8 * nt + c])
            for nt in range(N // 8) for mt in range(M // 8) for c in range(8)]


def _reorder(o_img, m_tiles, n_tiles):
    return [o_img[(nt * m_tiles + mt) * 8 + c]
            for mt in range(m_tiles) for nt in range(n_tiles) for c in range(8)]


def _random_int8(rng, shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)


def _epi_row(row):
    return sum((int(v) & 0xFFFFFFFF) << (32 * j) for j, v in enumerate(row))


def _output_path(prefix, name, suffix):
    return VECTORS / f"{prefix}_{name}_{suffix}.hex"


def _check_shape(A, W, bias, M0, s):
    M, K = A.shape
    assert A.dtype == np.int8 and W.dtype == np.int8
    assert M % 8 == 0 and W.shape[0] == K and W.shape[1] % 8 == 0
    N = W.shape[1]
    assert 1 <= K <= 1024 and M // 8 * K <= 8192
    assert bias.shape == M0.shape == s.shape == (N,)
    assert np.all((s >= 1) & (s <= 47))
    acc = A.astype(np.int64) @ W.astype(np.int64) + bias
    assert np.all((acc >= -2**31) & (acc < 2**31))
    return M, K, N


def _export_layers(data):
    """Yield exported layer parameters; layer 1 uses the exported input batch."""
    A = np.asarray(data["batch_x"], dtype=np.int8)
    for index in (1, 2, 3):
        W = np.asarray(data[f"W{index}"], dtype=np.int8)
        bias = np.asarray(data[f"bias{index}"], dtype=np.int32)
        M0 = np.asarray(data[f"M0{index}"], dtype=np.uint16)
        s = np.asarray(data[f"s{index}"], dtype=np.int64)
        z, lo, hi = (int(data[key][index - 1]) for key in ("z", "lo", "hi"))
        yield index, A, W, bias, M0, s, z, lo, hi
        A = layer(A, W, bias, M0, s, z, lo, hi)


# All paths are relative to the repository root (as are the bench vector paths).


def to_hex(value, digits):
    """int -> str: value masked to 4*digits bits, upper-case hex, zero-padded, no prefix."""
    return f"{int(value) & ((1 << (4 * digits)) - 1):0{digits}X}"


def write_hex(path, values, digits):
    """Write one to_hex(v, digits) per line to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as out:
        for value in values:
            out.write(to_hex(value, digits) + "\n")


def gen_tile(name, K, rng, fill=None):
    """Write tile_<name>_{a,w,bias,res}.hex for one random tile; every operand = fill if given."""
    a = _random_int8(rng, (K, 8)) if fill is None else np.full((K, 8), fill, np.int8)
    w = _random_int8(rng, (K, 8)) if fill is None else np.full((K, 8), fill, np.int8)
    bias = (rng.integers(-2**31, 2**31, size=8, dtype=np.int32)
            if fill is None else np.full(8, fill, np.int32))
    res = tile(a, w, bias)
    write_hex(_output_path("tile", name, "a"), map(_bytes_word, a), 16)
    write_hex(_output_path("tile", name, "w"), map(_bytes_word, w), 16)
    write_hex(_output_path("tile", name, "bias"), bias, 8)
    write_hex(_output_path("tile", name, "res"), map(_epi_row, res), 64)


def epi_corners():
    """-> (entries, (z, lo, hi)), including saturation, ties, shifts and clamps."""
    rows = [
        ([0, 1, -1, 3, -3, 2**24 - 1, 2**24, 2**24 + 1], 1, 1, 1),
        ([-2**24 - 1, -2**24, -2**24 + 1, 2**31 - 1,
          -2**31, 0, 100, -100], 65535, 1, 0),
        ([0, 1, -1, 2**24 - 1, -2**24, 2**31 - 1,
          -2**31, 12345], 65535, 47, 1),
        ([2**31 - 1, -2**31, 2**24, -2**24 - 1, 1, -1, 127, -128], 0, 47, 0),
        ([100, -100, 2**24, -2**24 - 5, 0, 1, -1, 3], 32768, 16, 1),
        ([127, -128, 0, 1, -1, 2**24 - 1, -2**24, 2], 1, 1, 0),
    ]
    return [(np.asarray(row, dtype=np.int32), m0, sh, last)
            for row, m0, sh, last in rows], (0, -128, 127)


def epi_random(rng, n):
    """-> (entries, (z, lo, hi)), with INT32 and saturation-boundary sums."""
    entries = []
    for _ in range(n):
        row = rng.integers(-2**31, 2**31, size=8, dtype=np.int32)
        for j in range(4):
            edge = 2**24 if rng.integers(2) else -2**24
            row[j] = edge + int(rng.integers(-300, 301))
        m0 = int(rng.choice([0, 65535])) if rng.integers(10) == 0 else int(rng.integers(0, 65536))
        sh = int(rng.choice([1, 47])) if rng.integers(10) == 0 else int(rng.integers(1, 48))
        entries.append((row, m0, sh, int(rng.integers(2))))
    return entries, (int(rng.integers(-128, 128)), -128, 127)


def gen_epi(name, entries, z, lo, hi):
    """Write epi_<name>_{in,out,cfg}.hex; expected words come from layer()."""
    inputs, outputs = [], []
    zero_a = np.zeros((1, 1), np.int8)
    zero_w = np.zeros((1, 8), np.int8)
    for row, m0, sh, last in entries:
        assert np.asarray(row).shape == (8,) and 0 <= m0 <= 65535
        assert 1 <= sh <= 47 and last in (0, 1)
        inputs.append((last << 278) | (sh << 272) | (m0 << 256) | _epi_row(row))
        y = layer(zero_a, zero_w, np.asarray(row, np.int32),
                  np.full(8, m0, np.uint16), np.full(8, sh, np.int64), z, lo, hi)
        outputs.append(_bytes_word(y[0]))
    write_hex(_output_path("epi", name, "in"), inputs, 70)
    write_hex(_output_path("epi", name, "out"), outputs, 16)
    write_hex(_output_path("epi", name, "cfg"), (z, lo, hi), 2)


def random_layer(M, K, N, rng):
    """-> (A, W, bias, M0, s, z, lo, hi) with INT32 sums."""
    assert M % 8 == N % 8 == 0 and 1 <= K <= 1024 and M // 8 * K <= 8192
    A = _random_int8(rng, (M, K))
    W = _random_int8(rng, (K, N))
    bias = rng.integers(-20000, 20001, size=N, dtype=np.int32)
    M0 = rng.integers(0, 65536, size=N, dtype=np.uint16)
    # Mostly useful output range, with occasional extreme shifts to check the edges.
    s = rng.integers(20, 29, size=N, dtype=np.int64)
    extremes = rng.random(N) < 0.1
    s[extremes] = rng.integers(1, 48, size=int(extremes.sum()))
    z = int(rng.integers(-32, 33))
    lo = z if rng.integers(2) else -128
    hi = 127
    _check_shape(A, W, bias, M0, s)
    return A, W, bias, M0, s, z, lo, hi


def gen_layer(name, A, W, bias, M0, s, z, lo, hi):
    """Write layer_<name>_{a,w,o,cfg}.hex; o_img comes from layer()."""
    M, K, N = _check_shape(A, W, bias, M0, s)
    y = layer(A, W, bias, M0, s, z, lo, hi)
    write_hex(_output_path("layer", name, "a"), _a_image(A), 16)
    write_hex(_output_path("layer", name, "w"), _w_image(W, bias, M0, s), 16)
    write_hex(_output_path("layer", name, "o"), _o_image(y), 16)
    write_hex(_output_path("layer", name, "cfg"), (M // 8, K, N // 8, z, lo, hi), 8)


def gen_reorder_header(path, rng):
    """Write a C header with one o_img and its Contract 2 reorder as uint64_t arrays."""
    m_tiles, n_tiles = 3, 2  # Non-square to expose swapped tile indices.
    o_img = [int(v) for v in rng.integers(0, 2**64, size=m_tiles * n_tiles * 8, dtype=np.uint64)]
    a_next = _reorder(o_img, m_tiles, n_tiles)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as out:
        out.write("/* Generated by python -m model.gen_vectors; do not edit. */\n"
                  "#ifndef REORDER_TEST_H\n#define REORDER_TEST_H\n#include <stdint.h>\n\n"
                  f"#define REORDER_M_TILES {m_tiles}\n#define REORDER_N_TILES {n_tiles}\n\n")
        for name, words in (("o_img", o_img), ("a_next", a_next)):
            out.write(f"static const uint64_t {name}[{len(words)}] = {{\n")
            for word in words:
                out.write(f"    UINT64_C(0x{to_hex(word, 16)}),\n")
            out.write("};\n\n")
        out.write("#endif /* REORDER_TEST_H */\n")


def main():
    """Generate every available set listed in README.md with a fixed seed."""
    rng = np.random.default_rng(0x20250401)
    for K in (1, 2, 9, 300):
        gen_tile(f"k{K}", K, rng)
    gen_tile("corner", 300, rng, fill=-128)
    entries, cfg = epi_corners()
    gen_epi("corners", entries, *cfg)
    entries, cfg = epi_random(rng, 1000)
    gen_epi("random", entries, *cfg)
    for name, shape in (("single", (8, 9, 8)), ("mid", (16, 64, 16))):
        gen_layer(name, *random_layer(*shape, rng))
    if EXPORT.exists():
        with np.load(EXPORT) as data:
            for index, A, W, bias, M0, s, z, lo, hi in _export_layers(data):
                                gen_layer(f"demo{index}", A, W, bias, M0, s, z, lo, hi)
    else:
        gen_layer("demo1", *random_layer(64, 784, 128, rng))
    for i in range(50):
        mt = int(rng.integers(1, 5))
        nt = int(rng.integers(1, 5))
        K = int(rng.integers(1, 65))
        gen_layer(f"rand_{i:02d}", *random_layer(8 * mt, K, 8 * nt, rng))
    gen_reorder_header("sw/demo/data/reorder_test.h", rng)

