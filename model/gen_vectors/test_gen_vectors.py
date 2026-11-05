"""Checks generated image layout against Contract 2 and arithmetic against golden."""
from pathlib import Path
import re

import numpy as np

from model.golden import layer, tile
from model.gen_vectors import gen_vectors as g


def lines(path):
    return Path(path).read_text(encoding="ascii").splitlines()


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def test_to_hex_masks_negative():
    assert (g.to_hex(-1, 8), g.to_hex(-128, 2), g.to_hex(5, 16)) == (
        "FFFFFFFF", "80", "0000000000000005")


def test_tile_file_shapes():
    for name, k in (("k1", 1), ("k2", 2), ("k9", 9), ("k300", 300), ("corner", 300)):
        for suffix, count, digits in (("a", k, 16), ("w", k, 16),
                                      ("bias", 8, 8), ("res", 8, 64)):
            data = lines(g._output_path("tile", name, suffix))
            assert len(data) == count
            assert all(re.fullmatch(rf"[0-9A-F]{{{digits}}}", v) for v in data)


def test_tile_res_matches_golden():
    for name in ("k1", "k2", "k9", "k300", "corner"):
        def words(suffix):
            return [int(v, 16) for v in lines(g._output_path("tile", name, suffix))]
        def bytes8(values):
            return np.array([[signed((v >> (8 * j)) & 255, 8) for j in range(8)]
                             for v in values], np.int8)
        bias = np.array([signed(v, 32) for v in words("bias")], np.int32)
        res = tile(bytes8(words("a")), bytes8(words("w")), bias)
        for c, word in enumerate(words("res")):
            assert [signed((word >> (32 * j)) & 0xFFFFFFFF, 32) for j in range(8)] == res[c].tolist()


def test_epi_out_matches_layer():
    for name in ("corners", "random"):
        z, lo, hi = [signed(int(x, 16), 8) for x in lines(g._output_path("epi", name, "cfg"))]
        inp = lines(g._output_path("epi", name, "in"))
        out = lines(g._output_path("epi", name, "out"))
        assert len(inp) == len(out) == (1000 if name == "random" else 6)
        for text, expected in zip(inp, out):
            assert len(text) == 70
            v = int(text, 16)
            assert v >> 279 == 0
            sh, m0 = (v >> 272) & 63, (v >> 256) & 65535
            row = np.array([signed((v >> (32 * j)) & 0xFFFFFFFF, 32) for j in range(8)], np.int32)
            actual = layer(np.zeros((1, 1), np.int8), np.zeros((1, 8), np.int8), row,
                           np.full(8, m0, np.uint16), np.full(8, sh, np.int64), z, lo, hi)
            assert int(expected, 16) == sum((int(x) & 255) << (8 * j) for j, x in enumerate(actual[0]))


def test_epi_corners_cover_cases():
    entries, cfg = g.epi_corners()
    sums = {int(x) for row, _, _, _ in entries for x in row}
    assert {2**24 - 1, 2**24, 2**24 + 1, -2**24 - 1, -2**24} <= sums
    assert {1, 47} <= {sh for _, _, sh, _ in entries}
    assert {0, 65535} <= {m0 for _, m0, _, _ in entries}
    assert any(m0 == 1 and sh == 1 and {1, -1} <= set(row.tolist()) for row, m0, sh, _ in entries)
    outs = [layer(np.zeros((1, 1), np.int8), np.zeros((1, 8), np.int8), row,
                  np.full(8, m0, np.uint16), np.full(8, sh, np.int64), *cfg)
            for row, m0, sh, _ in entries]
    assert any(-128 in o for o in outs) and any(127 in o for o in outs)


def test_layer_sets_match_golden():
    names = ["single", "mid", "demo1"] + [f"rand_{i:02d}" for i in range(50)]
    if g.EXPORT.exists():
        names += ["demo2", "demo3"]
    for name in names:
        mt, k, nt, z, lo, hi = (int(x, 16) for x in lines(g._output_path("layer", name, "cfg")))
        z, lo, hi = (signed(x, 32) for x in (z, lo, hi))
        a, w, out = ([int(x, 16) for x in lines(g._output_path("layer", name, suffix))]
                     for suffix in ("a", "w", "o"))
        assert (len(a), len(w), len(out)) == (mt * k, nt * (k + 8), mt * nt * 8)
        A = np.empty((8 * mt, k), np.int8)
        W = np.empty((k, 8 * nt), np.int8)
        bias, m0, sh = np.empty(8 * nt, np.int32), np.empty(8 * nt, np.uint16), np.empty(8 * nt, np.int64)
        for m in range(mt):
            for t in range(k):
                v = a[m * k + t]
                for j in range(8):
                    A[8 * m + j, t] = signed((v >> (8 * j)) & 255, 8)
        for n in range(nt):
            base = n * (k + 8)
            for c in range(8):
                v = w[base + c]
                assert v >> 54 == 0
                bias[8 * n + c] = signed(v & 0xFFFFFFFF, 32)
                m0[8 * n + c] = (v >> 32) & 65535
                sh[8 * n + c] = (v >> 48) & 63
            for t in range(k):
                v = w[base + 8 + t]
                for c in range(8):
                    W[t, 8 * n + c] = signed((v >> (8 * c)) & 255, 8)
        y = layer(A, W, bias, m0, sh, z, lo, hi)
        expected = [sum((int(y[8 * m + j, 8 * n + c]) & 255) << (8 * j) for j in range(8))
                    for n in range(nt) for m in range(mt) for c in range(8)]
        assert out == expected


def test_reorder_header_matches_reorder():
    text = Path("sw/demo/data/reorder_test.h").read_text(encoding="ascii")
    mt = int(re.search(r"#define REORDER_M_TILES (\d+)", text).group(1))
    nt = int(re.search(r"#define REORDER_N_TILES (\d+)", text).group(1))
    arrays = re.findall(r"static const uint64_t (\w+)\[\d+\] = \{([^}]+)\}", text)
    values = {name: [int(x, 16) for x in re.findall(r"0x([0-9A-F]{16})", body)]
              for name, body in arrays}
    assert len(values["o_img"]) == len(values["a_next"]) == mt * nt * 8
    assert values["a_next"] == [values["o_img"][(n * mt + m) * 8 + c]
                                 for m in range(mt) for n in range(nt) for c in range(8)]


def test_generation_is_deterministic(tmp_path, monkeypatch):
    outputs = []
    for folder in (tmp_path / "one", tmp_path / "two"):
        folder.mkdir()
        monkeypatch.setattr(g, "VECTORS", folder / "vectors")
        monkeypatch.chdir(folder)
        # Keep exported data location anchored to the original repository.
        monkeypatch.setattr(g, "EXPORT", Path(__file__).resolve().parents[1] / "export/mlp_int8.npz")
        g.main()
        outputs.append({p.relative_to(folder): p.read_bytes()
                        for p in folder.rglob("*") if p.is_file()})
    assert outputs[0] == outputs[1]

