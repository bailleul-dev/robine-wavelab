#!/usr/bin/env python3
"""Render a listening set: robine-nam vs the upstream reference, as WAV files.

For every model x input x size it writes, in <out>/<model>/<input>/:
  robine-<size>.wav        robine-nam render
  ref-generic-<size>.wav   upstream NeuralAmpModelerCore (generic WaveNet) render
  diff-<size>.wav          (robine - reference) amplified by --diff-gain-db
  ab-<size>.wav            blind A/B: alternates the two renders every
                           --ab-segment seconds in a random order
  ab-<size>.key.json       which implementation plays in which segment
  metrics.json             sample-exact deltas (max abs, SNR, ESR)
and an INDEX.md at the root of <out>.

Both renders get the same output gain: the model's loudness normalised to
--target-db (NAM convention: -18 dBFS), so files play at a sane level.

Needs the release build of the tools (make build) and the assets of a
robine-nam checkout (next to this repository, or $ROBINE_NAM). Stdlib only.

Examples:
  scripts/listening_set.py                              # all assets/models/**/*.nam x 2 DIs, 10 s
  scripts/listening_set.py --models "assets/models/AC30HWH-6580/TopBoost*.nam" --inputs "assets/inputs/Mayer*.wav" --seconds 20
  scripts/listening_set.py --impl-b ref-fast            # compare against the upstream A2 fast path instead
"""
from __future__ import annotations

import argparse
import os
import glob
import json
import math
import random
import re
import struct
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
NAM_LAB = LAB / "target" / "release" / "nam-lab"
# The robine-nam checkout, whose assets/ holds the models and the DI inputs.
REPO = Path(os.environ.get("ROBINE_NAM", LAB.parent / "robine-nam")).resolve()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def slug(text: str) -> str:
    text = re.sub(r"\.(nam|wav)$", "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text or "x"


# --- minimal float32 mono WAV I/O (what nam-lab render writes) -------------

def read_wav_f32(path: Path) -> tuple[list[float], int]:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    pos = 12
    fmt = None
    samples: list[float] = []
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = list(struct.unpack("<HHIIHH", body[:16]))
            if fmt[0] == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: real code in the sub-format GUID
                fmt[0] = struct.unpack("<H", body[24:26])[0]
        elif cid == b"data":
            if fmt is None:
                raise ValueError(f"{path}: data before fmt")
            audio_format, channels, rate, _, _, bits = fmt
            if audio_format == 3 and bits == 32:
                samples = list(struct.unpack(f"<{len(body) // 4}f", body[: len(body) // 4 * 4]))
            elif audio_format == 1 and bits == 16:
                samples = [v / 32768.0 for v in struct.unpack(f"<{len(body) // 2}h", body[: len(body) // 2 * 2])]
            elif audio_format == 1 and bits == 24:
                n = len(body) // 3
                samples = [int.from_bytes(body[i * 3:i * 3 + 3], "little", signed=True) / 8388608.0 for i in range(n)]
            else:
                raise ValueError(f"{path}: unsupported WAV format {audio_format}/{bits}")
            if channels > 1:
                samples = samples[::channels]
            return samples, rate
        pos += 8 + size + (size & 1)
    raise ValueError(f"{path}: no data chunk")


def write_wav_f32(path: Path, samples: list[float], rate: int) -> None:
    body = struct.pack(f"<{len(samples)}f", *samples)
    header = b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32)
    header += b"data" + struct.pack("<I", len(body))
    path.write_bytes(header + body)


# --- rendering -------------------------------------------------------------

def model_loudness(path: Path) -> float | None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for candidate in [doc] + [sm.get("model", {}) for sm in doc.get("config", {}).get("submodels", [])]:
        meta = candidate.get("metadata") or {}
        if isinstance(meta.get("loudness"), (int, float)):
            return float(meta["loudness"])
    return None


def render(model: Path, wav: Path, out: Path, impl: str, size: str, seconds: float, output_db: float) -> None:
    cmd = [str(NAM_LAB), "render", "-m", str(model), "-i", str(wav), "-o", str(out),
           "--impl", impl, "--size", size, "--block", "64", f"--output-db={output_db:.3f}"]
    if seconds > 0:
        cmd += ["--seconds", str(seconds)]
    subprocess.run(cmd, check=True, capture_output=True)


def sizes_of(model: Path) -> list[str]:
    out = subprocess.run([str(NAM_LAB), "inspect", str(model)], capture_output=True, text=True).stdout
    return re.findall(r"variant (small|large)", out)


def metrics(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    max_abs = 0.0
    err2 = 0.0
    ref2 = 0.0
    for i in range(n):
        d = a[i] - b[i]
        max_abs = max(max_abs, abs(d))
        err2 += d * d
        ref2 += b[i] * b[i]
    esr = err2 / ref2 if ref2 > 0 else float("inf")
    snr = 10 * math.log10(ref2 / err2) if err2 > 0 else float("inf")
    return {"samples": n, "max_abs": max_abs, "esr": esr, "snr_db": snr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=[str(REPO / "assets/models/**/*.nam")], help="globs of .nam files")
    parser.add_argument("--inputs", nargs="*", default=[str(REPO / "assets/inputs/Brit - Guitar.wav"), str(REPO / "assets/inputs/Mayer - Guitar.wav")], help="globs of DI WAVs")
    parser.add_argument("--out", type=Path, default=REPO / "assets/listening")
    parser.add_argument("--seconds", type=float, default=10.0, help="seconds of DI to render (0 = whole file)")
    parser.add_argument("--sizes", default="small,large")
    parser.add_argument("--impl-a", default="robine")
    parser.add_argument("--impl-b", default="ref-generic", help="robine | ref-generic | ref-fast | ref-auto")
    parser.add_argument("--target-db", type=float, default=-18.0, help="loudness normalisation target (dBFS)")
    parser.add_argument("--diff-gain-db", type=float, default=60.0)
    parser.add_argument("--ab-segment", type=float, default=2.0, help="seconds per A/B segment")
    parser.add_argument("--seed", type=int, default=None, help="A/B order seed (default: random)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not NAM_LAB.exists():
        raise SystemExit(f"{NAM_LAB} not found: run `make build` first")
    models = sorted({Path(p) for g in args.models for p in glob.glob(g, recursive=True)})
    inputs = sorted({Path(p) for g in args.inputs for p in glob.glob(g, recursive=True)})
    if not models or not inputs:
        raise SystemExit(f"nothing to do: {len(models)} model(s), {len(inputs)} input(s)")
    if args.limit:
        models = models[: args.limit]
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    index = ["# Listening set", "", f"A = `{args.impl_a}`, B = `{args.impl_b}`. Output normalised to {args.target_db:g} dBFS, diff amplified by {args.diff_gain_db:g} dB, A/B segments of {args.ab_segment:g} s in random order (key in the `.key.json`).", "",
             "| model | input | size | max abs | SNR dB | files |", "|---|---|---|---:|---:|---|"]

    for model in models:
        loud = model_loudness(model)
        output_db = (args.target_db - loud) if loud is not None else 0.0
        available = sizes_of(model)
        for wav in inputs:
            dest = args.out / slug(model.stem) / slug(wav.stem)
            dest.mkdir(parents=True, exist_ok=True)
            all_metrics = {}
            for size in sizes:
                if size not in available:
                    continue
                a_path = dest / f"{args.impl_a}-{size}.wav"
                b_path = dest / f"{args.impl_b}-{size}.wav"
                if args.overwrite or not a_path.exists():
                    render(model, wav, a_path, args.impl_a, size, args.seconds, output_db)
                if args.overwrite or not b_path.exists():
                    render(model, wav, b_path, args.impl_b, size, args.seconds, output_db)
                a, rate = read_wav_f32(a_path)
                b, _ = read_wav_f32(b_path)
                n = min(len(a), len(b))
                gain = 10 ** (args.diff_gain_db / 20)
                write_wav_f32(dest / f"diff-{size}.wav", [(a[i] - b[i]) * gain for i in range(n)], rate)

                seg = max(1, int(args.ab_segment * rate))
                ab: list[float] = []
                key = []
                fade = min(64, seg // 8)
                order: list[str] = []
                for start in range(0, n, seg):
                    if not order:  # every pair of segments has one A and one B, in random order
                        order = rng.choice([["A", "B"], ["B", "A"]])
                    pick = order.pop(0)
                    src = a if pick == "A" else b
                    chunk = src[start:start + seg]
                    for i in range(min(fade, len(chunk))):  # short fades to avoid clicks
                        w = i / fade
                        chunk[i] *= w
                        chunk[-1 - i] *= w
                    ab.extend(chunk)
                    key.append({"start_s": start / rate, "plays": pick})
                write_wav_f32(dest / f"ab-{size}.wav", ab, rate)
                (dest / f"ab-{size}.key.json").write_text(json.dumps({"A": args.impl_a, "B": args.impl_b, "segments": key}, indent=1))

                m = metrics(a, b)
                all_metrics[size] = m
                rel = dest.relative_to(args.out)
                index.append(f"| {model.stem} | {wav.stem} | {size} | {m['max_abs']:.2e} | {m['snr_db']:.1f} | `{rel}/` |")
                log(f"{model.stem} / {wav.stem} / {size}: max abs {m['max_abs']:.2e}, SNR {m['snr_db']:.1f} dB -> {rel}")
            (dest / "metrics.json").write_text(json.dumps({"model": str(model), "input": str(wav), "output_db": output_db, "A": args.impl_a, "B": args.impl_b, "sizes": all_metrics}, indent=1))
    (args.out / "INDEX.md").write_text("\n".join(index) + "\n")
    log(f"index: {args.out / 'INDEX.md'}")


if __name__ == "__main__":
    main()
