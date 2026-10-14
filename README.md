# robine-wavelab

![AI Declaration Format](https://img.shields.io/badge/AI_Declaration_Format-validated-blue)

The measurement bench of the robine audio libraries:
[robine-spice](https://github.com/bailleul-dev/robine-spice) (circuits
simulated from netlists) and
[robine-nam](https://github.com/bailleul-dev/robine-nam) (Neural Amp Modeler
captures). The libraries hold code and tests; everything that measures them
from the outside is here, so that both are held to the same instruments and
can be compared with each other.

Three command-line tools over one library:

| tool | what it does |
|------|--------------|
| `wavelab` | measures on audio files, whatever produced them: `probe`, `judge`, `null` |
| `spice-lab` | robine-spice: `convert` LiveSPICE schematics, `inspect`, `render`, `compare` with ngspice, `survey`, `bench`, `golden`, `alias` (aliasing against a run at a higher rate) |
| `nam-lab` | robine-nam: `inspect`, `render`, `compare` with the upstream NeuralAmpModelerCore, `compare-wav`, `bench`, `golden` |

## Setting up

The two libraries are expected next to this repository:

```
robine-spice/   robine-nam/   robine-wavelab/
```

```sh
make build      # fetches the pinned upstream NAM reference, builds it and the three tools
make test       # format, lint, tests
```

Optional, by target:

| needs | for |
|-------|-----|
| [ngspice](https://ngspice.sourceforge.io) on `PATH` | `spice-compare`, `spice-survey`, `spice-golden` |
| GStreamer and pkg-config (`brew install gstreamer pkgconf`), then `make peaq` | the PEAQ grade of `wavelab null --peaq` and `make judge` |
| the assets of robine-nam (`make assets` there) | `nam-compare-wav`, `nam-listen`, and the guitar take the listening and judging targets use |
| a checkout of [LiveSPICE](https://github.com/dsharlet/LiveSPICE) | `spice-convert` |

Always use release builds; the tools warn otherwise and the timings mean
nothing.

## Judging whether two devices sound alike

`wavelab probe` and `wavelab judge` answer "do these two devices behave
alike?" for anything that turns a WAV into a WAV: a circuit, a NAM capture,
a real pedal reamped. The probe is 22 seconds of steady tones at known
frequencies and levels; pass it through both devices and `judge` compares
what came out, criterion by criterion, with a tolerance on each:

| criterion | what is compared |
|-----------|------------------|
| frequency response | level of twelve small tones from 80 Hz to 7 kHz |
| dynamics | output level against input level, over 36 dB, at 220 and 880 Hz |
| harmonics | harmonics 2 to 8 of each of those tones |
| attack | the first 10 ms and the next 40 ms of a loud tone, against the settled tone |
| recovery | gain of a quiet tone 20, 100 and 500 ms after a loud burst |

Overall level is removed first and reported separately.

Those criteria only look at levels of steady tones. `wavelab null` is the
measure that cannot miss a kind of difference, and the one meant for
iterating: it aligns two renders of the same material in time and level,
subtracts them, and rates what is left in two established ways.

* **Error-to-signal ratio (ESR)**, plain and with the `1 - 0.95 z^-1`
  pre-emphasis: the figure neural amp models are trained and reported with.
* **Noise-to-mask ratio (NMR)**: the residual set against the masking
  threshold the reference itself produces in the ear, per quarter-Bark band
  and per frame, with the ear model of the basic version of PEAQ (ITU-R
  BS.1387). Below 0 dB the difference is under the model's threshold, and
  the verdict is whether it ever rises above it. Being computed on spectra,
  it ignores a pure phase difference, which the ESR does not.

* **PEAQ grade** (`--peaq`): the objective difference grade of ITU-R
  BS.1387, from 0 (imperceptible) to -4 (very annoying), computed by
  [GstPEAQ](https://github.com/HSU-ANT/gstpeaq) on the aligned signals.
  `make peaq` fetches and builds it inside `peaq/`; it needs GStreamer
  (`brew install gstreamer pkgconf`). 48 kHz only.

`--json` prints one line for scripts; `--max-esr-db`, `--max-nmr-db`,
`--inaudible` and `--min-odg` turn it into a pass or fail. For scale, on a
guitar take through the Distortion +:

| difference | ESR | NMR | frames above the mask | PEAQ grade |
|------------|----:|----:|----------------------:|-----------:|
| none | | -121 dB | none | +0.20 |
| ideal supply against a real one, Drive at half | -81 dB | -85 dB | none | +0.12 |
| Drive moved from 0.500 to 0.501 | -68 dB | -62 dB | none | +0.10 |
| ideal supply against a real one, Drive at full | -55 dB | -50 dB | none | +0.16 |
| Drive moved from 0.50 to 0.51 | -48 dB | -41 dB | none | +0.10 |
| Drive moved from 0.50 to 0.55 | -33 dB | -26 dB | none | +0.01 |
| Drive moved from 0.50 to 0.60 | -26 dB | -19 dB | 0.1 % | -0.41 |
| Drive moved from 0.50 to 0.70 | -18 dB | -9 dB | 24 % | -2.04 |
| Drive moved from 0.50 to 0.85 | -10 dB | +5 dB | 89 % | -3.81 |
| a Rat instead | -3 dB | +22 dB | all of them | -3.91 |

Two readings of that table. The PEAQ grade says whether a difference is
heard, but it is flat near zero: it cannot rank two small differences, so it
is the verdict, and the ESR and NMR are the ruler to iterate with. And the
NMR computed here, an implementation of the published description of the
model (P. Kabal, "An Examination and Interpretation of ITU-R BS.1387") that
has not been run against the standard's conformance set, starts to find
frames above the mask exactly where the independent PEAQ grade leaves zero.
GstPEAQ itself is close to the standard without being certified.

`make judge REF=a.cir CAND=b.cir` runs the criteria and the null test for
two netlists. For anything else, render the probe and a take of real playing
through both devices and call `wavelab` on the files:

```sh
wavelab probe -o probe.wav
nam-lab render -m capture.nam -i probe.wav -o reference.wav
spice-lab render -c circuit.cir -i probe.wav -o candidate.wav
wavelab judge -r reference.wav -c candidate.wav
wavelab null -r reference-guitar.wav -c candidate-guitar.wav --peaq --json
```

The null test aligns the two files in time and level before subtracting,
which is more lenient than the ESR of a NAM training, where the signals are
compared as they are.

## robine-spice

```sh
make spice-compare    # robine-spice vs ngspice at rising sample rates
make spice-survey     # every circuit: speed, solver health, distance from ngspice (minutes)
make spice-bench      # throughput and cost of a control move
make spice-listen     # every circuit rendered to out/listen/spice
make spice-golden     # regenerate the fixtures of robine-spice's tests (runs ngspice)
make spice-convert    # regenerate its netlists from a LiveSPICE checkout
```

`spice-lab compare` gives ngspice the same flattened circuit and the same
samples, lets it solve with a time step at least eight times finer, and
compares at every sample time. `spice-lab render` takes `--set
Control=value`, `--source buffer|single-coil|humbucker`, `--sweep
Control=low:high:hertz` to move a control while rendering, and `--normalize`.

## robine-nam

```sh
make nam-compare      # sample-exact delta against the upstream reference
make nam-bench        # ns/sample, real-time factor, per-block latency percentiles
make nam-listen       # robine vs upstream renders, difference and blind A/B files
make nam-golden       # regenerate the fixtures of robine-nam's tests
```

`crates/nam-reference-sys` compiles the upstream
[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore),
fetched at the commit pinned in its `UPSTREAM` file, and exposes it through a
small C shim, so that the upstream generic WaveNet and its A2 fast path are
driven in-process with the same buffers as robine-nam. `scripts/ab.sh`
benchmarks two builds of `nam-lab` interleaved, which is the only way to
compare timings on a laptop.

## Repository layout

```
crates/wavelab/            the library: probe and criteria, null test, ESR, noise-to-mask ratio, PEAQ runner, deltas, WAV, test signals
crates/wavelab-cli/        `wavelab`
crates/spice-lab/          `spice-lab`, with the ngspice runner (the LiveSPICE converter is robine-spice's schematic crate)
crates/nam-lab/            `nam-lab`
crates/nam-reference-sys/  the upstream NAM reference, built from pinned sources
peaq/setup.sh              builds GstPEAQ from pinned sources into peaq/build
scripts/                   convert-livespice.sh, listening_set.py, ab.sh
out/                       what the targets write (git-ignored)
```

## AI usage

This project was built with intensive use of an AI coding agent (Claude
Code), directed and reviewed by the maintainer. The details are recorded in
[`aidecl.yaml`](aidecl.yaml) following the
[AI Declaration Format](https://ai-declaration.org/); `make aidecl-check`
validates it.

## License

MIT. Fetched at build time and not distributed here: NeuralAmpModelerCore
(MIT, Steven Atkinson) and GstPEAQ (LGPL, Martin Holters), the latter used
as an external program. ngspice is used as an external program too.
