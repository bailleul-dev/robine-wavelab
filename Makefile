# robine-wavelab — measurement tooling for robine-spice and robine-nam.
# Everything audio-related runs in release builds only.

# The libraries under test, checked out next to this repository.
SPICE ?= ../robine-spice
NAM   ?= ../robine-nam
# A checkout of https://github.com/dsharlet/LiveSPICE, for `make spice-convert`.
LIVESPICE ?= ../LiveSPICE

WAVELAB   := target/release/wavelab
SPICE_LAB := target/release/spice-lab
NAM_LAB   := target/release/nam-lab
PEAQ      := peaq/build/peaq
OUT       := out

CIRCUITS  := $(SPICE)/circuits
SPICE_FIX := $(SPICE)/crates/robine-spice/tests/fixtures
NAM_FIX   := $(NAM)/crates/robine-nam/tests/fixtures
ASSETS    := $(NAM)/assets
MODEL     ?= $(NAM_FIX)/A2.nam
INPUT     ?= $(ASSETS)/inputs/Brit - Guitar.wav

.PHONY: all build test fmt clippy clean peaq judge aidecl-check \
	spice-compare spice-survey spice-bench spice-golden spice-convert spice-listen \
	spice-tools spice-grade-measure spice-grade-device spice-grade-check spice-grade-table \
	nam-compare nam-compare-wav nam-bench nam-golden nam-listen

all: build

## Fetches the pinned upstream NAM reference, builds it and the three tools
build:
	crates/nam-reference-sys/vendor.sh
	cargo build --release

test: build
	cargo fmt --all --check
	cargo clippy --release --all-targets -- -D warnings
	cargo test --release

fmt:
	cargo fmt --all

clippy:
	cargo clippy --release --all-targets -- -D warnings

clean:
	cargo clean
	rm -rf $(OUT) peaq/build

## GstPEAQ, for the perceptual grade of `wavelab null --peaq` and `make judge`.
## Needs GStreamer and pkg-config installed first:
##   macOS:  brew install gstreamer pkgconf
##   Debian: apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good
## Fetched at a pinned commit and built inside peaq/; nothing is installed elsewhere.
peaq: $(PEAQ)
$(PEAQ): peaq/setup.sh
	peaq/setup.sh

## Do two circuits behave alike? make judge REF=a.cir CAND=b.cir
## Criteria on a probe signal (frequency response, dynamics, harmonics, attack,
## recovery), then a null test on the guitar take: error-to-signal ratio,
## noise-to-mask ratio and PEAQ grade (exit code != 0 if the difference rises
## above the mask or the grade falls below MIN_ODG).
## For anything else than a circuit, pass $(OUT)/judge/probe.wav through it and
## run `wavelab judge` and `wavelab null` on the results.
VOLTS ?= 0.5
MIN_ODG ?= -0.5
judge: build peaq
	mkdir -p $(OUT)/judge
	$(WAVELAB) probe -o $(OUT)/judge/probe.wav
	$(SPICE_LAB) render -c "$(REF)" -i $(OUT)/judge/probe.wav -o $(OUT)/judge/reference.wav --volts $(VOLTS)
	$(SPICE_LAB) render -c "$(CAND)" -i $(OUT)/judge/probe.wav -o $(OUT)/judge/candidate.wav --volts $(VOLTS)
	-$(WAVELAB) judge -r $(OUT)/judge/reference.wav -c $(OUT)/judge/candidate.wav
	$(SPICE_LAB) render -c "$(REF)" -i "$(INPUT)" -o $(OUT)/judge/reference-guitar.wav --volts $(VOLTS)
	$(SPICE_LAB) render -c "$(CAND)" -i "$(INPUT)" -o $(OUT)/judge/candidate-guitar.wav --volts $(VOLTS)
	$(WAVELAB) null -r $(OUT)/judge/reference-guitar.wav -c $(OUT)/judge/candidate-guitar.wav --inaudible --min-odg $(MIN_ODG)

# ---------------------------------------------------------------- robine-spice

## robine-spice vs ngspice at rising sample rates (needs ngspice on PATH)
spice-compare: build
	$(SPICE_LAB) compare -c $(CIRCUITS)/mxr-distortion-plus.cir --rates 48000,96000,192000
	$(SPICE_LAB) compare -c $(CIRCUITS)/big-muff-pi.cir --peak 0.2 --rates 48000,96000,192000 \
		--ngspice-reltol 1e-5 --threshold-db -60

## Every circuit: does it run, how fast, how far from ngspice (minutes)
spice-survey: build
	$(SPICE_LAB) survey $(CIRCUITS)

## Throughput and cost of a control move
spice-bench: build
	$(SPICE_LAB) bench -c $(CIRCUITS)/mxr-distortion-plus.cir
	$(SPICE_LAB) bench -c $(CIRCUITS)/big-muff-pi.cir --peak 0.3

## The tools robine-spice needs, without the NAM reference (which `build` fetches)
spice-tools:
	cargo build --release -p spice-lab -p wavelab-cli

## Grades of the circuits, from tin to platinum (see $(CIRCUITS)/GRADES.md): the evidence
## and the grades are in $(CIRCUITS)/grades.json.
## Measure the solver, ngspice and the controls (needs ngspice; minutes per circuit
## with many controls). All circuits, or ONLY="big-muff-pi fuzz-face".
spice-grade-measure: spice-tools
	$(SPICE_LAB) grade measure --circuits $(CIRCUITS) $(foreach c,$(ONLY),--only $(c))

## Record a comparison with a real device, from `wavelab null --json --peaq`:
##   make spice-grade-device CIRCUIT=big-muff-pi.cir CAPTURE="nyc.nam (TONE3000 #123)" \
##     SET="Sustain=0.7 Tone=0.5" NULL=out/null.json
spice-grade-device: spice-tools
	$(SPICE_LAB) grade device --circuits $(CIRCUITS) -c $(CIRCUITS)/$(CIRCUIT) \
		--capture "$(CAPTURE)" $(foreach s,$(SET),--set $(s)) --null $(NULL)

## The stored grades are what their evidence gives (exit code != 0 if not)
spice-grade-check: spice-tools
	$(SPICE_LAB) grade rate --circuits $(CIRCUITS) --check

## Rate again, and write the table of the circuits README
spice-grade-table: spice-tools
	$(SPICE_LAB) grade rate --circuits $(CIRCUITS)
	$(SPICE_LAB) grade table --circuits $(CIRCUITS) --readme $(CIRCUITS)/README.md

## Regenerate the golden fixtures of robine-spice's `cargo test` (runs ngspice)
SPICE_GOLDEN = $(SPICE_LAB) golden -o $(SPICE_FIX)
spice-golden: build
	$(SPICE_GOLDEN) -c $(CIRCUITS)/mxr-distortion-plus.cir --name mxr-distortion-plus
	$(SPICE_GOLDEN) -c $(CIRCUITS)/big-muff-pi.cir --name big-muff-pi --peak 0.2 --ngspice-reltol 1e-5
	$(SPICE_GOLDEN) -c $(CIRCUITS)/pro-co-rat.cir --name pro-co-rat --peak 0.3 --ngspice-reltol 1e-5
	$(SPICE_GOLDEN) -c $(CIRCUITS)/marshall-jcm800-2203-preamp.cir --name marshall-jcm800-2203-preamp --peak 0.2 --ngspice-reltol 1e-5
	$(SPICE_GOLDEN) -c $(CIRCUITS)/fender-5e3.cir --name fender-5e3 --peak 0.5 --seconds 0.02 --ngspice-reltol 1e-5
	$(SPICE_GOLDEN) -c $(CIRCUITS)/robine-mosfet-overdrive.cir --name robine-mosfet-overdrive --peak 1 --ngspice-reltol 1e-6
	$(SPICE_GOLDEN) -c $(CIRCUITS)/robine-treble-booster.cir --name robine-treble-booster --peak 0.3 --ngspice-reltol 1e-6
	$(SPICE_GOLDEN) -c $(CIRCUITS)/klon-centaur.cir --name klon-centaur --peak 0.5 --ngspice-reltol 1e-6

## Regenerate robine-spice's netlists from the LiveSPICE schematics
spice-convert: build
	scripts/convert-livespice.sh "$(LIVESPICE)" $(CIRCUITS)

## WAVs to hear, in out/listen/spice: one per circuit (the Fuzz Face also behind
## a guitar and behind a buffer), each normalised to -1 dBFS. The amplifiers
## come out raw (no cabinet).
RENDER = $(SPICE_LAB) render -i "$(INPUT)" --normalize
LISTEN = $(OUT)/listen/spice
spice-listen: build
	mkdir -p $(LISTEN)
	$(RENDER) -c $(CIRCUITS)/mxr-distortion-plus.cir -o $(LISTEN)/distortion-plus.wav --volts 0.5 --set Drive=0.7
	$(RENDER) -c $(CIRCUITS)/big-muff-pi.cir -o $(LISTEN)/big-muff.wav --volts 0.3 --set Sustain=0.7
	$(RENDER) -c $(CIRCUITS)/pro-co-rat.cir -o $(LISTEN)/rat.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/fuzz-face.cir -o $(LISTEN)/fuzz-face.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/fuzz-face.cir -o $(LISTEN)/fuzz-face-guitar.wav --volts 0.3 --source single-coil
	$(RENDER) -c $(CIRCUITS)/fuzz-face.cir -o $(LISTEN)/fuzz-face-guitar-volume-down.wav --volts 0.3 --source single-coil --set Guitar.Volume=0.6
	$(RENDER) -c $(CIRCUITS)/fuzz-face.cir -o $(LISTEN)/fuzz-face-buffer.wav --volts 0.3 --source buffer
	$(RENDER) -c $(CIRCUITS)/boss-super-overdrive-sd-1.cir -o $(LISTEN)/sd-1.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/marshall-blues-breaker.cir -o $(LISTEN)/blues-breaker.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/orange-squeezer-compressor.cir -o $(LISTEN)/orange-squeezer.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/mxr-phase-90-sweeping.cir -o $(LISTEN)/phase-90.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/dunlop-cry-baby-gcb-95.cir -o $(LISTEN)/cry-baby.wav --volts 0.3 --sweep Filter=0.05:0.95:1.2
	$(RENDER) -c $(CIRCUITS)/robine-analog-delay.cir -o $(LISTEN)/analog-delay.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/robine-analog-delay.cir -o $(LISTEN)/analog-delay-runaway.wav --volts 0.3 --set Repeat=1 --set Time=0.4
	$(RENDER) -c $(CIRCUITS)/robine-analog-delay.cir -o $(LISTEN)/analog-delay-time-swept.wav --volts 0.3 --sweep Time=0.45:0.65:0.2
	$(RENDER) -c $(CIRCUITS)/robine-plate-reverb.cir -o $(LISTEN)/plate-reverb.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/robine-plate-reverb.cir -o $(LISTEN)/plate-reverb-long.wav --volts 0.3 --set Decay=0.8 --set Mix=0.5
	$(RENDER) -c $(CIRCUITS)/59-bassman-preamp-plus-tone-stack.cir -o $(LISTEN)/bassman-preamp.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/marshall-jcm800-2203-preamp.cir -o $(LISTEN)/jcm800-preamp.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/marshall-jcm800-2203-preamp-modded.cir -o $(LISTEN)/jcm800-preamp-modded.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/marshall-jcm2000-dsl-preamp.cir -o $(LISTEN)/jcm2000-preamp.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/orange-rockerverb-50-preamp.cir -o $(LISTEN)/rockerverb-preamp.wav --volts 0.3
	$(RENDER) -c $(CIRCUITS)/fender-5e3.cir -o $(LISTEN)/fender-5e3.wav --volts 0.3

# ------------------------------------------------------------------ robine-nam

## Sample-exact delta against the upstream reference (exit code != 0 on failure)
nam-compare: build
	$(NAM_LAB) compare -m "$(MODEL)" --seconds 5

nam-compare-wav: build
	$(NAM_LAB) compare -m "$(MODEL)" -i "$(INPUT)" --blocks 64

## Throughput / latency of robine-nam vs upstream fast and generic paths
nam-bench: build
	mkdir -p $(OUT)
	$(NAM_LAB) bench -m "$(MODEL)" --json $(OUT)/nam-bench.json

## Regenerate the golden fixtures of robine-nam's `cargo test`
nam-golden: build
	$(NAM_LAB) golden -m $(NAM_FIX)/A2.nam -o $(NAM_FIX) --name a2

## Listening set: robine vs upstream renders, diff and blind A/B WAVs
## (in the robine-nam checkout's assets/listening)
nam-listen: build
	python3 -I scripts/listening_set.py $(LISTEN_ARGS)

## Validate the AI usage declaration (https://ai-declaration.org/)
aidecl-check:
	uvx --from git+https://github.com/ai-declaration/cli.git aidecl --strict validate aidecl.yaml
