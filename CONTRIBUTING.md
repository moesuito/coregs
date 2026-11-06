# Contributing to CoreGS

Thank you for helping. CoreGS is a clean-room renderer for the PlayStation 2 GS; two rules matter
more than any style guide: **no code or text from another emulator** and **no game data**.

## The clean-room rule

- Do **not** read, copy, paraphrase or "translate" the source of PCSX2, GSdx, ARMSX2, paraLLEl-GS or
  any other emulator or renderer while working on CoreGS. That includes their tests, comments and
  issue discussions.
- Do **not** paste code or text from those projects into an issue, a pull request or a commit.
- Work from the PS2 hardware documentation (the GS register map, the GIF packet format, the memory
  layout), from public hardware descriptions and from observed behaviour of the hardware or of
  CoreGS's own CPU reference.
- If you have read such a source in the past, say so in the pull request: a maintainer will review
  the change against the clean-room rule before it is merged. A contribution that cannot be
  separated from another renderer's implementation is rejected, however good it is.

The same rule applies to game data: no disc image, BIOS, save, capture, texture pack extracted from
a game, or game executable. Fixtures use synthetic GS streams written in the test source.

## Building and testing

Requirements: CMake 3.24+, a C++20 compiler and the Vulkan SDK (headers, loader and
`glslangValidator`).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

GPU fixtures exit 77 when no Vulkan 1.2 device exists; CTest reports them as skipped. A skipped GPU
test is not a pass: run the suite on a machine with a device before a pull request that touches the
GPU path.

Keep the shader compiler pin unless you deliberately update it: the SHA-256 is part of the
reproducible build. If your distribution's `glslangValidator` differs, set `COREGS_GLSLANG_PIN=OFF`
locally and say so in the pull request.

## What a good change looks like

- **A fixture with it.** New behaviour needs a CTest case; a rendering fix needs a before/after that
  the fixture asserts (picture, GS memory, or both).
- **One variable at a time.** Keep A/B switches for diagnostics; the default path has no switch that
  changes the picture silently.
- **Defined behaviour.** Fixed-width arithmetic, explicit wrapping, no host undefined behaviour
  standing in for guest semantics.
- **Documented.** Update `docs/FEATURES.md` (status), `docs/ARCHITECTURE.md` (design) or
  `docs/PROFILES.md` (a rule) in the same change.

## Adding a game profile

A profile is how a game-specific optimisation enters CoreGS without weakening the generic core. See
[docs/PROFILES.md](docs/PROFILES.md) for the rules, the GT4 example and a step-by-step recipe. In
short:

1. Reproduce the problem with the generic profile; record the scene, a capture and the measurement.
2. Add `profiles/<game>/` with a `MakeProfile()` and every rule off by default.
3. Turn on one rule at a time, with a fixture that proves equivalence (or the documented trade).
4. Explain in the header why the rule is correct for that game.

Do not use a profile to hide a core bug. If a game renders wrong with every rule off, the core is
wrong.

## Reporting a game that renders wrong

Open an issue with:

- the game (region and disc revision), the scene and how to reach it;
- what you see and what the hardware/console shows;
- the CoreGS options, the device and the driver;
- a capture if you can make one (the replay tool reads synthetic and recorded GS streams; attach a
  small one, not a game image);
- the renderer's interval line (`[core2] ...`) if the log is available.

Do not attach game data. A screenshot of the wrong frame and one of the right one are enough to
start.

## Code style

- C++20, no exceptions in the renderer's hot path, no RTTI dependency.
- `snake_case` for the public API, the existing internal style elsewhere; keep a file's local style.
- Comments explain *why* (the hardware rule or the measured reason), not *what* the next line does.
- Public headers stay platform-free and Vulkan-free. Vulkan includes belong in implementation files.
- Keep new dependencies out. The project uses the C++ standard library and Vulkan only; texture
  replacement already vendors stb_image and xxHash.

## Commits and pull requests

- One topic per pull request. Say what changed, why, and which fixture covers it.
- Run the full CTest suite; paste the summary in the pull request.
- Never weaken a test, a pin or a gate to make a change pass. If a check fails and you cannot find
  the cause, say so: a known failure is better than a hidden one.

## Licence

By contributing you agree that your contribution is licensed under the MIT licence of this
repository (see `LICENSE`). Do not contribute code you cannot license that way.
