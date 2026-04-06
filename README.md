# PIM-graphs — HBM + Ramulator 2 pseudo-channel driver

Cycle-accurate DRAM timing from [Ramulator 2](https://github.com/CMU-SAFARI/ramulator2) with a **16 logical pseudo-channel** front-end: **round-robin multiplexer**, **disjoint address regions**, **parallel admission** (multiple `send()` successes per memory cycle, wave-based RR sweeps), and an optional **TSV byte budget** (default **512 GB/s**, `set_tsv_peak_gbps(0)` to disable).

## Build

Requires **CMake 3.20+**, **C++20**, and **Clang** or **GCC** (Ramulator uses C++20).

On macOS with a recent CMake, if yaml-cpp warns about policy, configure with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j
cd build && ctest
```

Ramulator is built as `third_party/ramulator2/libramulator.dylib` (macOS) or `.so` (Linux). Tests set `BUILD_RPATH` so the `pim_tests` binary finds the library.

### AppleClang note

Ramulator’s `param.h` uses a pattern that AppleClang requires as `.template as<T>()`; a one-line patch is applied under `third_party/ramulator2/src/base/param.h`.

### CMake as a subdirectory

Ramulator’s upstream `CMakeLists.txt` uses `PROJECT_SOURCE_DIR` for includes and FetchContent paths so it can be nested under this project (replaces `CMAKE_SOURCE_DIR` in the top-level Ramulator CMake file).

## Layout

- [`configs/hbm_pnm.yaml`](configs/hbm_pnm.yaml) — HBM2 + GEM5 frontend + GenericDRAM (OpenRowPolicy; HBM has no `rank` level, so ClosedRowPolicy is not used).
- [`docs/HBM_SPEC.md`](docs/HBM_SPEC.md) — parameters, PC mapping, `source_id` note.
- [`src/`](src/) — `RamulatorHbmSimulator`, `PseudoChannelMultiplexer`, bandwidth oracle helpers.
- [`tests/test_pseudo_channels.cpp`](tests/test_pseudo_channels.cpp) — GoogleTest harness.

## Optional Python module

```bash
cmake -S . -B build -DBUILD_PYBIND=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j
```

Requires Python development headers. Produces a `pim_bindings` module (see [`python/pim_bindings.cpp`](python/pim_bindings.cpp)).

## Configuration path

Tests embed the absolute path to `configs/hbm_pnm.yaml` at compile time. Override at runtime with:

```bash
PIM_CONFIG=/path/to/config.yaml ./build/pim_tests
```
