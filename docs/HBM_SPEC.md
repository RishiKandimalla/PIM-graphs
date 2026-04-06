# HBM configuration reference (PIM-graphs)

This project uses **Ramulator 2** with the config file [`configs/hbm_pnm.yaml`](../configs/hbm_pnm.yaml).

## Device

- **Standard:** HBM2 device model (`DRAM.impl: HBM2`).
- **Organization preset:** `HBM2_8Gb` — density, 128-bit DQ, hierarchy `channel` → `pseudochannel` → `bankgroup` → `bank` → `row` → `column` (see Ramulator `src/dram/impl/HBM2.cpp`). The preset has **1 channel** and **2 hardware pseudo-channels** in the org vector; Ramulator enforces **density** if you override `pseudochannel` count (e.g. 16× would break the 8 Gb check unless row/bank counts are rescaled). We keep the **stock preset** and model **16 logical PCs** in software; addresses still decode through `RoBaRaCoCh` into that geometry.
- **Timing preset:** `HBM2_2Gbps` — includes `tCK_ps` (cycle time in picoseconds) and JEDEC-style constraints used by the controller.

## Bandwidth (order-of-magnitude)

- **Per-cycle transfer size:** Ramulator uses the device prefetch and channel width to derive the minimum transaction granularity (see `LinearMapperBase` in `linear_mappers.cpp`: `tx_bytes = m_internal_prefetch_size * m_channel_width / 8`).
- **TSV budget (Week-1 style):** `RamulatorHbmSimulator` applies an optional **bytes-per-memory-cycle** cap on **admitted** requests: approximately `tsv_peak_gbps × 1e9 × tCK` (decimal GB/s). Default is **512 GB/s**; set **`set_tsv_peak_gbps(0)`** to disable. This models aggregate **logic ↔ stack** bandwidth without replacing Ramulator’s internal DRAM timing.

## 16 pseudo-channels (logical ports)

We model **16 logical ports** in software (`pc_id` 0–15). Each port uses a **disjoint flat address region** (`kRegionSize` in `pseudo_channel.hpp`, **1 MiB** per PC). The Ramulator **address mapper** (`RoBaRaCoCh`) maps those addresses onto channel / pseudochannel / bank indices for the **HBM2_8Gb** organization.

## Parallel admission per memory cycle

Each **simulated memory clock**, before `memory_system->tick()`:

1. Repeat **waves** until a full round-robin sweep over all 16 PCs produces **no** new `send()` success (capped at 256 waves to bound pathological cases).
2. Within each sweep, PCs are visited in order **`round_robin_cursor`, `cursor+1`, …** so every non-empty queue gets **at most one issue attempt per sweep**.
3. Ramulator may **reject** `send()` when controller queues are full (**backpressure**); then fewer than 16 accepts happen in that wave.

So **up to 16 new reads** (and often multiple waves) can be **admitted** in one memory cycle, subject to controller acceptance and the **TSV byte cap** for that cycle.

## Ramulator `source_id` and pseudo-channels

Ramulator’s `GenericDRAMController` indexes per-core statistics with `req.source_id` when it is not `-1`, and the array size comes from `frontend->get_num_cores()` (the GEM5 frontend defaults to **1**). Our logical **PC id** (0–15) must **not** be passed as `source_id` unless the frontend reports at least 16 cores. The driver therefore sends **`source_id = -1`** so DRAM timing is unchanged but per-core stats are not split by pseudo-channel (we track PCs only in our multiplexer).
