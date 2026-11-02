# Design

## Context

See proposal.md - Why. The stage-profile API currently describes configurable parameters as bounded numeric overrides, and SFT command construction does not emit quantization arguments. The lower-level CPT command has an optional BNB 4-bit path, but the recorded environment for the affected run did not have bitsandbytes detected. This change introduces one supported SFT mode choice without exposing arbitrary recipe fields.

## Goals / Non-Goals

**Goals:**
- Give SFT a server-owned mode catalog with standard LoRA as the unchanged default and a pinned 4-bit QLoRA preset as opt-in.
- Bind the choice to preflight, confirmation, run metadata, output verification, and user-visible status.
- Fail closed when runtime compatibility or QLoRA export/merge support is missing.

**Non-Goals:**
- Automatically selecting QLoRA, installing dependencies, retrying OOM runs, or changing sequence length, data, parent, or training budget.
- Exposing arbitrary quantization method/bit-width/compute-dtype values in the browser.
- Enabling quantization modes for CPT, DPO, SimPO, or GRPO before their complete command and artifact workflows are independently validated.

## Decisions

1. **Represent memory mode as a typed profile option, separate from numeric parameter overrides.** The API exposes stable mode IDs and presentation metadata, while the server resolves each ID to its approved settings. This avoids broadening the current numeric allowlist or accepting arbitrary CLI/config input. The SFT UI sends the mode ID together with its normal parameter overrides; existing requests that omit it resolve to standard LoRA.

2. **Pin the initial QLoRA preset to BNB 4-bit NF4 with double quantization and FP16 compute.** A fixed preset is easier to validate on the project's Windows/CUDA environment and makes runs comparable. Exposing 4/8-bit, quantization type, and compute dtype independently would multiply unsupported combinations; those can be considered later from compatibility evidence.

3. **Preflight is capability validation, not a promise of fit.** Check package presence/version, CUDA availability, selected profile support, and required CLI capabilities without installing or downloading anything. Show sequence length and a clear OOM caveat. If an explicit bounded GPU probe is offered, it is a separately confirmed diagnostic action; its status and sampled peak VRAM are evidence for that exact profile/length, not a guarantee for the full run.

4. **Do not silently change data or training semantics to fit memory.** The selected mode does not rewrite `max_length`, filter records, alter epochs/steps, or change the parent/dataset. Users may explicitly adjust existing bounded parameters, and the changed effective config must trigger a fresh preflight.

5. **Keep the existing default and extension boundary.** Standard LoRA remains the default. Other stages stay unavailable for this mode until their own CLI support, preflight, merge/export, model inference, and lineage paths pass tests; shared parameter names alone are not sufficient evidence of compatibility.

## Risks / Trade-offs

- [QLoRA dependencies may be absent or incompatible on Windows] -> Detect the exact local capability and block with actionable detail; do not install or fallback automatically.
- [4-bit loading reduces weight memory but long-sequence activations can still OOM] -> Preserve the configured length, state the limitation, and use a bounded probe/real run evidence before claiming feasibility.
- [Quantized adapter export may differ from the existing LoRA merge path] -> Add an end-to-end test that trains a bounded adapter, exports/merges against the exact parent, verifies artifact identity, and loads it for inference before enabling downstream handoff.
- [Mode selection is a semantic training change] -> Include its resolved settings in the config digest and run lineage, and invalidate stale preflight confirmation whenever it changes.

## Migration Plan

No migration is required. Existing profiles and historical runs without a memory-mode field resolve/display as standard LoRA only for compatibility; their stored configurations and digests remain unchanged. Rollback removes the SFT QLoRA option while preserving existing standard-LoRA runs and artifacts.

