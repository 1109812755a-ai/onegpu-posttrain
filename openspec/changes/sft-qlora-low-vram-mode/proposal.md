# Proposal

## Why

The SFT workbench currently exposes bounded numeric recipe overrides but no supported way to select a memory-saving training mode. A real 8 GiB RTX 4060 run OOMed during backward at a rendered sequence length of about 3,082 tokens despite micro-batch 1 and gradient checkpointing, so users need an explicit, reproducible low-VRAM option rather than silently changing sequence length or dropping data.

## What Changes

- Add an explicit SFT mode selector for standard LoRA (default) and opt-in 4-bit QLoRA (NF4), with the effective mode shown during preflight, confirmation, and run inspection.
- Connect the selected quantization settings through the SFT command adapter and validate local dependency/runtime support before launch.
- Preserve the selected mode and its parameters in the resolved configuration and run lineage; never silently fall back between modes.
- Add a bounded feasibility probe/resource warning for the selected mode and sequence length; communicate that quantization reduces weight memory but does not guarantee that long-sequence activations fit.
- Keep other stages out of the initial implementation unless their command, merge/export, and inference compatibility are separately validated.

## Capabilities

### New Capabilities
- `training-memory-modes`: Explicit, validated, and traceable memory-saving modes for local stage training.

### Modified Capabilities
- None.

## Impact

- Affects the stage-profile API and parameter resolution, SFT training command construction, training preflight/resource checks, run configuration/lineage, and the web training-parameter UI.
- Requires compatibility checks for bitsandbytes, CUDA, PyTorch, ms-swift, and QLoRA adapter merge/export behavior on the supported local environment.
- Does not change the default LoRA recipe or automatically truncate, filter, quantize, retry, or otherwise relax an existing run configuration.
