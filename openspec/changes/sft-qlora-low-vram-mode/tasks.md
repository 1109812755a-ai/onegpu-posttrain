# Tasks

## 1. SFT Memory-Mode Contract

- [x] 1.1 Add a server-owned SFT memory-mode catalog with standard LoRA as the backward-compatible default and one fixed BNB 4-bit NF4 QLoRA preset; verify profile/API tests reject unknown mode IDs and browser-supplied quantization knobs.
- [x] 1.2 Bind the resolved mode and approved settings into preflight summaries and confirmation identity; verify changing the mode invalidates the old confirmation and existing requests without a mode still resolve to LoRA.
- [x] 1.3 Add local dependency/runtime capability checks for the selected QLoRA mode without installing packages or downloading resources; verify missing/incompatible bitsandbytes or CLI support yields an actionable blocker and no training process.

## 2. SFT Execution and Artifact Lifecycle

- [x] 2.1 Extend SFT command construction to emit only the pinned QLoRA flags when that mode is explicitly selected; verify LoRA commands remain unchanged and QLoRA command tests assert all approved settings.
- [x] 2.2 Persist the selected mode, effective quantization settings, and config digest in Run/resolved-config/lineage records; verify Run inspection shows the actual mode for success, OOM, and failure states.
- [x] 2.3 Verify QLoRA adapter export/merge against the exact parent and load the merged artifact for inference; add tests proving export failure leaves the Run ineligible for downstream handoff. Merge now explicitly pins the parent and disables inherited checkpoint args so a QLoRA checkpoint cannot silently re-quantize the merged handoff model.

## 3. Workbench User Experience

- [x] 3.1 Add a clear SFT selector for standard LoRA versus opt-in QLoRA 4-bit; verify the default remains LoRA and changing selection clears stale preflight/confirmation state.
- [x] 3.2 Show the effective mode, dependency blocker, configured sequence length, and non-guaranteed OOM/resource warning in preflight and run details; verify UI tests cover both missing-dependency and supported-mode states.
- [x] 3.3 Document the supported QLoRA preset, local dependency requirements, explicit no-fallback behavior, and sequence-length limitation; verify documentation matches the profile/API output.

## 4. Bounded Feasibility Verification

- [x] 4.1 On an environment that passes compatibility checks, run a separately bounded one-step SFT probe with the selected QLoRA preset and the target maximum sequence length; record environment, exact config, completion/OOM status, and sampled peak VRAM, and verify the probe neither changes the dataset nor claims full-run success. Evidence: `runs/diagnostics/qlora-4bit-nf4-maxlen4048-step1-20261101/probe-report.json`; the registered dataset SHA-256 is unchanged and the report explicitly records that the peak-memory reading exceeds nominal dedicated VRAM and does not establish full-run safety.
- [x] 4.2 Run focused profile, preflight, command, confirmation, artifact, and web tests plus the relevant full backend/frontend suites; verify no regression to standard LoRA defaults and report any local hardware/dependency limitation explicitly.
