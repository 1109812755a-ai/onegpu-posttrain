# Local Experiment Launcher

The optional Launcher provides a narrow local workflow for the validated Mini SFT recipe and its exploratory post-training evaluation. It is not a general job scheduler and does not accept arbitrary recipes, paths, or command arguments.

## Start the UI

From the repository root, install the project's `train` extra into `.venv`, then run:

```powershell
.\.venv\Scripts\python.exe -m onegpu_posttrain serve-ui --workspace . --enable-experiment-launcher
```

Open `http://127.0.0.1:8765/?page=launcher`. Reuse an already-running UI process; do not start a second server on the same port. The Launcher is loopback-only and requires same-origin requests.

## Training flow

1. Select **Check environment and configuration**. This read-only preflight validates the pinned SFT recipe, parent M1 model digest, data manifest, `swift` availability, and output path.
2. Review the resolved configuration and warnings, then confirm the one-time token before training starts.
3. The run panel polls the real trainer/merge logs and metrics. Scroll upward to pause auto-follow; choose **New logs · return to latest** to resume. Browser reload restores the latest training run from local state.
4. Training succeeds only when the CLI's run and lineage records confirm success and the merged M2 artifact is registered.

## Exploratory evaluation

After successful SFT and verification of the registered merged M2-SFT artifact, open **Evaluation** and run its read-only preflight. The only supported Launcher protocol is `math-eval-v1`, configured by:

- Recipe: `recipes/mini/30-eval-math-v1.yaml`
- Protocol: `configs/evaluation/math-v1.yaml`
- Benchmarks: GSM8K and MATH-500, at most 5 examples each, seed 42

Preflight does not create run outputs, download benchmarks, or start a model. After explicit confirmation, a separate evaluation `run_id` is created under the training run's `evaluations/` folder. Training and evaluation share one local GPU task lock. Browser reload preserves status and logs; a UI server shutdown interrupts the active task and records that state.

EvalScope may need to download benchmark data during the confirmed evaluation if it is not already cached. The UI warns about this before launch. The result is exploratory and must not be presented as a full GSM8K or MATH-500 benchmark score. Each displayed score includes the actual sample count reported by EvalScope. Missing benchmark reports remain missing and make the evaluation partial; no zero score is synthesized.

Evaluation lineage records the exact merged `model_id` and digest, its producing merge run when registered, the evaluation `run_id` and `eval_id`, protocol, benchmark dataset IDs, and available metrics. Inspect `eval.json`, `evaluation_lineage.json`, and `evaluation_run.json` from the Evaluation tab.

## Safety and scope

- Launcher mutations are local-only, token-protected, and require explicit confirmation.
- Training/evaluation are sequential on the same GPU; no queue, remote execution, user-provided shell command, or arbitrary model/data path is exposed.
- Preflight is read-only. Confirmed evaluation may use network/cache behavior managed by EvalScope.
- Mini SFT data and 5-sample evaluation are exploratory evidence, not release acceptance.

