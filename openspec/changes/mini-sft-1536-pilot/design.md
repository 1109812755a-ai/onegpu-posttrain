# Design

## Context

See `proposal.md`. The calibration candidate already has a stable dataset ID, source revision, sampling/eligibility details, and canonical/native SHA256 values. Training lineage currently searches only `manifests/*.yaml`; the candidate's JSON calibration manifest therefore falls back to an incomplete file reference. The SFT probe also identifies M1 by digest, while M1's model artifact definition lives with the CPT/merge evidence.

## Goals / Non-Goals

**Goals:** Represent the calibration candidate through the project's existing `DatasetManifest` and `RunRecord` contracts; derive a shorter subset from its measured record lengths without rescanning upstream data; make training resolve exact consumed bytes to the child dataset ID; preserve the observed 1,536 OOM; then complete and evaluate an exploratory SFT run if the shorter configuration fits.

**Non-Goals:** Rewrite or regenerate the historical JSONL/calibration manifest, claim semantic quality approval, change official Mini readiness targets, generate missing rows, or redesign lineage schemas.

## Decisions

- Add an explicit, idempotent registration operation for the existing calibration candidate. It verifies the declared source revision and canonical/native file digests before emitting a canonical `DatasetManifest` under `manifests/` and a `data_build` registration Run under `runs/<run_id>/lineage.json`.
- The canonical manifest retains the existing dataset ID and actual sample count, records `dataset_only` record provenance, source/tokenizer revisions, length/answer-selection protocol, canonical digest, and native path/digest metadata. It does not invent a processing report or quality approval; those remain unavailable/not evaluated.
- Training lineage resolves canonical or native dataset bytes only when their digest matches a supported manifest. A mismatch is an error, not a fallback to a misleading complete dataset reference. The SFT Run continues to reference the actual M1 model artifact and writes its adapter; the existing merge path writes M2 as a distinct artifact with its own merge Run.
- Treat the recorded 1,536 full-run OOM as the reason to derive the 1,024 subset from the registered parent cohort. Keep only canonical records whose measured `rendered_tokens <= 1024`, regenerate their native representation, and write a child dataset manifest plus a `data_build` Run. The processing report reconciles parent count, retained count, and length-based rejections; the child preserves source record IDs and is not an official Mini replacement.
- Run strict validation from the project artifact root so local data, M1, SFT adapter, merge, and evaluation references can resolve together. Expected external/unavailable evidence remains explicit; local dangling references, duplicate IDs, or digest mismatches block progression.
- The three-step probe did not predict full-run feasibility: 1,536 OOMed at step 20/599. Preserve that failure and use a distinct 1,024 recipe/run; do not rewrite or resume the failed output as if it were a completed model.

## Risks / Trade-offs

- Registration may encounter stale or mismatched source files → fail closed on revision/path/hash mismatch and leave historical inputs untouched.
- The 1,024 subset contains only 307 rows → it is sufficient to exercise the pipeline, not to claim robust capability or meet an official Mini target.
- The shorter run may still OOM → retain failure lineage and do not claim an M2 unless a checkpoint and merge actually exist.
- Exact answer agreement is not reasoning verification and review is pending → label the dataset and all dependent outputs exploratory; make no official-readiness or causal-quality claim.

## Migration Plan

No migration of existing records. Additive registration records make the existing calibration cohort consumable by lineage-aware training. New execution outputs use unique directories and do not overwrite the 2,048 pilot or the existing probe.
