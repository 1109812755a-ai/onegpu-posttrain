# Design

## Context

See `proposal.md` for the user problem and `specs/web-workspace-ux/spec.md` for required behavior. The data workbench already exposes trusted GRPO seed discovery, preflight, deterministic derivation, registration and a “use this dataset” handoff. The stage Launcher disables ineligible GRPO datasets, but its recovery CTA only appears when the stage has no dataset entries at all; legacy and uploaded entries therefore hide the next step. The existing data-to-training handoff carries `stage` and `dataset_id`, but does not preserve the selected GRPO profile and parent model.

## Goals / Non-Goals

**Goals:**

- Make the existing trusted GRPO data preparation path discoverable from blocked training choices.
- Preserve safe, minimal GRPO training context across the Data/Launcher round trip.
- Align operation docs with the actual profile/adapter capability and validate one real local GRPO smoke chain after implementation.

**Non-Goals:**

- Allow legacy/uploaded GRPO rows to train by synthesizing or trusting missing evidence.
- Add arbitrary reward plugins, a new dataset builder, automatic training, or automatic license approval.
- Change the approved reward contract, GRPO objective, recipe defaults, or evaluation protocol.

## Decisions

1. **Keep preflight authoritative and expose its existing recovery route.** Render each ineligible item’s server-provided reason and show a prominent GRPO-preparation action whenever no eligible dataset is selected, even if blocked entries exist. The action opens the existing Data workbench flow; it does not convert blocker status or write metadata.
   - Alternative: silently filter blocked datasets and show an empty state. Rejected because it hides why existing data cannot run and gives less actionable recovery.

2. **Carry context in the local workspace URL, not a second persistence layer.** The Launcher-to-Data context will include `stage=grpo`, `return_to=launcher`, selected `profile_id`, and compatible `parent_model_id`. After registration, the explicit return action adds the exact new `dataset_id`; Launcher revalidates that model/profile still exist, preselects only compatible values, and clears preflight/confirmation state.
   - Alternative: persist the whole training form in server state. Rejected because existing URL handoff and local draft state are sufficient, and a second state owner risks stale identities.

3. **Keep registration and training as separate user-confirmed actions.** The seed builder continues to preflight and register through the existing API. Returning to training only restores choices; a fresh stage preflight and explicit start remain required.
   - Alternative: start training immediately after derivation. Rejected because registration may retain unknown license/quality status and the user must review the exact dataset and profile first.

4. **Use current blocker text and surface contract-specific guidance.** Do not recalculate trust in React or infer an error from display names. Backend dataset eligibility and stage preflight remain the source of truth; UI wording and links are covered by API/interaction tests.

## Risks / Trade-offs

- [URL context may refer to a model/profile removed or made incompatible while in Data] → Re-query current workbench options on return, keep invalid values unselected, explain why, and require a fresh preflight.
- [Unknown source license remains] → Preserve the existing explicit dataset-ID acknowledgement at launch; do not represent it as verified.
- [GRPO smoke consumes significant single-GPU time or fails from memory pressure] → Use the approved low-memory profile and bounded smoke recipe; record OOM/failure truthfully without weakening data or reward checks.
- [Docs are stale about Web support] → Update the capability table and run the documented path against current API/UI tests before claiming support.

## Migration Plan

No data migration is required. Existing GRPO datasets keep their current ineligible status until users derive/register compliant data. Rollback can remove the navigation affordance while leaving newly registered datasets and historical Runs readable.
