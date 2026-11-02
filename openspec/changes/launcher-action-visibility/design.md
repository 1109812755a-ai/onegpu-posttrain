# Design

## Context

See `proposal.md` for the motivation and `specs/web-workspace-ux/spec.md` for user-visible behavior. The Launcher currently uses `button primary` for consequential actions, but shared button rules provide only compact neutral styling; filled primary styling exists only in unrelated scoped areas. Training setup and evaluation preflight can also extend below the initial viewport, while their confirmation actions appear only after preflight.

## Goals / Non-Goals

**Goals:**

- Make the current consequential action visually dominant and discoverable throughout long training/evaluation forms.
- Keep preflight and launch as separate, explicitly named steps with the relevant summary adjacent to confirmation.
- Explain disabled actions and preserve one-GPU serialization and active-run recovery behavior.
- Keep keyboard, screen-reader, and narrow viewport interaction usable.

**Non-Goals:**

- Change API contracts, preflight policy, stage capability, training/evaluation behavior, or task supervision.
- Restyle every button across the entire workspace.
- Change Run stage identity labels or resolve unrelated evaluation presentation issues.

## Decisions

### Scope primary styling to Launcher actions

Add a strong filled style, larger hit target, and visible focus/disabled states for consequential Launcher actions, rather than globally changing every `.primary` control. This fixes the current class/style mismatch without creating unrelated visual regressions in policy or data workflows. Keep secondary navigation and less consequential links neutral.

### Use one sticky-in-flow action area per workflow step

Place the relevant training or evaluation action in a sticky action area within the Launcher content flow. It remains reachable during long forms, but does not cover content like a fixed viewport overlay can. The area shows a short step label and the current button; after successful preflight it switches to the confirmation action and presents the preflight summary next to it. Only one primary action is shown at a time.

### Derive disabled reasons from current UI state

When required stage inputs are missing, show the exact missing choices beside the action (for example, dataset or parent model). When the backend preflight reports blockers, keep those authoritative details in the preflight result. When a local GPU task is active, replace launch affordance with the active-task state and a “view run” route. Do not duplicate backend readiness rules in the browser or weaken server validation.

### Keep the action area accessible on narrow screens

At narrow widths, allow the action area to wrap into a compact vertical layout, retain a minimum touch target, respect safe-area insets, and ensure focused controls are not hidden behind sticky content. Keyboard focus and disabled reasons remain programmatically associated with their controls.

## Risks / Trade-offs

- [Sticky areas may consume viewport space on small screens] → Keep the area compact, in-flow sticky rather than fixed, and validate at 390 CSS pixels and keyboard focus states.
- [A visually prominent action may imply readiness before preflight] → Label the first action as configuration check/preflight and expose the launch confirmation only after successful preflight.
- [Frontend readiness summaries can drift from backend rules] → Use frontend checks only to explain missing local selections; treat server preflight as authoritative for blockers and launch eligibility.

## Migration Plan

1. Add Launcher-scoped primary action styling and reusable action-area layout.
2. Connect training and stage evaluation steps to clear labels, disabled explanations, and active-task navigation.
3. Add interaction/accessibility tests for missing inputs, successful preflight, active GPU task, narrow viewport, and keyboard use.
4. Build the web client and verify the training and evaluation screens in a browser. Rollback is limited to reverting the scoped styles and Launcher action markup; persisted runs and APIs are unchanged.
