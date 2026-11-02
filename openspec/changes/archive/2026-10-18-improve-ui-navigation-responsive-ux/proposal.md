# Proposal

## Why

The workspace currently traps users on the targeted-data page when they choose the Data navigation item, and its 390px layout overflows horizontally. The UI should remain understandable and operable across desktop and narrow screens, especially when reports are unavailable or local-only mode disables actions.

## What Changes

- Make the Data navigation item return to the Data workspace from every subpage.
- Rework narrow-screen layout to fit common mobile widths without page-level horizontal scrolling.
- Give icon-only navigation controls accessible names and preserve visible keyboard focus.
- Distinguish an absent report from a failed report request, with a clear recovery action where applicable.
- Add UI regression coverage for navigation, responsive sizing, accessible navigation labels, and report loading states.

## Capabilities

### New Capabilities
- `web-workspace-ux`: predictable navigation, responsive interaction, accessible controls, and clear report loading/empty/error states in the local web workspace.

### Modified Capabilities
- None.

## Impact

- Frontend shell and responsive styles in `web/src/main.tsx` and `web/src/style.css`.
- Targeted-data report loading and state presentation in `web/src/TargetedExperiments.tsx` as needed.
- Frontend regression tests under `web/tests/`.
- No API contract, training behavior, report format, or dependency changes are intended.
