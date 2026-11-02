# Proposal

## Why

The workspace currently uses one light appearance with no user control, which can be uncomfortable in different lighting conditions and for users who prefer dark interfaces. A simple, consistent preference across the whole local UI improves readability without changing data, training, or evaluation behavior.

## What Changes

- Add global Light, Dark, and System appearance modes, available from the shared workspace chrome on every page.
- Preserve the current Light appearance as the first-visit default; persist an explicit selection in the current browser and follow operating-system preference when System is selected.
- Apply the selected mode consistently to workspace surfaces, controls, navigation, status states, logs, tables, and dialogs, while preserving semantic status distinctions and accessible contrast/focus indicators.
- Keep this preference presentation-only; it does not modify backend settings, training configuration, dataset artifacts, or run lineage.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `web-workspace-ux`: define global appearance modes, persistence, system preference behavior, and accessible theme switching.

## Impact

- Frontend shared shell and styles in `web/src/main.tsx` and `web/src/style.css`, plus focused frontend tests.
- No backend API, dependency, artifact schema, or training behavior changes are expected.
