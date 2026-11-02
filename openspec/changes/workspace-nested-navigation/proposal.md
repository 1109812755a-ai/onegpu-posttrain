# Proposal

## Why

The sidebar currently renders a module's secondary pages in a separate block after all primary navigation items. This weakens the parent-child relationship and makes it harder to understand where pages such as Datasets, Data Quality, and Policy Comparison belong.

## What Changes

- Render secondary pages directly beneath their owning module in the desktop sidebar when that module is active.
- Keep the active module expanded while the user is on any page within it, and visibly distinguish the selected child page.
- Apply the same hierarchy consistently to Data and Analysis & Improvements; modules with a single destination do not need redundant child navigation.
- Preserve the existing narrow-screen horizontal sub-navigation while keeping its active-page and accessibility behavior consistent.

## Capabilities

### New Capabilities

### Modified Capabilities
- `web-workspace-ux`: Define hierarchical module navigation, active/expanded states, and consistent desktop/mobile behavior.

## Impact

- Affects the workspace sidebar rendering and navigation styles in `web/src/main.tsx` and `web/src/style.css`.
- Affects navigation interaction/accessibility tests and the `web-workspace-ux` behavior contract.
- No API, data, or training-run behavior changes.
