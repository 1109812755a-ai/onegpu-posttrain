# Design

## Context

See `proposal.md` for motivation. The React workspace is a single-page application with one shared shell in `web/src/main.tsx` and a large shared stylesheet in `web/src/style.css`. The stylesheet defines a small set of root color variables, but many page components also use hard-coded light colors. The preference is local to the browser and must not become training or artifact state.

## Goals / Non-Goals

**Goals:**

- Resolve Light, Dark, or System to one effective global appearance.
- Avoid a light-theme flash when restoring a saved preference.
- Keep theme selection keyboard and screen-reader accessible and all pages legible.

**Non-Goals:**

- Additional accent palettes, per-page themes, account/cloud synchronization, or backend settings.
- Reworking page layout, navigation taxonomy, or semantic status meanings.

## Decisions

### Store selected mode separately from effective mode

Represent the user's choice as `light`, `dark`, or `system`; resolve `system` through `matchMedia('(prefers-color-scheme: dark)')`. Keep the selected value in browser `localStorage` under a versioned key. Initialize synchronously from storage before the first render, falling back to `light` when absent or invalid, then subscribe to media-query changes only while `system` is selected.

This preserves the distinction between the user's preference and the currently resolved colors. A system change therefore updates the appearance without overwriting the user's `system` choice. No server persistence is needed for this single-user local workspace.

### Apply one theme at the shared shell and use semantic color tokens

Expose the effective theme on the shared application root as a `data-theme` attribute. Define semantic CSS custom properties for page background, surfaces, text, muted text, borders, controls, interactive accents, and status colors, with Light defaults and Dark overrides. Migrate existing hard-coded surface/text colors used by pages, tables, logs, and dialogs to these tokens; retain colors only where they encode intentional data visualization or brand assets, and ensure those remain legible in both modes.

Update the browser `theme-color` metadata alongside the effective theme so browser chrome does not contradict the page. Avoid CSS inversion filters: they distort images, status colors, and chart encodings.

### Place the control in shared workspace chrome

Add a compact labeled select or menu to the shared top bar so the choice is available from every page, including narrow layouts. Use explicit Chinese labels (“浅色”, “深色”, “跟随系统”), expose the current choice accessibly, and retain existing focus indicators. Do not place theme controls in individual page settings.

## Risks / Trade-offs

- [Hard-coded light colors may remain in less-used pages] → Audit all shared page components and add visual/assertion coverage for representative data, training, runs, dialogs, and logs in both effective themes.
- [System preference APIs differ in older browsers or test environments] → Guard `matchMedia` access and support modern `addEventListener` with a compatible fallback where necessary; default safely to Light.
- [Dark colors can collapse semantic status distinctions] → Use dedicated status tokens and verify contrast/focus behavior rather than blindly reusing the Light palette.

## Migration Plan

No data migration is required. Existing browsers without a saved preference remain in Light mode. Rollback removes the selector and theme resolution while leaving page data and run artifacts untouched; any saved preference key can be safely ignored.
