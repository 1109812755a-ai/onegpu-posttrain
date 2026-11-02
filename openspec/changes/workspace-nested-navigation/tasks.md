# Tasks

## 1. Nested desktop navigation

- [x] 1.1 Render each module's multi-page destinations directly beneath its owning primary item, derive expansion from the current page, and verify navigation still routes to the existing default pages.
- [x] 1.2 Add component-level navigation tests for Data and Analysis & Improvements nesting, selected child state, and no redundant subnav for single-destination modules; update the Vitest test include if needed and verify with `npx vitest run --config vitest.config.ts`.
- [x] 1.3 Style the nested groups with clear indentation and selected/focus states; verify the sidebar remains readable and scrollable with the longest module expanded.

## 2. Narrow-screen and accessibility behavior

- [x] 2.1 Preserve the compact horizontal sub-navigation at the existing narrow-screen breakpoint and keep it driven by the same active page/module state; verify visually at desktop and 390 CSS-pixel viewport widths that subpages remain reachable and there is no page-level horizontal overflow.
- [x] 2.2 Add component assertions for visible focus styling hooks, `aria-current`, and the expanded module state; verify with `npx vitest run --config vitest.config.ts`.

## 3. Regression verification

- [x] 3.1 Run `node --test tests/workspaceNavigation.test.mjs`, `npx vitest run --config vitest.config.ts`, and `npm run build` from `web`; verify route behavior, rendered navigation tests, and the TypeScript/Vite production build pass.
