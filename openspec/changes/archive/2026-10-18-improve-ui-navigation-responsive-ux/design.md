# Design

## Context

See `proposal.md` for the observed usability issues and `specs/web-workspace-ux/spec.md` for user-visible requirements. The workspace is a React single-page app with a compact sidebar at mobile widths; the Targeted Data page currently treats all overview HTTP failures alike and has unit tests for independent report/config loading.

## Goals / Non-Goals

**Goals:**
- Keep the existing page model and make every enabled navigation item behave consistently.
- Make the three current workspace pages fit mobile widths without shrinking text or forcing page-level horizontal scrolling.
- Preserve keyboard operation and expose labels on collapsed icon controls.
- Keep missing local artifacts distinct from transient/API errors without coupling report and capabilities loading.

**Non-Goals:**
- Replacing the sidebar with a new navigation framework or adding a mobile drawer.
- Changing API routes, report schemas, read-only behavior, or data/training workflows.
- Adding a component library or end-to-end testing dependency.

## Decisions

1. **Use the existing page state for navigation.** Route the Data item to `setPage("data")` and derive active/current state from the same page value. This is simpler than adding URL routing and preserves current query-string behavior.
2. **Fix intrinsic-width sources at responsive breakpoints.** Keep the existing compact icon sidebar, but allow the main shell, top bar, page grids, forms, and long labels to shrink/wrap below 560px. Do not globally hide overflow, since that would conceal controls; retain local horizontal scrolling only for genuinely tabular/code content.
3. **Use native accessible names and visible focus.** Add `aria-label` to navigation buttons so their names remain available when visible labels are hidden. Retain native buttons and the current `:focus-visible` styling pattern.
4. **Represent report availability from HTTP status.** Treat overview 404 as “report not generated yet”; keep other failures as a retryable read error. Leave capabilities loading independent. This reflects the current local artifact model and avoids adding a new API response envelope.
5. **Test behavior at the narrow viewport and state boundaries.** Extend existing Node tests for report status classification and use browser-based checks for navigation and `documentElement.scrollWidth <= clientWidth` at 390px on each workspace page. Avoid a new browser-test dependency; use the existing Selenium setup for manual/regression verification where available.

## Risks / Trade-offs

- [Responsive fixes may hide a control by over-constraining a child] → Check all three pages at 390px and desktop width, and inspect actual viewport/document dimensions.
- [A 404 could also indicate a missing route rather than a missing report] → Apply the empty-report treatment only to the overview/report request; retain a clear endpoint/config error for capabilities or other requests.
- [Automated browser checks are not part of the current package scripts] → Keep deterministic logic in Node tests and record the manual viewport/navigation verification in the change's validation results.

## Migration Plan

No data or API migration is required. Rebuild the static frontend and restart the local UI server when convenient; existing servers serve their current build until restarted. Rollback consists of reverting the frontend change.
