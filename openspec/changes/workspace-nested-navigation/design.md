# Design

## Context

The sidebar has a primary module list and one secondary list rendered after all primary items. `workspaceModuleFromPage` already maps pages to modules, and `MODULE_VIEWS` defines each module's destinations. At narrow widths the sidebar secondary list is hidden and the current module's pages are shown in a horizontal sub-navigation above the content.

## Goals / Non-Goals

**Goals:**
- Make ownership of each subpage visually clear without changing URL or page-routing behavior.
- Keep desktop and narrow-screen navigation states consistent and accessible.
- Avoid rendering redundant child navigation for modules with one destination.

**Non-Goals:**
- Reorganizing the page taxonomy or changing page names, routes, or module defaults.
- Adding user-configurable sidebar pinning, arbitrary multi-level nesting, or a new navigation framework.

## Decisions

- Render each primary module and its subpage group together in the sidebar rather than rendering all primary modules followed by a single global secondary block. Use the existing module-to-page mapping as the source of truth.
- Derive the expanded module from the current page. Its multi-page sublinks are shown directly beneath its parent, so navigation and expansion cannot drift out of sync. Selecting a parent continues to navigate to its existing default page; it is not repurposed as a collapse-only control.
- Keep modules with one destination as a single primary link. This avoids a second, duplicate link for Training & Evaluation and Runs.
- Retain the existing compact horizontal sub-navigation for narrow viewports. The same current page and owning module state must be exposed there; desktop-only nested groups remain hidden at the existing responsive breakpoint.
- Use semantic navigation/link state: the selected child exposes `aria-current="page"`; the active module exposes its current state and expanded state where a disclosure relationship is represented. Preserve visible keyboard focus indicators.

## Risks / Trade-offs

- [The sidebar may become taller when an expanded module has several children] → Keep only the current multi-page module expanded and retain the existing scrollable sidebar behavior.
- [Desktop and mobile sub-navigation could diverge] → Drive both from the same `MODULE_VIEWS` and current-page state, and cover both layouts in tests.
- [Marking both a parent and child as current may be ambiguous to assistive technology] → Keep the parent module marked as the current/expanded section and expose `aria-current="page"` only on the actual selected destination; verify semantics with accessibility tests.

## Migration Plan

No data migration is required. Replace the desktop's global secondary navigation placement with per-module nested groups while preserving route targets and the responsive mobile sub-navigation. Rollback consists of restoring the existing rendering and styles; URLs and stored state are unaffected.
