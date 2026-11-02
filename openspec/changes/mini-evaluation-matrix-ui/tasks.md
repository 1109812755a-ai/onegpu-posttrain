## 1. Read-only evaluation matrix API

- [x] 1.1 Discover Mini evaluation-matrix artifacts using server-owned paths only; enforce workspace containment, symlink rejection, and file/count limits. Verify with API tests for valid listing, empty workspace, traversal attempts, symlinks, and oversized inputs.
- [x] 1.2 Add matrix list/detail responses that preserve model and metric statuses, denominators, protocol/comparability, limitations, and stable lineage IDs. Verify with fixtures for complete, partial, failed, incomparable, and unresolved-lineage matrices; missing values must never become zero.

## 2. Stage Comparison workspace

- [x] 2.1 Activate the Stage Comparison route/navigation and preserve URL state across refresh, back, and return-to-Data actions. Verify with frontend navigation tests.
- [x] 2.2 Render the selected matrix with M2 baseline, per-model raw metrics/units, requested and observed sample counts, deltas, comparability, protocol identity, and limitations. Verify complete and partial fixture rendering, including explicit missing/failed states and no inferred better/worse styling.
- [x] 2.3 Resolve model/evaluation/run lineage by stable IDs and expose available source details without hiding valid metrics when lineage is unresolved. Verify resolved and unresolved lineage cases.

## 3. Usability and integration verification

- [x] 3.1 Implement loading, empty, API-error/retry, comparable, incomparable, and partial-result states; support keyboard operation and visible focus. Verify error/retry behavior and keyboard focus with frontend tests.
- [x] 3.2 Make the comparison usable at 390 CSS px without page-level horizontal overflow and document the read-only matrix view in the Chinese product/UI documentation. Verify the narrow viewport layout and documentation links.
- [x] 3.3 Run focused backend/frontend regression tests and rebuild the bundled UI; smoke-test the no-matrix and real-Mini matrix cases, and confirm existing Data/Runs/policy/targeted pages are unaffected and no training/evaluation action is triggered.
