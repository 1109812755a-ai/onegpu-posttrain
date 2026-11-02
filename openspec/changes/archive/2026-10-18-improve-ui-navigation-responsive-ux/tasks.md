# Tasks

## 1. Navigation and accessibility

- [x] 1.1 Route the Data navigation item to the Data workspace from all enabled pages, and add a regression test verifying the destination and active navigation state.
- [x] 1.2 Add accessible names to all sidebar controls and preserve visible keyboard focus; verify accessible names and focus behavior in browser checks at desktop and collapsed-sidebar widths.

## 2. Responsive workspace layout

- [x] 2.1 Fix intrinsic-width and breakpoint issues in the app shell, top bar, Targeted Data page, and Policy Comparison page; verify at 390px and desktop widths that document width does not exceed viewport width and controls remain visible.
- [x] 2.2 Add or document repeatable viewport regression coverage for all three workspace pages and verify no page-level horizontal overflow at 390px.

## 3. Report availability feedback

- [x] 3.1 Distinguish a missing overview report from other read errors while keeping capabilities loading independent; add tests for report 404, other failure, retry, and successful loading states.

## 4. Integration verification

- [x] 4.1 Run the frontend test suite and production build, then verify navigation, report states, keyboard focus, and responsive layout in the running UI at 390px and desktop widths.
