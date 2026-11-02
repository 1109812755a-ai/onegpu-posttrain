# Spec Delta

## ADDED Requirements

### Requirement: Runs is an active workspace destination
The primary navigation SHALL open the Runs list, identify Runs as the current page, and preserve selected list/detail navigation state across refresh and browser history actions.

#### Scenario: Navigate to Runs
- **WHEN** a user selects Runs from another workspace page
- **THEN** the Runs list is shown and the Runs navigation item is marked current

#### Scenario: Refresh a run detail URL
- **WHEN** a user refreshes while viewing a specific run detail
- **THEN** the same run detail is restored or an explicit not-found state is shown

### Requirement: Runs list and detail remain usable and accessible
Runs list and detail SHALL support keyboard navigation, visible focus, accessible control names, and responsive layouts without page-level horizontal overflow at 390 CSS pixels and above.

#### Scenario: Browse Runs on a narrow viewport
- **WHEN** the Runs list or detail is displayed at 390 CSS pixels wide
- **THEN** the page has no horizontal overflow and run identity, status and primary actions remain usable

#### Scenario: Operate Runs with a keyboard
- **WHEN** a user navigates the Runs list and detail using a keyboard
- **THEN** filters, rows/links and return navigation are operable with visible focus and meaningful accessible names
