# Spec Delta

## ADDED Requirements

### Requirement: Module subpages are nested under their owning navigation item
The workspace sidebar SHALL display a multi-page module's subpage links directly beneath that module and keep the module expanded while one of its pages is current.

#### Scenario: Open a data subpage
- **WHEN** the user navigates to Data Quality or Policy Comparison
- **THEN** the Data module remains expanded and its subpage links appear immediately beneath Data
- **AND** the current subpage is visibly identified as selected

#### Scenario: Select a module parent
- **WHEN** the user selects Data or Analysis & Improvements
- **THEN** the workspace navigates to that module's default page and displays its subpage links beneath the selected module

#### Scenario: Visit a module with one destination
- **WHEN** the current module has only one destination
- **THEN** the sidebar does not render a redundant subpage list for that module

### Requirement: Nested navigation preserves narrow-screen usability
The workspace SHALL preserve a usable subpage navigation pattern on narrow screens without introducing page-level horizontal overflow.

#### Scenario: Navigate between subpages on a narrow screen
- **WHEN** a module with multiple pages is open at a supported narrow viewport
- **THEN** its subpage links remain reachable through the compact horizontal sub-navigation and the current page is identified
- **AND** the document width does not exceed the viewport width

### Requirement: Nested navigation exposes expanded and selected state accessibly
The workspace SHALL expose the current module and subpage state to assistive technology and keyboard users.

#### Scenario: Use nested navigation with a keyboard
- **WHEN** a keyboard user navigates through an expanded module's links
- **THEN** every link has a visible focus indicator, the selected subpage exposes its current-page state, and the owning module's expanded state is programmatically available
