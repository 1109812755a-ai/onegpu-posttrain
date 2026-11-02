# Web Workspace Ux Specification

## Purpose

The local web workspace provides reliable navigation and clear, accessible interactions across desktop and narrow-screen layouts, including understandable report availability states.

## Requirements

### Requirement: Data navigation returns to the data workspace
The workspace SHALL let users return to the Data page from any workspace subpage using the Data navigation item.

#### Scenario: Return from targeted data
- **WHEN** a user selects Data while viewing Targeted Data
- **THEN** the Data workspace is shown and its navigation item is marked current

#### Scenario: Return from policy comparison
- **WHEN** a user selects Data while viewing Policy Comparison
- **THEN** the Data workspace is shown and its navigation item is marked current

### Requirement: Workspace content fits narrow viewports
At supported mobile viewport widths of 390 CSS pixels and above, workspace pages SHALL avoid page-level horizontal overflow while keeping controls and content usable.

#### Scenario: View targeted data on a mobile viewport
- **WHEN** the Targeted Data page is displayed at 390 CSS pixels wide
- **THEN** the document width does not exceed the viewport width and the form remains usable

#### Scenario: View policy comparison on a mobile viewport
- **WHEN** the Policy Comparison page is displayed at 390 CSS pixels wide
- **THEN** the document width does not exceed the viewport width and both policy forms remain usable

### Requirement: Icon-only navigation remains accessible
Every workspace navigation control SHALL expose a programmatic name and a visible keyboard-focus indicator, including when its text label is visually hidden.

#### Scenario: Use collapsed navigation with a keyboard
- **WHEN** a keyboard user tabs to an icon-only navigation control
- **THEN** assistive technology can identify the control and a visible focus indicator is shown

### Requirement: Report absence and request failure are distinct
The Targeted Data page SHALL distinguish a report that has not been generated from a report request that failed, and SHALL offer a relevant recovery action for request failures.

#### Scenario: No report exists yet
- **WHEN** the local report endpoint indicates that no report is available
- **THEN** the page explains that no report has been generated and points to the report-generation command without presenting it as a service failure

#### Scenario: Report request fails
- **WHEN** the report request fails for a reason other than an absent report
- **THEN** the page identifies a read error and provides a way to retry loading the report
