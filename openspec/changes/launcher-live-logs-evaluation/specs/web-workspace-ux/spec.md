# Spec Delta

## MODIFIED Requirements

### Requirement: Workspace content fits narrow viewports
At supported mobile viewport widths of 390 CSS pixels and above, workspace pages SHALL avoid page-level horizontal overflow while keeping controls and content usable.

#### Scenario: View targeted data on a mobile viewport
- **WHEN** the Targeted Data page is displayed at 390 CSS pixels wide
- **THEN** the document width does not exceed the viewport width and the form remains usable

#### Scenario: View policy comparison on a mobile viewport
- **WHEN** the Policy Comparison page is displayed at 390 CSS pixels wide
- **THEN** the document width does not exceed the viewport width and both policy forms remain usable

#### Scenario: View Launcher run status and controls on a mobile viewport
- **WHEN** the Launcher displays preflight, an active training/evaluation run, or completed results at 390 CSS pixels wide
- **THEN** the document has no page-level horizontal overflow, primary controls remain reachable, and logs or metric tables can be inspected without obscuring run status
