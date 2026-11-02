# Spec Delta

## ADDED Requirements

### Requirement: Users can select a global appearance mode
The workspace SHALL provide Light, Dark, and System appearance modes through a control available across all workspace pages. On a browser with no saved preference, the appearance SHALL remain Light.

#### Scenario: Select an appearance mode
- **WHEN** a user selects Light, Dark, or System from the workspace appearance control
- **THEN** the selected mode is applied consistently across the workspace without changing data, training, or evaluation state

#### Scenario: First visit has no saved preference
- **WHEN** the workspace is opened in a browser with no saved appearance choice
- **THEN** the workspace uses Light mode

### Requirement: Appearance preference persists and follows the system when requested
The workspace SHALL remember an explicit appearance choice in the current browser and SHALL resolve System mode from the operating-system color preference, including changes while the workspace is open.

#### Scenario: Restore a saved explicit appearance
- **WHEN** the user reloads or revisits the workspace after choosing Light or Dark
- **THEN** the saved mode is restored before the page becomes interactively usable

#### Scenario: Follow operating-system preference
- **WHEN** System mode is selected and the operating-system color preference changes
- **THEN** the workspace updates its effective appearance while retaining System as the selected mode

### Requirement: All workspace surfaces remain legible in each appearance mode
The workspace SHALL apply its selected appearance to shared navigation, page surfaces, form controls, tables, logs, dialogs, and semantic status indicators while preserving readable contrast and visible keyboard focus.

#### Scenario: Read status and operate controls in Dark mode
- **WHEN** Dark mode is active on any supported workspace page
- **THEN** text, interactive controls, logs, and status states remain distinguishable and keyboard focus remains visible
