# Spec Delta

## ADDED Requirements

### Requirement: Weight-space experiment has a dedicated workspace entry
The workspace SHALL expose the weight-space experiment as a distinct, directly discoverable module with an accessible name, visible keyboard focus, and a usable layout at supported narrow viewport widths.

#### Scenario: Navigate to the experiment workbench
- **WHEN** a user selects the weight-space experiment module
- **THEN** its experiment page is displayed, the navigation item is marked current, and keyboard and assistive-technology users can identify and focus the control

#### Scenario: View the workbench on a narrow viewport
- **WHEN** the workbench is displayed at a supported 390 CSS pixel viewport width
- **THEN** page-level horizontal overflow is avoided and experiment controls remain usable

### Requirement: Experiment state is visible in the workbench
The workbench SHALL distinguish unconfigured, blocked, ready, running, completed, partial, discarded, and failed experiment states and provide the relevant next action when a state is recoverable.

#### Scenario: Distinguish target and editor models
- **WHEN** a user reviews experiment setup or a run
- **THEN** the workbench clearly identifies Qwen3.5-0.8B as the target checkpoint and Bailian Qwen3.8-Max as the edit-planning model, and shows the target model revision and pinned Docker runtime identity

#### Scenario: Final evaluation is blocked
- **WHEN** a candidate lacks a completed practice iteration or its checkpoint provenance is invalid
- **THEN** final evaluation is disabled and the workbench identifies the blocking condition

#### Scenario: Bailian iteration fails
- **WHEN** Qwen3.8-Max is unavailable, times out, or returns an invalid tool call
- **THEN** the workbench displays a sanitized failure, preserves the parent candidate, and offers retry or correction without displaying a fabricated score

### Requirement: Provider cost is visible before iteration submission
The workbench SHALL show the selected provider model, maximum planned calls, and available cost estimate before a user confirms a billable iteration.

#### Scenario: Pricing is unavailable
- **WHEN** no current provider price is configured
- **THEN** the workbench explicitly marks the estimate as unknown and requires the user to confirm before sending any provider request

#### Scenario: Pricing is configured
- **WHEN** the provider profile includes current input and output prices
- **THEN** the workbench displays an estimate derived from the request budget and labels it as an estimate before submission
