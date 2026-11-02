# Spec Delta

## Purpose

Allows users to explicitly select a supported lower-memory training mode and understand its environment requirements, resource limits, and effect on the resulting model artifacts.

## ADDED Requirements

### Requirement: SFT memory mode is an explicit bounded choice
The SFT workbench MUST offer the approved standard LoRA mode and an opt-in approved 4-bit QLoRA mode, defaulting to standard LoRA; it MUST NOT accept arbitrary quantization settings from the browser.

#### Scenario: Keep the default training mode
- **WHEN** the user leaves the memory mode unchanged
- **THEN** preflight and launch use the profile's standard LoRA mode and display that effective mode

#### Scenario: Opt into 4-bit QLoRA
- **WHEN** the user selects QLoRA and requests preflight
- **THEN** the system resolves the approved quantization configuration and includes it in the effective training configuration

### Requirement: QLoRA compatibility is checked before launch
The system MUST verify required local quantization dependencies and supported runtime compatibility before allowing a QLoRA SFT run; an unavailable dependency MUST be reported as a blocker and MUST NOT trigger installation or fallback.

#### Scenario: QLoRA dependencies are unavailable
- **WHEN** the selected QLoRA mode cannot be supported by the installed training environment
- **THEN** preflight identifies the missing or incompatible dependency and launch remains blocked without creating a training process

#### Scenario: QLoRA environment is supported
- **WHEN** the required dependency and runtime checks pass
- **THEN** preflight permits explicit confirmation and shows that the resource check is an estimate rather than a guarantee against OOM

### Requirement: Selected memory mode is immutable across preflight and launch
The system MUST bind the selected SFT memory mode and its effective settings to preflight confirmation and the resulting Run; changing the mode MUST invalidate an earlier confirmation.

#### Scenario: Launch with the confirmed mode
- **WHEN** the user confirms a QLoRA preflight without changing its inputs
- **THEN** the Run uses the same mode and quantization settings and records them in its resolved configuration and lineage

#### Scenario: Change mode after preflight
- **WHEN** the user changes between LoRA and QLoRA after preflight
- **THEN** the prior confirmation is rejected and a new preflight is required

### Requirement: Memory optimization does not silently alter training data
Selecting QLoRA MUST NOT implicitly change sequence length, truncate or filter records, change the parent model or dataset, or alter the training budget.

#### Scenario: Long sequence still exceeds available memory
- **WHEN** preflight or a bounded resource probe indicates that the selected mode may not fit the configured sequence length
- **THEN** the system reports the constraint and requires the user to make an explicit configuration change; it does not silently shorten sequences or omit examples

### Requirement: QLoRA output remains verifiable and usable
A completed QLoRA SFT run MUST verify and record its adapter and merged-model artifacts using the exact parent identity and selected quantization mode before presenting the result as a usable downstream model.

#### Scenario: QLoRA model export succeeds
- **WHEN** training completes and the adapter is exported or merged successfully
- **THEN** the resulting model is registered with its source Run, parent, dataset, and memory-mode configuration

#### Scenario: QLoRA artifact export fails
- **WHEN** the trained adapter cannot be exported or merged into a supported model artifact
- **THEN** the Run is reported as incomplete for downstream handoff and no usable merged model is presented
