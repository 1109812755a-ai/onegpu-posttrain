# Spec Delta

## Purpose

Provides controlled network-interface selection for the local experiment UI while preserving a loopback-only default and requiring explicit opt-in for access from other devices.

## ADDED Requirements

### Requirement: UI server defaults to loopback
The UI server MUST listen only on the loopback interface by default unless the user explicitly selects another interface.

#### Scenario: Default startup
- **WHEN** the user starts `serve-ui` without a listen-host option
- **THEN** the server listens on `127.0.0.1` and accepts only loopback-compatible Host values

### Requirement: Remote UI access requires explicit opt-in
The UI server MUST require an explicit remote-access opt-in before accepting non-loopback Host values or listening on a non-loopback or wildcard interface, and MUST warn that reachable clients can access local UI data and enabled policy-editing endpoints.

#### Scenario: Wildcard bind without opt-in
- **WHEN** the user selects `0.0.0.0` without enabling remote access
- **THEN** startup is rejected with actionable guidance and no server socket is opened

#### Scenario: Wildcard bind with opt-in
- **WHEN** the user selects `0.0.0.0` and explicitly enables remote access
- **THEN** the server listens on all IPv4 interfaces, accepts the requested remote Host header, and prints a warning describing the exposure

#### Scenario: Remote Host without opt-in
- **WHEN** a request uses a non-loopback Host while remote access is disabled
- **THEN** the request is rejected by Host validation

### Requirement: Remote-access mode does not alter host firewall settings
Enabling remote UI access MUST NOT create or modify operating-system firewall rules.

#### Scenario: Remote mode startup
- **WHEN** the user enables remote access
- **THEN** the application only configures its own listener and Host validation and leaves firewall configuration unchanged
