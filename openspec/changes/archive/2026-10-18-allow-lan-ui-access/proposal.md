# Proposal

## Why

The UI is intentionally loopback-only today, which prevents access from another device on the local network. For explicitly requested demos and remote browser testing, users need an opt-in way to listen on all interfaces while retaining loopback-only behavior by default.

## What Changes

- Add an explicit listen-host option to `serve-ui`, preserving `127.0.0.1` as the default.
- Require an explicit remote-access opt-in before binding to a non-loopback/wildcard interface, and make the UI's Host validation consistent with that opt-in.
- Clearly warn that remote mode exposes local UI data and any enabled policy-editing endpoints to reachable clients; do not change firewall rules.

## Capabilities

### New Capabilities
- `local-ui-serving`: Control the UI server's network interface and Host validation, with safe loopback defaults and explicit remote-access opt-in.

### Modified Capabilities

## Impact

- Affects `serve-ui` CLI options and `src/onegpu_posttrain/ui/server.py`.
- Adds automated tests for default loopback behavior, remote opt-in, and Host validation.
- Does not open ports in the OS firewall or guarantee reachability from sandboxed environments.
