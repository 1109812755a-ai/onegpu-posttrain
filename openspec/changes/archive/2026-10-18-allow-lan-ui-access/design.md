# Design

## Context

`serve-ui` currently has no host option: the CLI invokes `serve()` and Uvicorn binds to `127.0.0.1`; the FastAPI app separately restricts Host headers to loopback aliases. Remote access therefore needs both an explicit listener choice and matching Host validation. The app can expose local artifact/data endpoints, and optional policy editing is separately controlled by `--enable-policy-editor`.

## Goals / Non-Goals

**Goals:**
- Preserve current loopback-only behavior for existing invocations.
- Make remote exposure a deliberate CLI opt-in with clear startup warning.
- Keep the listener and Host policy aligned so remote requests work only in opt-in mode.

**Non-Goals:**
- Configure OS firewall rules, authentication, TLS, or router forwarding.
- Make the service reachable from an isolated Codex/browser environment; network reachability remains environment-dependent.
- Enable policy editing by default.

## Decisions

- Add a `--host` option defaulting to `127.0.0.1`, and a separate `--allow-remote` flag. Reject any non-loopback/wildcard host unless the opt-in is present; this avoids silently widening exposure when users set a bind address.
- Keep `TrustedHostMiddleware` restrictive in normal mode. In remote mode, allow the Host values needed for LAN clients, with the CLI warning that all reachable clients can access the UI's local read APIs and any policy-editing APIs enabled separately.
- Do not automatically enable `--enable-policy-editor`; remote mode and editing remain independent switches.
- Do not add firewall mutations. A successful bind only means the process listens locally; firewall and network routing determine external reachability.
- Test app Host validation through `TestClient` and CLI/serve argument forwarding with unit tests, avoiding tests that depend on opening a real external socket.

## Risks / Trade-offs

- Remote mode exposes local report and artifact data to reachable clients → require explicit `--allow-remote`, print a prominent warning, and retain the loopback default.
- Allowing arbitrary Host headers in remote mode weakens DNS-rebinding protection → only permit this when the user explicitly opts into remote access; keep loopback allowlist otherwise.
- The UI may remain unreachable due to firewall policy or sandbox networking → print the bind address and explain that application binding does not change firewall rules.

## Migration Plan

No migration is required. Existing commands continue to bind to `127.0.0.1`; users who intentionally want LAN access pass a non-loopback `--host` and `--allow-remote`. Rollback is to omit those options and restart the process.
