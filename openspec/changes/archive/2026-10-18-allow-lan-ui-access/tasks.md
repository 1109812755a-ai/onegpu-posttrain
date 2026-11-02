# Tasks

## 1. Listener configuration and safety checks

- [x] 1.1 Add `--host` (default `127.0.0.1`) and explicit `--allow-remote` CLI options; reject non-loopback binds without opt-in before starting Uvicorn, and verify with CLI unit tests.
- [x] 1.2 Pass listener and remote-access settings through `cmd_serve_ui` and `serve`; verify via mocked serve forwarding tests and a startup warning assertion.
- [x] 1.3 Make app Host validation conditional on remote-access opt-in while preserving the loopback allowlist by default; verify loopback and non-loopback Host scenarios in API tests.

## 2. User-facing guidance and regression checks

- [x] 2.1 Update `serve-ui` help and local UI usage documentation with the LAN command, exposure warning, and statement that firewall rules are not changed; verify documented option names match CLI help.
- [x] 2.2 Run the focused UI/CLI tests and OpenSpec strict validation; verify default loopback behavior and opt-in remote behavior both pass.
