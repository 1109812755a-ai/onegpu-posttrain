# Tasks

## 1. Local credential loading

- [x] 1.1 Implement strict `.env.local` credential parsing with process-environment precedence and sanitized errors; verify parser unit tests cover valid, duplicate, malformed, disallowed, and precedence cases.
- [x] 1.2 Load credentials only from explicit targeted generation execution; verify dry-run leaves environment untouched and makes no provider call.

## 2. Documentation and verification

- [x] 2.1 Add a safe local env template and document setup, supported syntax, repository-root invocation, and exposed-key rotation; verify `.env.local` matches the ignore rule and documented command syntax is accurate.
- [x] 2.2 Run targeted synthetic-generation tests and OpenSpec strict validation; verify no secret value appears in logs or generated artifacts.
