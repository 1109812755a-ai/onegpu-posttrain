# Security Policy

## Supported versions

The project is currently pre-v1.0. Security fixes are applied to the latest development branch and the latest tagged release once releases begin.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could expose credentials, execute untrusted code unexpectedly, or compromise a training host.

Until a dedicated private security contact is configured for the repository, use the hosting platform's private security advisory feature when available.

## Operational safety notes

Training workflows may execute external Python packages, ms-swift plugins, model code and dataset-processing logic. Before running third-party recipes or plugins:

- review the code and dependency changes;
- avoid committing tokens or credentials;
- use isolated environments;
- treat remote model/dataset code as untrusted unless reviewed;
- do not expose training services directly to untrusted networks by default.
