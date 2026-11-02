# Changelog

## Unreleased / v0.1 development

Current implementation version: `0.1.0.dev6`.

The code-level smoke pipeline is implemented for:

```text
Base
→ PEFT-CPT / Full-CPT feasibility probe
→ SFT
→ DPO / SimPO-original / GRPO-standard / GRPO-lowmem
→ unified evaluation and comparison
```

Real RTX 4060 smoke execution completed on 2026-10-15. The generated run artifacts are local and are not included in the source release; v0.1 Mini/OVAT release gates remain open. Smoke feasibility must not be presented as an algorithm-quality claim.

Detailed implementation history: [`docs/project/CHANGELOG-implementation.md`](docs/project/CHANGELOG-implementation.md).

Design revision history: [`docs/design/v0.1.1/CHANGELOG-v0.1.1.md`](docs/design/v0.1.1/CHANGELOG-v0.1.1.md).
