# GRPO Reward Contract

v0.1 optimizes two explicit outcome components:

```text
R = 1.0 * accuracy + 0.1 * boxed_format
```

## Accuracy

`onegpu_accuracy` receives the model completion plus the GRPO dataset's top-level `solution` field. It uses the same final-answer extraction and optional `math_verify` equivalence path as project evaluation.

The reward is binary:

```text
correct   -> 1.0
incorrect -> 0.0
```

## Boxed format

`onegpu_boxed_format` checks for a complete final `\boxed{...}` expression and returns 1/0. It intentionally does **not** use ms-swift's built-in `format` reward because that built-in targets `<think>...</think><answer>...</answer>`, while the frozen OneGPU-PostTrain math protocol uses boxed final answers.

The parser is balanced-brace aware, so nested LaTeX is valid:

```text
\boxed{\frac{1}{2}}
```

## Plugin

The ms-swift external plugin is:

```text
plugins/grpo_rewards.py
```

and is registered through:

```text
--external_plugins plugins/grpo_rewards.py
--reward_funcs onegpu_accuracy onegpu_boxed_format
--reward_weights 1.0 0.1
```

Each component is retained separately in GRPO training metrics. A rise in total reward is never interpreted as a task-capability improvement without stage-end evaluation.
