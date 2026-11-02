"""ms-swift external reward plugin for OneGPU-PostTrain GRPO.

Usage:
    --external_plugins plugins/grpo_rewards.py \
    --reward_funcs onegpu_accuracy onegpu_boxed_format

The data adapter retains ``solution`` at the same level as ``messages``.  That
column is therefore forwarded to reward kwargs by ms-swift.
"""
from __future__ import annotations

from typing import Any

from swift.rewards import ORM, orms

from onegpu_posttrain.rewards.math import accuracy_reward, boxed_format_reward


def _as_aligned_list(value: Any, n: int) -> list[Any]:
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) == n:
            return values
        if len(values) == 1:
            return values * n
        raise ValueError(f"Reward field length mismatch: expected {n}, got {len(values)}")
    return [value] * n


class OneGPUAccuracyReward(ORM):
    def __call__(self, completions, **kwargs):
        solutions = kwargs.get("solution")
        if solutions is None:
            raise ValueError("OneGPUAccuracyReward requires dataset column 'solution'")
        golds = _as_aligned_list(solutions, len(completions))
        return [accuracy_reward(c, g) for c, g in zip(completions, golds)]


class OneGPUBoxedFormatReward(ORM):
    def __call__(self, completions, **kwargs):
        del kwargs
        return [boxed_format_reward(c) for c in completions]


orms["onegpu_accuracy"] = OneGPUAccuracyReward
orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward
