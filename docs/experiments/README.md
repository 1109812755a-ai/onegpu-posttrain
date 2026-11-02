# Experiments

This directory is reserved for **verified experiment reports**.

Do not place hypothetical or placeholder result tables here.

A report should contain:

```text
Question
Hypothesis
Parent checkpoint
Dataset manifest
Control variables
Independent variable
Training budget
Runs / seeds
Metrics
Results
Resource cost
Observation
Interpretation
Limitations
Conclusion
```

Initial v0.1 experiment families:

1. CPT dataset size;
2. SFT learning rate;
3. DPO beta;
4. DPO vs SimPO-original;
5. GRPO generation count;
6. GRPO reward design.

See [`../design/v0.1.1/04-experiment-design.md`](../design/v0.1.1/04-experiment-design.md).
