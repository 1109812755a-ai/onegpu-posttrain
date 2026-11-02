# 本地血缘完整性只读审计

- 日期：2026-10-24
- 范围：仓库根目录下可发现的 v1 lineage、数据 manifest、模型登记和 evaluation lineage；pytest 临时目录排除。
- 命令：`python -m onegpu_posttrain validate-lineage --root .`
- 安全性：只读；没有改写、删除或自动修复任何 artifacts。
- 结果：**未通过**。扫描 123 runs、143 artifact 引用、47 datasets、34 models、23 evaluations；44 errors、100 warnings。

## 错误分类

| 错误 | 数量 | 含义 |
|---|---:|---|
| `dataset_input_digest_missing` | 24 | 消费数据集的 Run 未记录对应 canonical SHA-256 |
| `duplicate_artifact_id` | 7 | 同一 artifact ID 在多个 manifest/Run 输出中重复定义 |
| `dataset_producer_mismatch` | 5 | 数据登记声明的 producer 与实际 build Run 不一致 |
| `dangling_local_ref` | 4 | Run 引用的本地文件不存在 |
| `model_path_unavailable` | 2 | 模型登记路径不可解析或不在可审计范围 |
| `model_digest_mismatch` | 1 | 模型内容与登记 SHA-256 不一致 |
| `dangling_model_producer` | 1 | 模型引用的 producer Run 不存在 |

错误涉及的代表性对象包括：

- `artifacts/data-experiments/offline-example-20261017...`：重复 dataset/run ID、缺失 dataset input hash、producer 不一致。
- `tmp/sft1024-*-20261020/...`：派生数据路径已失效且重复登记；这是保留的历史 scratch 记录，没有擅自清理。
- `runs/dev7/grpo-completion-calibration-attempt3/...`：parent model ID 只有引用、没有本地登记。
- `runs/mini/cpt-finemath-5m-s42-v3/model_artifacts/...`：parent model ID 未在当前审计根登记。
- `runs/dev7/unified-m3-matrix*/.../evaluation_lineage.json`：多个历史 eval 引用的模型登记缺失。

## 警告分类

`unavailable_evidence` 42、`dataset_lineage_missing` 16、`unresolved_model_id` 12、`external_or_missing_path` 10、`incomplete` 8、`unresolved_model_parent` 6、`unresolved_parent_id` 6。警告主要来自早期/外部来源记录、未登记的历史模型，以及缺少完整 lineage 的 smoke/calibration artifacts。

## 结论与建议

当前实现能够定位真实历史链路断点；将外部 source ID 和 unavailable parent 归为提醒、并按 merge 的 parent+adapter 语义核验后，本地历史 artifacts 仍有 **44 个可定位错误**，尚未达到全仓库 lineage integrity 通过标准。优先针对仍会继续使用的正式实验补齐 dataset canonical hash 和 producer 闭环；重复/失效 scratch 记录先由用户决定保留、隔离或清理；外部模型登记应在获得可信父模型证据后补录。不要根据名称推断关系，也不要把本审计结果视作自动修复授权。

## 真实 SFT/评测后的增量复核（2026-10-24）

为核验本次真实探索性训练与阶段评测，在优化扫描器后重新执行 `validate_lineage_root('.', require_complete=True)`。结果为 **未通过**：134 runs、151 artifact references、47 datasets、40 models、25 evaluations；74 errors、74 warnings。增量对象是 SFT Run `run-mini-sft-1024-0fe815479943` 与 evaluation Run `run-launcher-mini-math-eval-41b6a382f28f`；定向身份/hash 闭合检查通过，这两个对象没有产生 lineage errors。评测 benchmark 的外部数据摘要不可用，按协议记录为 warning。

较初始审计新增的模型/评测 artifacts 和错误数属于新增审计覆盖与历史本地对象变化，不代表本次探索 Run 断链。没有自动修改历史错误，也没有把这些局部检查表述为全仓库通过。扫描器改为单次 `os.walk` 并在遍历时跳过 `.git`、`.venv`、`node_modules`，避免先完整遍历环境依赖再过滤。
