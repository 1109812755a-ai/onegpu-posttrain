# Tasks

## 1. 验证 OpenR1 偏好目标扫描与短缺证据

- [x] 1.1 补充或修正数据构建测试，验证构建会跨过早期候选继续扫描到目标、在显式 `max_raw_rows` 停止、在输入耗尽时报告短缺，并保持 train/eval problem IDs 不相交及现有质量门槛不变；运行 `pytest tests/test_data_build.py`。
- [x] 1.2 用真实 OpenR1 流程复核 preference train/eval 目标扫描，保存构建报告并检查 raw rows、selected/target counts、rejection counters、stop reason；目标未达时确认 CLI 非零退出，记录实际数据源/上限/失败原因。真实 pinned source 前 5,000 行报告：`data/processed/dev7/openr1-real-check/reports/openr1_data_report.json`；显式上限停止，preference train 0/200、eval 1/50，CLI 非零退出。该证据不代表全量源耗尽。
- [x] 1.3 根据测试和真实报告，仅在发现合同缺陷时修复 builder/CLI；更新数据证据说明、dev7 milestone 与实现状态，并以 `pytest tests/test_data_build.py tests/test_data_evidence.py` 和对应示例构建报告验证。未发现合同缺陷，未改变过滤语义；相关测试 22 passed，真实处理报告已保存并在 `docs/implementation/data/DATA_EVIDENCE.md` 说明范围限制。

## 2. 实施 GRPO completion 长度校准

- [x] 2.1 为 512/768/1024/1536 候选准备可复用的 recipe/运行清单与结果汇总，保证每次只改变 completion 长度、使用固定模型/prompt IDs/seed/预算，并以共享 `experiment_id`、独立 `run_id` 记录；用单元测试验证控制变量和报告字段。工具：`prepare-grpo-calibration` / `summarize-grpo-calibration`；设计说明：`docs/implementation/training/GRPO_COMPLETION_CALIBRATION.md`；控制测试通过。
- [x] 2.2 在 RTX 4060 8GB 上顺序执行四个候选（如某候选 OOM/失败也保留记录），汇总实际生成长度、clipping/truncation、reward/verifier、耗时、峰值显存及 execution status；验证报告包含四个候选且失败指标标为 unavailable。四个正式候选均 PASS，run IDs 与完整实测见 `runs/dev7/grpo-completion-calibration-attempt3/grpo_calibration_report.json`；两次训练前 cache-path 失败 attempt 另行保留。
- [x] 2.3 基于保存的校准报告选择受支持配置，或明确记录没有候选达到低于 30% clipping 的门槛；仅在证据支持时更新默认 recipe，并更新 GRPO 校准说明/dev7 milestone，运行相关指标、recipe 和报告测试。没有候选达标；1536 clipping 最低但 accuracy reward 均值为 0，因此没有改动默认 recipe。报告及限制已写入 GRPO 校准说明/dev7 milestone。

## 3. 完成统一 M2/M3 任务评测矩阵

- [x] 3.1 增加或复用矩阵汇总能力，覆盖 M2-SFT、M3-DPO、M3-SimPO、M3-GRPO-standard、M3-GRPO-lowmem，并显式区分 completed/partial/failed/missing；为协议一致性、缺失指标和 lineage 引用添加测试。`compare-evaluation-matrix` 与 [`UNIFIED_EVALUATION_MATRIX.md`](../../docs/implementation/evaluation/UNIFIED_EVALUATION_MATRIX.md) 已实现；矩阵、stage comparison 和 evaluation lineage 相关测试 9 passed。
- [x] 3.2 使用冻结的 GSM8K、MATH-500、internal-holdout 数据 ID、prompt/decoder/parser/limits，实际运行 M2 与四个 M3 模型的统一评测；保存各模型 `eval.json`/evaluation lineage 和矩阵比较产物，确认协议哈希与覆盖一致。五个模型全部 completed，均为 GSM8K 5、MATH-500 25、internal 20、PPL 20；protocol/config hash/dataset IDs 一致。
- [x] 3.3 更新 smoke 报告、dev7 milestone 与实现状态，标明可用、缺失及失败模型/指标并限定结论；运行评测、lineage、stage-comparison 相关测试。矩阵报告标明小样本 smoke 限制；相关测试 9 passed。

## 4. 清理 Ruff 并核对项目状态

- [x] 4.1 在当前开发环境记录 Ruff 版本并运行 `ruff check src tests`，修复发现项且不添加宽泛忽略；复跑同一命令验证零 findings。Ruff 0.17.0，检查零 findings。
- [x] 4.2 核对 `implement-lineage-v1` 已完成任务与当前数据集/run/model/evaluation lineage 产物，运行 `validate-lineage` 及相应测试；只将证据支持的项目状态和设计状态标为完成，对不完整历史 Smoke lineage 保留明确限制。dev7 matrix root 为 5 runs/5 artifacts/0 errors；40 条为扫描根目录之外引用的 warnings。`runs` 总根的 10 个 dangling refs 属于 GRPO calibration 缺少保留的 parent/checkpoint 路径，已在状态文档记录。
- [x] 4.3 运行完整测试套件与 `python -m compileall -q src plugins`，记录命令、工具版本及通过/环境阻塞状态，更新 `IMPLEMENTATION_STATUS.md` 和 dev7 milestone；最终确认 OpenSpec 四个验收组均有对应产物且未改写历史 Smoke 结果。全测 112 passed/1 skipped，compileall 与 Ruff 通过；UI API 全测在沙箱外通过，历史 Smoke 报告仅追加补充章节。
