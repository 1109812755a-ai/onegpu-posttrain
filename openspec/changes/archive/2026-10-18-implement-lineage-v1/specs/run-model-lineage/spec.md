# 规格增量

## Purpose

本能力以唯一 Run 串联数据处理、质量评估、训练、合并和评测的实际输入输出，使单卡阶段比较能追溯来源，并区分同一实验标签下的多次执行。

## ADDED Requirements

### Requirement: 统一运行身份与类型
每次实际的数据构建、质量评估、训练、合并、评测操作 SHALL 取得唯一不可复用的 `run_id`，并记录类型、状态、输入与实际输出；`experiment_id` 是可选的受控实验分组，`run_label` 是可选、可重复的展示标签，二者 SHALL NOT 充当血缘主键。新训练的 `run.json` 与 `lineage.json` SHALL 使用相同 `run_id`，并记录阶段、模型与数据、配方路径、解析后配置摘要及代码修订或明确的不可得状态。dry-run SHALL NOT 声称发生实际运行或生成模型。

#### Scenario: SFT 成功运行
- **WHEN** SFT 运行完成
- **THEN** 血缘能标明输入 CPT 模型、SFT 数据、配方／配置证据和实际输出产物

#### Scenario: dry-run 命令
- **WHEN** 训练命令以 dry-run 模式执行
- **THEN** 如有计划产物，其状态为计划中，不分配代表已执行操作的正式 `run_id`，且不宣称存在 checkpoint 或合并模型

#### Scenario: 同一标签重复执行
- **WHEN** 同一 recipe、`experiment_id` 与 `run_label` 被实际运行两次
- **THEN** 两次运行拥有不同 `run_id`，标签与实验分组可相同，且不会互相覆盖

#### Scenario: 数据构建与质量评估
- **WHEN** 数据构建和后续质量检查均执行
- **THEN** 它们拥有不同 `run_id` 与类型，分别连接输入／输出数据和质量报告

### Requirement: 无 manifest 数据的兼容训练
CLI SHALL 继续允许以普通 JSONL 文件作为 `--dataset` 训练输入。缺少可验证数据集 manifest 时，血缘 SHALL 记录实际文件路径与可获得的内容摘要，并标为不完整；SHALL NOT 虚构数据集 ID 或来源。Mini／OVAT 正式可复现验收 SHALL 要求完整、可验证的 manifest 链路。

#### Scenario: 自带 JSONL 无 manifest
- **WHEN** 用户用未登记 manifest 的本地 JSONL 启动训练
- **THEN** 训练仍可执行，血缘记录文件证据并标记为不完整

#### Scenario: Mini 验收检查
- **WHEN** 一次 Mini 运行只有 JSONL 文件证据而无数据集 manifest
- **THEN** 它不能被标为“血缘完整”或通过完整可复现验收

### Requirement: 模型产物身份与祖先
每个新模型产物 SHALL 具有稳定模型 ID、类型（`base`、`adapter` 或 `merged`）、位置及摘要或明确的不可得状态。adapter SHALL 指向产生它的训练 `run_id` 和父模型；合并模型 SHALL 指向独立合并 `run_id`、父模型与用于合并的 adapter。

#### Scenario: DPO 模型合并
- **WHEN** DPO adapter 与 SFT 父模型合并
- **THEN** 合并模型能够追到该确切 SFT 父模型、DPO adapter、训练 Run 和合并 Run

#### Scenario: 无法计算摘要的外部基础模型
- **WHEN** 外部基础模型无法在本地计算摘要
- **THEN** 产物记录可获得的仓库与修订证据，并将摘要标为不可得

### Requirement: 准确记录执行结果
失败或 OOM Run 的血缘 SHALL 保留已知输入与结果，但只列出实际生成且核实过的输出。缺失产物 SHALL NOT 被写成成功输出；如记录部分 checkpoint，SHALL 明确标为 `partial`。训练成功而合并失败时 SHALL 保留两个 Run 的各自终态。

#### Scenario: GRPO 在 checkpoint 前 OOM
- **WHEN** GRPO 在写出 checkpoint 前发生 OOM
- **THEN** 血缘记录 OOM，且不包含 adapter 或合并模型

#### Scenario: 训练成功但合并失败
- **WHEN** 训练生成 adapter，但合并失败
- **THEN** 训练 Run 记录 adapter 成功，独立合并 Run 记录失败，不宣称存在合并模型

### Requirement: 可比较的并行分支
血缘 SHALL 保留 DPO、SimPO、GRPO 各分支实际父模型的 ID 与摘要。声称共用 SFT 父模型的比较 SHALL 核对父模型身份和摘要；不一致时 SHALL 标出差异，不得呈现为受控的同源分支。

#### Scenario: 相同 SFT 父模型
- **WHEN** DPO 与 SimPO 执行引用相同的 SFT 模型 ID 和摘要
- **THEN** 可以把它们归为并行子分支

#### Scenario: 父模型内容不同
- **WHEN** 两个分支使用相同展示名称但父模型摘要不同
- **THEN** 比较结果报告父模型不匹配

### Requirement: 本地图完整性与兼容性
本地读取器／校验器 SHALL 检测重复的唯一产物 ID／`run_id`、缺失的本地引用、祖先环与摘要不符，并区分不可得的外部证据和损坏的本地引用。没有 `lineage.json` 的旧运行 SHALL 继续可读，但标为旧版，不能将旧 `run_id` 标签静默重解释为唯一执行身份；不支持的未来主版本 SHALL 明确拒绝。

#### Scenario: 缺少本地父数据集
- **WHEN** 运行引用本地数据集 ID，却找不到对应 manifest
- **THEN** 校验报告悬空引用

#### Scenario: 旧版 Smoke 运行
- **WHEN** 现有运行有 `run.json` 但无 `lineage.json`
- **THEN** 读取器仍展示运行，并标记血缘为不完整／旧版

#### Scenario: 模型祖先成环
- **WHEN** 模型祖先关系形成环
- **THEN** 校验失败并列出涉及的模型 ID
