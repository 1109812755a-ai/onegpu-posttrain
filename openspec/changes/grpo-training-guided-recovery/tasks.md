# Tasks

## 1. GRPO 阻断恢复入口

- [x] 1.1 在 GRPO Launcher 数据选择区保留不合格数据的具体后端阻断原因；当没有可选 GRPO 数据时，即使列表中存在禁用数据项，也显示“准备 GRPO 数据”入口。用 React 测试覆盖无数据、全阻断、部分可用三种状态，并确认阻断项不能选中。
- [x] 1.2 为恢复入口增加训练上下文深链，带上 `stage=grpo`、当前 `profile_id`、兼容的 `parent_model_id` 与返回目标；数据页注册完成后将精确 `dataset_id` 带回。用导航往返测试验证兼容值恢复、无效模型不恢复、旧预检/token 清除且不自动启动。

## 2. 数据派生与操作说明

- [x] 2.1 验证 GRPO 数据工作台从有效 OpenR1 SFT seed 完成预检、派生、登记和返回训练页；新增 API/UI 回归测试覆盖缺少可信 seed、契约不匹配、许可未核验提醒和重复刷新恢复，确保不合规数据保持 fail-closed。
- [x] 2.2 更新中文阶段能力与数据工作台指南，准确说明 GRPO Web 受控启动、trusted seed 派生入口、reward/reference 阻断含义及返回后需重新预检并显式启动；按指南人工走查一次并确认步骤和按钮可见。

## 3. GRPO 本地闭环验收

- [x] 3.1 用已登记 OpenR1 train_seed 派生一份带当前 reward contract 的新 GRPO dataset；保存预检结果，确认所有 reference/contract 行校验通过且来源许可未知仍被明确提示。
- [x] 3.2 使用兼容的已登记 SFT merged model 和批准 low-memory GRPO profile，在本地 GPU 运行有界 smoke 训练并执行 GRPO 阶段评测；核对 Run 状态、真实 reward/rollout 指标、模型产物及 dataset/model/training/evaluation lineage，修复并验证标准 GRPO profile 的 TensorBoard 日志目录初始化，遇到 OOM 或失败如实记录而不放宽校验。
