# 验证记录（2026-10-29）

## 生成模型身份与 revision 血缘

- CPT、SFT、Preference 的登记产物及 GRPO 生成报告保存 Provider profile/adapter、声明模型、可得 revision 与身份状态。
- 统一使用 `profile_declared_not_independently_verified`；没有可验证 loaded-model attestation 时，登记模型关联状态为 `not_proven`，`registered_model_id/hash` 保持空值，模型名相同不构成身份匹配。
- 外部 Provider revision 缺失明确保存为 `unavailable`；不能据此声称精确复现。
- 验证：Provider identity/生成登记/阶段工作流相关测试 **34 passed**；Python compileall 通过。4 条 Pydantic protected namespace warning 为既有提示。
- 未发现当前 OpenAI-compatible Provider 路径提供可核验的实际 loaded-model 身份证明，因此没有猜测关联项目模型 ID/hash，也没有部署新模型服务。

## 只读生成状态快照与回归

- 新增 token 保护的 `GET /api/synthetic/status/{stage}/{execution_id}`，对 CPT/SFT/Preference 以稳定结构返回 partial/失败、进度、请求错误、输入身份、登记输出和缺失证据；查询只读取现有任务状态，不调用 Provider 或启动执行。
- 数据工作台中文指南已纠正 Preference 的当前能力描述，并说明 generation status、artifact catalog 与 Launcher 评测只读查询边界。
- 首次完整后端回归发现一个真实兼容缺陷：定向训练预检错误地要求 Preference pair 使用 `train_seed`。已修正为 Preference 必须为 `train_pairs` 且来源角色为 `train_seed`，其他阶段仍要求 `train_seed`。
- 修复后 `python -m pytest -q`：**764 passed, 9 skipped**，约 99 秒；6 条既有 Pydantic protected-namespace warning。
- 定向 status/identity/registration/preflight 回归：**18 passed**；`openspec validate multistage-generation-closed-loop --strict` 通过。

## 已执行

- 本轮 Preference 定向回归：`pytest -q tests/test_preference_generation_jobs.py tests/test_ordinary_preference.py tests/test_targeted_preference.py tests/test_preference_stage_adapter.py`：**21 passed**。覆盖普通/定向生成任务、read-only preflight、持久 execution 恢复式读取、人工审核、独立答案证据、exact train_seed 父集 ID/hash、DPO/SimPO 使用同一登记 dataset 与父模型但构造独立 Run 命令。
- 本轮前端：`npm run build` 通过；`npm run test:dataset-mixture`：**7 个测试文件、23 passed**（含 Preference 普通与定向工作台流程）；`npm run test:policy-editor`：4 passed；`npm run test:dataset-preview`：6 passed。
- 本轮 `python -m compileall -q` 对 Preference 任务/API/领域模块通过；`openspec validate multistage-generation-closed-loop --strict` 通过。OpenSpec 进度由 11/22 更新为 **12/22**。

- `python -m pytest -q`：**758 passed, 9 skipped**，约 115 秒；有 6 条既有 Pydantic protected-namespace warning。
- 前端 `npm run build`：TypeScript 检查与 Vite 生产构建通过。
- `npm run test:dataset-mixture`：6 个测试文件、21 项通过。
- `npm run test:policy-editor`：4 项通过。
- `npm run test:dataset-preview`：6 项通过。
- `openspec validate multistage-generation-closed-loop --strict`：通过。
- `python -m compileall -q`：本次新增/修改的 GRPO 与 Preference Python 模块通过。

## 未通过 / 未执行

- `python -m onegpu_posttrain validate-lineage --root . --require-complete`：**失败**。当前 workspace 汇总 213 Runs、57 datasets、69 models、38 evaluations，报告 **108 errors、100 warnings**。错误集中于历史 calibration/probe 缺数据集 manifest、旧 `tmp/` Run 指向已不存在文件、历史评测缺 benchmark revision/hash/摘要、旧评测/模型引用未登记等。此扫描只读，未改写历史证据；需单独逐类审计后修复，不能通过批量伪造 ID/hash 消除。
- 未调用真实 Provider、未发起付费请求、未启动 GPU 训练或评测；这部分按任务要求保留为用户逐项授权后的真实验收。

## 当前实现边界

- Preference 普通与定向路径已接到持久生成任务、受 token 保护的本地 API 和数据工作台 UI；预检为零 Provider 调用，未知请求按账本阻断重放。普通候选由用户按同题候选 ID 审核后登记，定向只使用登记计划的 train_seed 身份并由独立最终答案检查排序；生成/构建/质量/数据集 ID、hash、Provider 声明及身份未独立核验状态被保存。登记数据使用训练契约所需的 `train_pairs` 角色；输入种子仍保留 `train_seed` 父血缘。DPO/SimPO 仍是独立 Run，不会由数据登记自动启动。
- Preference provider 任务仍未接真实 Provider，本轮只用 fake Provider 纵向测试；不是付费/网络调用或真实训练验收。
- GRPO 普通/定向 prompt-only 候选路径已接入数据工作台与受控任务状态/API，并保留准确 plan/来源身份；候选仍不可直接训练，缺可信 reference 或 reward contract 时必须阻断。该路径尚无真实 Provider/GPU 验收证据。

## 最终集成复核（2026-10-29）

- 后端完整测试：`.venv\\Scripts\\python.exe -m pytest -q`：**771 passed, 8 skipped, 1 warning**，129.72 秒。唯一 warning 为 Starlette `TestClient` 与 `httpx` 的弃用提示；跳过项由测试标记跳过，不计作阶段真实训练覆盖。
- 前端完整 Vitest：`npm run test:dataset-mixture`：**9 个文件、28 passed**。补充 Node 测试：`npm run test:policy-editor` **4 passed**，`npm run test:dataset-preview` **6 passed**。
- 类型检查与生产构建：`npm run build`（含 `tsc --noEmit`）通过；Vite 提示 JS chunk 为 505.25 kB（gzip 144.50 kB），属于拆包/加载性能提示，不是构建失败。
- `openspec validate multistage-generation-closed-loop --strict`：通过。
- 全仓只读血缘审计：`.venv\\Scripts\\python.exe -m onegpu_posttrain validate-lineage --root .` **未通过**，退出码 1；扫描 213 runs、213 artifacts、57 datasets、69 models、38 evaluations，报告 **69 errors、139 warnings**。错误涉及历史 `tmp/` 引用失效、旧产物重复 ID、早期 smoke/示例 manifest 缺 lineage、旧评测/模型的父 ID 未登记等。未批量修补或伪造历史 ID/hash；这些全仓存量问题不等同于本变更新增闭环测试失败，但全项目血缘目前不能宣称干净。
- 未调用真实 Provider、未发起付费请求、未启动 GPU 训练或评测；逐阶段真实验收仍按清单待授权/待资源。当前验证证明的是离线测试与构建通过，不代表真实训练或真实 Provider 已验收。
