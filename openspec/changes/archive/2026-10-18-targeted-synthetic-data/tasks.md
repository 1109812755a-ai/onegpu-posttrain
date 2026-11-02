# Tasks

## 1. 依赖、场景诊断与定向计划

- [x] 1.1 核对 data-policy-experiments 的策略、source/split、探索用途、实验报告和 editor 接口已实现且有关测试通过；依赖未完成时记录具体缺口并停止依赖实现，不复制另一套策略/血缘逻辑。
- [x] 1.2 新增场景定义、诊断报告、生成/复测计划的版本化类型与身份，绑定数据/模型/评测协议及阈值来源；用配置变化、未知字段、空场景及缺分母测试验证，不能从旧总分生成虚构场景结果。
- [x] 1.3 实现场景聚合 CLI，将 internal predictions 与固定 metadata/sidecar 按 case/problem ID 关联，报告解析失败、不支持、unknown 标签和多场景重叠；用手工可核对的逐题 fixture 验证分母、低样本状态及源文件不变。
- [x] 1.4 实现 train_seed/diagnostic_dev/final_holdout 角色、祖先隔离和训练种子选择，并对接污染索引；用评测原题/改写祖先、缺索引、角色历史及完全独立训练题测试验证，生成输入不含评测正文/gold。
- [x] 1.5 提供中文场景诊断与计划指南、项目自有训练/诊断/最终留出小样本和示例阈值声明；用无网络命令验证可登记计划，示例标准不被声称为普适质量或准确率门槛。

## 2. 可选生成接口、预算与恢复

- [x] 2.1 新增可选 synthetic extra、provider profile、输出协议和可配置模板，提供 fake 与一个 OpenAI-compatible adapter；用 fake/mock transport 测试验证支持参数、非支持参数拒绝、缺凭据不发请求、核心导入不依赖生成 extra。
- [x] 2.2 实现计划校验、dry-run 和显式生成 CLI，凭据只在实际调用时读取；用请求计数与秘密哨兵测试验证保存/页面/dry-run 不联网，拒绝密钥 URL/模板环境插值，日志/异常/配置/UI 不出现凭据。
- [x] 2.3 实现串行请求账本、max_requests、输出/重试/超时和可用 token/费用统计；用限流、持续错误、无 usage、费率估算及预算耗尽 tests 验证所有尝试都计数、未知费用不补零、不能无限填满配额。
- [x] 2.4 实现完整 request key、hash 校验缓存、原子完成状态、同计划执行锁与恢复；用中断/篡改/并发执行拒绝/配置变化/完成缓存复用测试验证同计划预算不重置，uncertain 只有幂等能力或显式选择才重试。
- [x] 2.5 提供中文 provider/提示词配置及恢复指南，说明计费边界、密钥变量、模型 revision 不可用与缓存复用；使用 fake adapter 验证文档命令和停止/恢复示例，无需真实账户或费用。

## 3. 候选验证、质量对照与血缘

- [x] 3.1 实现候选 schema、输入身份/slot、完成原因和响应摘要，并对格式、final_answer 与 solution 一致性、真实渲染长度分别检查；用格式正确但答案错误、截断、超长与字段冲突 fixtures 验证不自动标成 verified 或截断充数。
- [x] 3.2 包装已有答案抽取/等价验证形成 correct/incorrect/unparseable/unsupported/unverified 状态和方法/覆盖证据；用缺可信 reference、缺可选依赖、解析异常、不支持类型及可核验答案测试验证，不把自评当独立验证。
- [x] 3.3 通过先行策略服务导出 SFT 与同题 Preference 数据，记录缺侧、全正确/全错误/未知、长度比、去重和污染；用对应 fixture 验证 pairs 不伪造、不重复充数且阶段/评测分组不因生成改变。
- [x] 3.4 新增支持范围明确的 SFT 最终答案噪声变换，保存 parent、版本、seed、目标/实际比例和答案不等价证据；用相同问题匹配、稳定变换、不支持答案跳过及 metadata 真实状态测试验证，不引入默认 Preference 错误排序。
- [x] 3.5 生成标准 data_build/quality_assess lineage、候选/训练数据 manifests 和成本/验证报告，标记 llm_generate transform 与 provider revision 不可用；用生成→选择→数据→模拟训练/评测 refs 的 fixture 验证摘要和完整血缘，private 响应不进入 UI/公开 artifact 白名单。
- [x] 3.6 完成中文候选/验证/噪声与来源说明，提供程序化有独立 reference 的新题示例；离线验证候选导出、质量报告、lineage 及前后数据差异，说明答案验证不证明完整推理正确。

## 4. 复测报告与前端计划体验

- [x] 4.1 实现 before_after_pipeline 复测汇总，关联起点/新模型、相同评测协议和实际追加数据/训练预算；用固定逐题结果验证目标指标、保护指标、成本及阈值状态，未执行/无收益/退化都保留且不自动发起下一轮。
- [x] 4.2 对 matched_budget_ab 复用现有数据效果比较，登记普通补数与定向补数的共同起点/预算及实际控制条件；用 parent/预算/协议/题目集合不匹配反例验证 NOT_COMPARABLE，不把追加训练前模型伪装成相同训练预算控制组。
- [x] 4.3 前端加入场景缺口、定向计划、生成验证产出和复测区块，复用策略 editor 保存/导出，支持 provider profile 和提示词/格式配置；用 API/交互测试验证无收费请求启动入口、无密钥输入持久化、未执行显示待生成/待训练/待复测。
- [x] 4.4 完成场景分母、未知费用、partial/失败、缺输入、过期草稿与真实/fake 证据区分；记录键盘/窄屏 walkthrough，验证用户能从薄弱场景到配置导出再追踪模型和数据来源，失败不丢编辑。
- [x] 4.5 更新中英文质量/处理效果设计、前端/开发/开源路线及中文复测指南，明确提前交付的合成切片、诊断/最终留出角色和无收益结果；检查文档命令、链接与实际支持阶段一致，不宣称已运行真实 API/GPU 或提升能力。

## 5. 集成与可复现验收

- [x] 5.1 使用项目自有样本和 fake/mock transport 完成“诊断→计划→生成→验证→策略导出→模拟训练评测引用→复测报告”的 CPU 集成测试，验证污染拒绝、请求预算/恢复、血缘和保护指标退化路径；所有模拟结果明确标记，不充当真实训练证据。
- [x] 5.2 运行场景/生成/策略/血缘/UI 有关测试、Ruff、compileall、前端构建与 dry-run 指南；确认默认安装、只读 UI 和旧开源数据路径仍可用，无下载、真实调用、付费或 GPU 副作用，记录实际验证结果。
- [x] 5.3 执行 `openspec validate targeted-synthetic-data --strict`，逐项核对规格场景、实现和证据状态；真实模型/费用/能力效果留待用户配置并明确执行，不能因功能测试通过标成已获得收益。
- [x] 5.4 定向补数页的报告与配置接口独立加载；任一接口失败时保留另一接口已加载内容并显示对应错误。用 overview 404/capabilities 200、overview 200/capabilities 404 及双成功测试验证。

### 验收证据（2026-10-18）

- `pytest -q --ignore=tests/test_ui_api.py --basetemp=.pytest-tmp`：254 passed；4 条 Pydantic `model_` protected-namespace 警告。`test_ui_api.py` 的 TestClient socket 集成测试未纳入本沙箱全量命令；定向 UI API 路由/allowlist/秘密哨兵测试由 `test_ui_targeted.py` 与 `test_ui_policy.py` 覆盖。
- `.venv\\Scripts\\ruff.exe check --fix .`：修正 22 项可自动修复问题；随后 `ruff check .` 全部通过。
- `python -m compileall -q src scripts tests`：通过。
- `web/` 中 `pnpm build` 与 `pnpm test:policy-editor`：构建通过，2 项前端测试通过。
- `scripts/targeted_data_offline_example.py` 与 `scripts/targeted_synthetic_pipeline_example.py`：通过；输出明确标记 fixture/fake，`network_used=false`，无真实 provider 调用或模型训练。
- `PYTHONPATH=src python -m onegpu_posttrain.cli --help`：通过，旧数据与训练命令仍注册可见。
- `openspec validate targeted-synthetic-data --strict`：通过。
- `web/` 中 `node --test tests/targetedLoad.test.mjs tests/policyPreview.test.mjs`：5 passed，覆盖 overview 404/capabilities 200、反向 404 及双成功；`pnpm build`：通过。
- 未运行真实 API、GPU 训练或效果复测；不据此声称模型能力提升。真实浏览器/屏幕阅读器手工验收仍需在用户本机完成。

## Workflow follow-up

- 先审阅/归档先行 data-policy-experiments 并同步其主规格，再在依赖已实现的基础上推进本变更；若先行尚未归档，按其已审阅 delta 核对契约，不自动改写或重复生成主规格。
- 本变更任务和证据审阅完成后单独执行 OpenSpec archive。真实 API/GPU 补数实验通过显式计划后续运行，完整自动 flywheel/launcher/调度器另建变更。
