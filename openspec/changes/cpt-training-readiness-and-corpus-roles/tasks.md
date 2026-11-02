# 任务

## 1. CPT 构建角色与证据

- [x] 1.1 更新 FineMath 及项目支持的其他 CPT 构建器，在训练/留出 canonical 记录写入 `train_seed`/`final_holdout`，并在 processing report/manifest 记录角色协议和计数；用构建器契约测试验证实际角色及行数完全一致。
- [x] 1.2 补充 CPT split/schema 测试，覆盖缺失、非法、混合角色和留出误入训练；确认现有上传注册不会自动补角色，`pytest tests/test_targeted_cpt_corpus.py tests/test_data_build.py` 通过。

## 2. 可信历史数据派生修复

- [x] 2.1 实现基于受支持构建协议、manifest/canonical 摘要及处理报告的只读 dry-run，输出可证明的 split→role 映射；用可信/伪造/证据缺失夹具验证仅可信输入可预览升级。
- [x] 2.2 实现显式确认后的新版本派生与登记，保留输入 ID/摘要、协议版本、角色计数和新 processing report/lineage；测试新 ID/摘要、血缘校验以及旧 JSONL/manifest/report 字节级不变。
- [x] 2.3 增加用户可调用的 CPT 历史数据修复入口和使用文档；测试不可信上传被拒绝且失败时不创建文件、不修改 registry。

## 3. CPT 预检与工作台诊断

- [x] 3.1 拆分 CPT 数据/证据、pinned Base 本地配置与权重、当前 Python 对应 scripts/bin 中 `swift`、GPU 检查，使单项失败不会短路其他检查；测试多阻断并存及 Windows venv PATH 探测。
- [x] 3.2 更新预检 API 类型/响应和工作台 UI，按类别展示通过/阻断/提醒、各自修复指引及未启动状态；用 API/UI 测试验证响应字段、错误可读性以及按钮仅在所有硬条件通过时启用。
- [x] 3.3 增加 `swift` 缺失、模型配置缺失、权重/分片缺失和许可/质量提醒的文档说明；核对预检没有 pip、HTTP 下载或训练副作用的测试。

## 4. 端到端回归

- [x] 4.1 对修复后的 CPT 数据执行构建/登记/预检/API 端到端夹具，验证新数据可越过角色校验、旧数据仍可追溯、缺失的本地依赖仍准确阻断；运行 CPT 相关测试集。
- [x] 4.2 运行完整后端测试与 OpenSpec strict validate，修复本 change 引入的回归；在依赖和本地 Base 权重尚未准备时明确报告未执行真实 GPU 训练，不将模拟测试标为训练通过。
