# 阶段预检确认的安全边界

阶段预检与开始训练是两个独立动作。`POST /api/training/preflight` 只返回兼容性检查；用户显式确认后，`POST /api/training/preflight/confirm` 才签发一次性确认令牌。确认接口不会创建目录、Run 或训练进程，也不会下载模型/数据。当前 Mini SFT 仍继续使用既有 Launcher 的启动协议；通用 stage Run 启动将在后续任务接入该令牌消费。

## 令牌绑定内容

令牌在服务进程内存保存，默认有效期 300 秒。绑定摘要包含：

- `stage`、批准的 `profile_id`、算法和请求中的 dataset/model ID；
- workspace 内批准 recipe 的相对路径及 SHA-256；
- 每个数据集的 manifest SHA-256 与 canonical 内容 SHA-256；
- 父模型 ID 与核验后的模型内容摘要；
- 服务端规划的输出位置 `runs/<stage>/<profile_id>/<confirmation_id>`。

浏览器不应把确认令牌持久化进 URL、localStorage、配置文件或日志。页面刷新后若令牌已丢失/过期，应重新执行预检并再次确认。

## 单次消费规则

未来训练启动端点必须在创建进程前调用 `StagePreflightConfirmations.consume()`：

1. 先从内存存储中移除 token，保证重放不能成功；
2. 拒绝不存在、过期或已使用的 token；
3. 将提交的阶段/profile/dataset/model ID 与原确认请求逐字比对；
4. 重新执行只读预检，对比配方、manifest/content hash、父模型摘要及预检状态；
5. 任一字段或证据发生变化即拒绝，要求用户重新预检确认。

失败的消费尝试同样会使 token 失效。token 不写入文件系统或 Run artifact。后续如改成持久确认存储，必须保留同等 TTL、抗重放和证据重算语义。

## 其他约束

- 通用阶段确认只允许 loopback UI 服务；启用 `allow_remote` 时拒绝签发。
- 请求 DTO 采用拒绝未知字段策略，不允许传入任意 recipe、文件路径、shell 命令、环境变量或超参数覆盖。
- 输出位置完全由服务端生成；确认阶段只展示规划路径，不预创建目录。
- 本地 GPU 状态只作为观察信息，不作为显存保证。质量/许可状态未知会明确提示；摘要不匹配、阶段/角色冲突及无效父模型属于硬阻断。
- 签发确认令牌本身不等于开始训练，训练只会在后续显式启动操作中发生。
