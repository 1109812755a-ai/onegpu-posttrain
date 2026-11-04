# AgentScope 兼容性检查

工作区当前使用 Windows Python 3.10.11。AgentScope 2.x 要求 Python 3.11 以上；本项目的 Python 下限仍为 3.10，因此候选版本先固定为 AgentScope 1.0.21。

在 Windows Python 3.10.11 的隔离虚拟环境中，`agentscope==1.0.21` 默认解析到 `mcp==2.3.0` 时导入失败：该版本移除了 AgentScope 依赖的 `streamablehttp_client`。将 MCP 约束为 `>=1.13,<2` 后，导入仍需补充 `tqdm`。安装项目 `.[ui,assistant-agentscope]` 后，AgentScope 1.0.21 和 FastAPI 0.143.3 可导入，`pip check` 通过。

后续隔离环境验证：Windows Python 3.11、3.12 可导入 AgentScope 1.0.21；两者需要 `sqlalchemy[asyncio]` 提供的 `greenlet`，因此已加入可选依赖。WSL Python 3.10 的隔离环境也可安装并导入同一版本。项目额外依赖组合在 Windows Python 3.10 上通过 `pip check`。

这些检查只覆盖安装和导入；其他版本的真实模型调用和工作台运行仍需单独验证。旧工作台的运行环境未被隔离检查修改。
