# 本地 Run Supervisor 协议

`LocalRunSupervisor` 是训练和评测共用的本地单活动任务生命周期基础。它只负责状态、持久记录、有界日志和进度证据，不接收任意命令，也不自行启动子进程。阶段 adapter 负责启动固定命令，并把真实事件交给该协议；SFT Launcher 的迁移由后续 OpenSpec 任务完成。

## 状态机

```text
queued → starting → running → completed
             │          ├──→ partial
             │          ├──→ failed
             │          ├──→ oom
             │          └──→ interrupted
             ├──→ failed / oom / interrupted
             └──→ interrupted
queued → failed / interrupted
```

终态不可恢复为活动态。服务重启时若本地记录仍是 queued/starting/running，协议会标记为 interrupted，不会猜测进程状态或自动续训。活动 Run 完成、失败、OOM 或中断后才释放单活动锁。

## 持久化与日志

- 状态保存在 `.onegpu-posttrain/supervisor/runs/<run_id>.json`；日志保存在对应 `logs/<run_id>.log`。
- 默认单日志上限 1 MiB，超过后只保留尾部；API读取默认最多 64 KiB。日志不参与训练控制。
- 记录使用原子替换写入；拒绝无效 run_id、越出 workspace 的输出相对路径、符号链接和重复 Run ID。
- Run ID、stage、profile、output location 在服务端生成/allowlist 解析；浏览器不得传 shell 命令或任意路径。

## 进度

只有日志中明确出现 `step/total` 证据时，`parse_training_progress()` 才输出 measured 数值；没有可解析证据时必须保持 `progress_status=unknown`，不能估算或用耗时百分比伪造训练进度。

## 当前接入范围

Mini SFT Launcher 的训练过程已接入共用生命周期，同时保留原 `/api/launcher/*`、URL、recipe 和既有详细状态记录。通用记录位于 `.onegpu-posttrain/supervisor/`，旧 Launcher 页面仍读取其兼容状态文件；训练确认后才延迟创建 supervisor 状态目录，单纯打开 UI/执行预检不写入该目录。DPO/SimPO/GRPO/CPT adapter 尚未因此自动获得执行能力。
