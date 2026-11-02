# Mini 数据构建与就绪度验收

本流程把数据预算、质量证据覆盖、质量判定、处理证据和血缘作为独立维度。预算达标不代表质量通过；“证据覆盖完整”只表示适用指标均有显式状态，不代表数据获批。外部来源不可用也不等于已验证。每次正式构建使用新的、保留的输出目录，避免覆盖历史证据。

## 1. 构建 Mini 数据

在项目根目录执行（确保已安装项目依赖并可访问 Hugging Face 数据源）：

### 可选：先将完整原始来源缓存到本地

如果后续需要反复调整清洗/筛选参数，可以先缓存固定 revision 的完整源 split：

```powershell
$cacheInfo = (onegpu-posttrain cache-source `
  --config configs/data/openr1.yaml `
  --split train `
  --cache-dir data/raw/cache | Out-String) | ConvertFrom-Json
$cacheManifest = $cacheInfo.manifest
$cacheInfo | Select-Object status, row_count, byte_count, content_sha256, manifest
```

FineMath 示例：

```powershell
onegpu-posttrain cache-source `
  --config configs/data/finemath.yaml `
  --split train `
  --cache-dir data/raw/cache
```

该命令会读取**整个**上游 split 后才发布带 source revision、行数、字节数和 SHA256 的完整缓存清单；中断时不会留下可误用的完整缓存。首次运行会下载完整 split，可能耗时且占用较多磁盘空间；可用 `--cache-dir` 指定有足够空间的位置。默认目录 `data/raw/cache` 已被 Git 忽略。后续同 source identity 会校验并复用缓存；若要重新抓取，显式增加 `--refresh`。命令输出包含缓存 manifest 路径，可用于 build：

```powershell
$out = "data/processed/mini/2026-10-17-build-02"
onegpu-posttrain build-openr1 `
  --config configs/data/openr1.yaml `
  --output "$out/openr1" `
  --source-cache $cacheManifest `
  --sft-target 5000 `
  --alignment-target 3000 `
  --preference-target 2000 `
  --preference-eval-target 200 `
  --grpo-target 2000 `
  --holdout-target 500
```

FineMath build 同样使用 `--source-cache $cacheManifest`。如需保持在线流式读取，省略 `--source-cache` 和 `--input-jsonl`；这两个本地输入选项互斥。缓存校验会比较 repo/config/revision/split，并检查完整状态、行数和内容 SHA256；缺失、改动或版本不匹配时命令会失败，不会悄悄回退到网络下载。

注意：缓存只覆盖配置中的 FineMath/OpenR1 原始 split。`--benchmark-config` 仍可能从 Hugging Face 读取去污染基准；离线构建需改用本地 `--benchmark-jsonl` 文件。Tokenizer 也必须已在本机缓存；否则 Transformer 下载仍需要网络。缓存 source 不会下载或运行在本次开发测试中，只有显式执行上述 `cache-source` 才会抓取完整数据。

常见处理：

- `source identity mismatch`：检查构建 YAML 的 repo/config/revision 是否与缓存清单完全一致；不要手改 manifest。revision 改变时为新版本单独创建缓存。
- `SHA256/row count mismatch`：缓存文件可能损坏；保留原目录用于诊断，确认空间和网络正常后用 `--refresh` 重建。
- `status: failed`：此次没有提交完整 manifest；修复网络或磁盘问题后重新运行。中断下载不会复用为完整缓存。

```powershell
$out = "data/processed/mini/2026-10-17-build-01"
onegpu-posttrain build-finemath `
  --config configs/data/finemath.yaml `
  --benchmark-config configs/data/benchmarks.yaml `
  --output "$out/finemath" `
  --train-token-budget 5000000 `
  --holdout-token-budget 50000

onegpu-posttrain build-openr1 `
  --config configs/data/openr1.yaml `
  --benchmark-config configs/data/benchmarks.yaml `
  --output "$out/openr1" `
  --sft-target 5000 `
  --alignment-target 3000 `
  --preference-target 2000 `
  --preference-eval-target 200 `
  --grpo-target 2000 `
  --holdout-target 500
```

OpenR1 的 alignment core 目标用于支持 preference pair 和 SFT 分支构建，不是 readiness profile 中的独立验收集；最终验收以 processing report 记录的实际产出为准。构建未达到目标时不要伪造计数或修改 profile 来使其通过。

## 2. 运行就绪度检查

```powershell
onegpu-posttrain validate-mini-data `
  --root $out `
  --profile configs/data/mini-v1.yaml
```

默认生成到 `$out/reports/mini-readiness/<assessment_id>/`：

- `mini_readiness.json`：机器可读的 profile/hash、dataset ID/digest、单位、分维度状态、issues、验证时间和模型效果状态。
- `mini_readiness.md`：便于审阅的摘要及证据链接。

退出码 `0` 表示所有 profile 必需维度达到 ready；退出码 `2` 表示未就绪或输入/profile 无效。非零时仍会在报告可生成的情况下保存阻塞原因。不要把 `NOT_EVALUATED`、`not_measured`、缺失报告或外部 unavailable 改写成 PASS。

当同一训练阶段存在多个来源 cohort 时，profile 可用 `manifest_path`（相对 `--root`）逐个绑定 manifest。每个 key 独立读取数量、review、processing 和 lineage；路径不得越出 `--root`。DeepMath/OpenR1 SFT 对照使用 `configs/data/mini-v2.yaml`，执行时 `--root` 指向包含 profile 所列 build 目录的 `data/processed/mini`。

## 2.1 对保留数据生成新的质量复评

复评只读 canonical JSONL 和现有 manifest/processing report，并在新的目录生成报告 bundle；不会修改数据集、manifest、旧质量报告或旧 readiness 报告。复评前会重新核对 canonical 内容 SHA256。数据过大、路径越界、摘要不匹配或无法解析时会停止并报错。

示例（将占位路径替换为本地 Mini artifact 根目录）：

    onegpu-posttrain assess-mini-data-quality --root <mini-artifact-root> --profile configs/data/mini-v1.yaml --output <mini-artifact-root>/reports/mini-quality-assessments/manual-review-01

复评 bundle 的 index.json 将每个 profile 数据集关联到 dataset ID、stage、内容摘要和新质量报告摘要。新报告使用 `stage-data-quality-v1`，记录 checker 名称/版本、方法和证据覆盖。复评指标只使用现有记录及 builder 证据；没有独立验证器或 scoring 结果时会标记 not_measured，不自动下载依赖、不把未知写成 0。

可显式让 readiness 使用该 bundle：

    onegpu-posttrain validate-mini-data --root <mini-artifact-root> --profile configs/data/mini-v1.yaml --quality-assessment <mini-artifact-root>/reports/mini-quality-assessments/manual-review-01/index.json

Readiness 同时显示 quality_coverage 与 quality_evidence：覆盖通过表示目录中每项指标都有状态；质量判定仍可能是 NOT_EVALUATED。复评 bundle 必须与当前 manifest 的 dataset ID、stage 和内容摘要一致，否则视为无效证据；未指定 bundle 时仍读取 manifest 原先引用的报告。

另行验证本地产物血缘：

```powershell
onegpu-posttrain validate-lineage --root $out --require-complete
```

`validate-lineage --require-complete` 非零代表本地血缘完整性未满足。检查 JSON 中每个 issue 的 `code`、`path` 和 detail；本地 dangling reference、hash mismatch、重复 ID 等需修复后重建/重新验证。外部模型或远端资源标记为 unavailable 时，应保留其 warning，并在记录中说明未下载/无法取证的原因；不得仅为获得零 warning 而填入猜测 hash 或放宽完整性 gate。

## 3. 如何解释结果

按 dataset 查看预算、质量证据覆盖、质量判定、处理证据和 lineage。Holdout split 单独报告，不计入训练预算。`model_effect_status` 在数据 readiness 阶段为 `not_evaluated`：清洗/筛选轨迹只能说明数据如何变化，不能证明模型效果提升；效果结论需要后续注册受控训练对比。

当前 profile 的目标和计数来源见 [`mini-v1.yaml`](../../../configs/data/mini-v1.yaml)，数据处理与质量证据语义见 [`DATA_EVIDENCE.md`](DATA_EVIDENCE.md)。
