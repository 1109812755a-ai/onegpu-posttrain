# 前端交互与响应式回归检查

在本地 UI 服务运行后，用 Chrome/Edge 开发者工具执行以下检查。无需连接外部模型或启动训练。

## 页面与视口

分别打开 `/?page=data`、`/?page=workbench`、`/?page=policies`、`/?page=targeted`，在桌面宽度和 390 CSS px 移动视口检查：

1. 在控制台运行 `document.documentElement.scrollWidth <= document.documentElement.clientWidth`，结果应为 `true`。
2. 页面没有被裁切的表单控件；若有表格或代码块需要横向滚动，应只在该组件内部滚动。
3. “定向补数”表单字段在窄屏下单列排列；“策略对照”两组配置可纵向查看。
4. 数据集工作台的阶段/来源筛选、来源能力卡片和登记卡片在手机宽度下单列排列；hash 和长 dataset ID 不撑破卡片。

## 导航与键盘

1. 从 Policies 和 Targeted 页面点击侧栏“数据 Data”，应返回数据页，且该项带有 `aria-current="page"`。
2. 在 560px 以下验证导航图标按钮仍有名称：控制台运行 `[...document.querySelectorAll('nav button')].every(b => !!b.getAttribute('aria-label'))`，结果应为 `true`。
3. 使用 Tab 移动焦点到折叠侧栏按钮，按钮应有可见焦点环；Enter/Space 可触发按钮。
4. 在工作台用 Tab 顺序操作视图 tab、阶段、文件、筛选和登记按钮；Enter/Space 应触发按钮，原生 select/file input 保留浏览器键盘行为。
5. 进入“选择文件→核对来源与许可”，输入来源信息后取消上传；文件、阶段、许可和 attribution 应保留，错误提示不清空上下文。修正错误后重试，成功显示 dataset ID/记录数/质量状态。

## 数据导入状态

1. 选超限文件时，客户端禁用继续按钮；服务端仍需独立返回 413。
2. 无效 JSONL 应显示行号/字段问题，不显示原始记录内容；登记后返回工作台应能筛到新 dataset ID。
3. 未知/未提供许可允许登记，但必须保留原状态并显示后续本地训练确认提醒；格式有效不展示成“质量通过”。
4. “查看数据质量与血缘”应回到对应数据构建与阶段，scope 自动切换到该构建所在的真实/示例范围。

## 报告状态

运行 `node --test tests/*.test.mjs`，应覆盖 overview 404 的“报告尚未生成”、其他 HTTP 错误的可重试状态、重试成功，以及 overview 与 capabilities 独立加载。工作台还需验证来源元数据 URL 编码、导入错误安全呈现、取消中止请求且保留调用方表单状态。真实环境没有报告时，页面应提示生成命令而不是显示服务读取错误。
