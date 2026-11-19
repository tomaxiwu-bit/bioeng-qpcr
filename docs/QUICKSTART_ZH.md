# 五分钟试用 bioeng-qpcr

本指南对应 0.6.8。软件在本机处理数据，面向科研与教学。

## Windows：下载后直接运行

1. 打开 [GitHub Releases](https://github.com/tomaxiwu-bit/bioeng-qpcr/releases)，选择
   `bioeng-qpcr-0.6.8-win64-portable.zip` 与同名 `.sha256` 校验文件。
   如果公开页面尚无 0.6.8，可先按下方源码方式运行；旧包不包含本页全部功能。
2. 使用解压工具解压整个 ZIP，保留 `_internal` 文件夹，再运行 `bioeng-qpcr.exe`。
   不要直接在压缩包内双击，也不要只复制 EXE。此便携版未签名。
3. 点击“打开综合实验工作台”，在默认浏览器使用软件。关闭 Windows 程序时，本地网页服务也会结束。

可在 PowerShell 中核对下载文件：

```powershell
Get-FileHash -Algorithm SHA256 .\bioeng-qpcr-0.6.8-win64-portable.zip
Get-Content .\bioeng-qpcr-0.6.8-win64-portable.zip.sha256
```

两处 SHA-256 应一致。

## Python：从源码启动

安装 Python 3.10 或更新版本（发布验证环境为 Python 3.12）和 Git 后，在 PowerShell 执行：

```powershell
git clone https://github.com/tomaxiwu-bit/bioeng-qpcr.git
cd bioeng-qpcr
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[web,fba]"
.\.venv\Scripts\python.exe -m bioeng_qpcr.web
```

浏览器打开 <http://127.0.0.1:8008/lab>。macOS / Linux 使用
`.venv/bin/python` 代替 `.\.venv\Scripts\python.exe`。首次安装需要联网，示例分析在本地运行。

## 第一次 qPCR 分析

1. 在综合首页进入“qPCR 实验”，打开“脑 / 肾基因表达对比”公开示例。
2. 确认“样品与引物 / 探针”“反应体系与板位”“温度循环与通道”三个设计步骤。
3. 在“运行与数据采集”点击“回放公开测量数据”。这会读入公开 Ct，不会连接仪器或生成测量数据。
4. 计算 Ct，执行质量控制，并查看警告和无法评估项。示例没有原始扩增、熔解和完整对照，这些信息保持缺失。
5. 使用预设目标 c_myc、内参 GAPDH、校准样品 brain；阅读并确认仅用于方法演示的 ΔΔCt 效率假设，再执行定量。
6. 预期结果：brain = 1，kidney ≈ **5.6373**；这是公开方法示例，不是组间生物学统计推断。
7. 进入报告页，生成报告版本，下载 CSV、完整项目 JSON 和自包含 HTML 报告。

[报告样例](demo/qpcr-report.html)可下载后在浏览器离线打开。它只包含公开示例数据，
包括数据出处、许可、方法假设和缺失证据。SVG/PDF/PNG 图表可在定量页下载。

## 换成自己的数据

先核对[CSV 导入规范](CSV_IMPORT_SPEC.md)，使用模块自带模板准备数据，或使用支持的 RDML 文件。
仪器孔位、样品和检测项目需与设计相符。技术重复共用生物样品 ID；独立生物学重复使用不同 ID。
未检出值保持未检出，不填 0 或最大周期。部分重复未检出的样品不会被当作完整定量结果。

效率校正需要实测效率及其依据。ΔΔCt 要求显式确认效率假设；同组不能与自身做处理/对照比较。
实验室阈值应按实际 SOP 调整。readiness 报告不构成完整 MIQE 合规认证。

## 保存与反馈

浏览器分析可在“统一分析历史”恢复；qPCR 的项目和运行历史保存在各自项目中。
Windows 默认数据库是 `%LOCALAPPDATA%\bioeng-qpcr\workbench.sqlite3`，位于程序文件夹外。
转移电脑前导出项目/分析 JSON，较大历史可在程序关闭后备份整个数据库。
导入备份会建立新项目/记录，历史结果需要复核和重新计算。

反馈请使用 [Issues](https://github.com/tomaxiwu-bit/bioeng-qpcr/issues)，附软件版本、操作步骤、
错误文字和允许公开的最小脱敏输入。不要提交真实身份信息、未公开实验数据或登录凭据。
