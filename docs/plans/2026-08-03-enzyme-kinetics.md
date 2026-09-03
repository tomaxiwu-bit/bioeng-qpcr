# Phase 2: 酶动力学模块实现计划

> **目标:** 为 bioeng-qpcr 添加酶动力学分析能力——米氏方程拟合、抑制类型判别、Km 范围校验，并通过 MCP 工具暴露给 AI 助手。

**架构:** 新增 `enzyme_kinetics/` 子包，遵循现有 calculations 模块模式（dataclass 结果 + scipy 拟合 + 纯函数）。校验规则遵循 `@validation_rule` 装饰器模式。MCP 新增 3 个工具（总数 12 < 15）。

**技术栈:** scipy.optimize.curve_fit, numpy, matplotlib, pydantic

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/bioeng_qpcr/enzyme_kinetics/__init__.py` | 公共 API 导出 |
| `src/bioeng_qpcr/enzyme_kinetics/michaelis_menten.py` | 米氏方程 V=Vmax*[S]/(Km+[S]) 拟合 |
| `src/bioeng_qpcr/enzyme_kinetics/inhibition.py` | 抑制类型判别（竞争/非竞争/反竞争/混合） |
| `src/bioeng_qpcr/enzyme_kinetics/validation.py` | ENZ-001 Km 物理范围, ENZ-002 底物浓度跨度 |
| `src/bioeng_qpcr/plotting/enzyme_kinetics.py` | MM 曲线图 + Lineweaver-Burk 图 |
| `src/bioeng_qpcr/mcp/server.py` | +3 MCP 工具（修改） |
| `tests/enzyme_kinetics/` | 全部测试 |
| `tests/plotting/test_enzyme_kinetics.py` | 绘图测试 |

## 抑制判别算法

通过多组 [I] 浓度下的 (Km_app, Vmax_app) 变化趋势判别：

| 类型 | Km 趋势 | Vmax 趋势 | LB 线特征 |
|------|---------|-----------|-----------|
| 竞争性 | 增大 | 不变 | 交于 y 轴 |
| 非竞争性 | 不变 | 减小 | 交于 x 轴 |
| 反竞争性 | 减小 | 减小 | 平行 |
| 混合型 | 变化 | 变化 | 交于第二象限 |
| 无抑制 | 不变 | 不变 | 重合 |

判别使用变异系数 (CV) 阈值 5%：Km_CV > 5% 且 Vmax_CV < 5% → 竞争性，以此类推。

## MCP 工具（3 个新增，总计 12 < 15）

1. `enzyme_fit_kinetics` — 从 CSV 拟合 MM 参数
2. `enzyme_classify_inhibition` — 从多 [I] 浓度 CSV 判别抑制类型
3. `enzyme_plot_kinetics` — 绘制 MM 曲线或 LB 图（base64 PNG）
