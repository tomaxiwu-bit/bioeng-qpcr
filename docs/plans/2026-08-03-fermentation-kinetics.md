# Phase 3: 发酵动力学模块实现计划

> **目标:** 为 bioeng-qpcr 添加发酵动力学分析能力——Monod 方程 ODE 求解、参数拟合、可辨识性分析、μmax 物理约束校验，并通过 MCP 工具暴露给 AI 助手。

**架构:** 新增 `fermentation_kinetics/` 子包，遵循现有 enzyme_kinetics 模块模式（dataclass 结果 + scipy 求解 + 纯函数）。MCP 新增 2 个工具（总数 14 < 15）。

**技术栈:** scipy.integrate.solve_ivp, scipy.optimize.least_squares, numpy, matplotlib, pydantic

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/bioeng_qpcr/fermentation_kinetics/__init__.py` | 公共 API 导出 |
| `src/bioeng_qpcr/fermentation_kinetics/monod.py` | Monod 方程 + 批式发酵 ODE 求解 + 参数拟合 |
| `src/bioeng_qpcr/fermentation_kinetics/identifiability.py` | Fisher 信息矩阵 + 参数相关性分析 |
| `src/bioeng_qpcr/fermentation_kinetics/validation.py` | FERM-001 μmax 范围, FERM-002 数据充分性 |
| `src/bioeng_qpcr/plotting/fermentation_kinetics.py` | 生长曲线 + 底物消耗图 |
| `src/bioeng_qpcr/mcp/server.py` | +2 MCP 工具（修改） |
| `tests/fermentation_kinetics/` | 全部测试 |
| `tests/plotting/test_fermentation_kinetics.py` | 绘图测试 |

## Monod 方程与 ODE 系统

Monod 动力学方程：
```
μ = μmax * S / (Ks + S)
```

批式发酵 ODE 系统：
```
dX/dt = μ * X          (生物量生长)
dS/dt = -μ * X / Yxs   (底物消耗)
```

其中：
- μmax: 最大比生长速率 (h⁻¹)
- Ks: 半饱和常数 (g/L)
- Yxs: 表观产率系数 (g biomass / g substrate)
- X: 生物量浓度 (g/L)
- S: 底物浓度 (g/L)

## 参数拟合算法

使用 `scipy.optimize.least_squares` 进行多变量参数拟合：
1. 从时间序列数据 (t, X, S) 中提取
2. 用 `solve_ivp` 求解 ODE 获取预测值
3. 最小化残差平方和 (观测值 - 预测值)
4. 初始估计：μmax₀ = 0.5, Ks₀ = 中位底物浓度, Yxs₀ = ΔX/ΔS

## 可辨识性分析

### Fisher 信息矩阵 (FIM)
```
FIM = J^T * Σ⁻¹ * J
```
其中 J 是灵敏度矩阵（∂y/∂p），Σ 是测量噪声协方差。

### 判据
- **特征值比**: max(λ)/min(λ) > 1000 → 参数相关性高，可辨识性差
- **参数相关系数**: |r| > 0.9 → 两个参数高度相关
- **置信区间**: CI 宽度 / 参数值 > 50% → 实际不可辨识

## 校验规则

| 规则 | 描述 | 阈值 |
|------|------|------|
| FERM-001 | μmax 物理范围 | 0.001 - 5.0 h⁻¹ |
| FERM-002 | 数据充分性 | ≥5 时间点, 含 X 和 S 数据 |

## MCP 工具（2 个新增，总计 14 < 15）

1. `ferment_fit_monod` — 从 CSV 拟合 Monod 参数 + 可辨识性分析 + 校验
2. `ferment_plot_kinetics` — 绘制生长曲线和底物消耗图（base64 PNG）
