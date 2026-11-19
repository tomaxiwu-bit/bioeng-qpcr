# CSV 导入格式规范

本文档由代码生成，是 `bioeng-qpcr` 唯一权威的导入格式说明。每个分析模块都只
接受下列表格格式之一；**列名不区分大小写**，空格、连字符、点号与下划线在匹配时
等价（`Sample Name`、`sample-name`、`sample_name` 视为同一列）。

## 通用规则

1. **编码**：优先 UTF-8（可带 BOM）。若文件不是合法 UTF-8，软件会退回
   GB18030/GB2312 解码并给出警告；请尽量从 Excel 另存为「CSV UTF-8」。
2. **分隔符**：支持逗号、制表符、分号、竖线，自动探测。同一个文件内必须统一。
3. **表头**：必须有表头行，且表头不能为空、不能重复。
4. **缺失值**：可留空，或写 `NA` / `N/A` / `NaN` / `null` / `none` / `-` / `ND`。
   **不要用 0 表示缺失**——0 会被当作真实测量值参与拟合。
5. **数值**：必须能解析为有限实数。不接受 `1,234`（千分位）、`12%`、`>100`
   等带修饰的写法；请在导出前清洗。
6. **行完整性**：每行列数应与表头一致，否则会报 `SHORT_ROW` / `LONG_ROW`。
7. **重复行**：同一主键（如孔位、循环数）重复时以首行为准并给出警告。

## 各模块格式


### `amplification_traces` — 扩增曲线原始荧光数据（long 格式）

每个循环一行的原始荧光数据，用于基线校正、逐孔 sigmoid 拟合与效率估计。没有该表也能做相对定量，但只有它能做曲线级质控。

* 布局：`tidy`　|　分隔符：`','`　|　编码：`utf-8`　|　最少数据行：3

| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `position` | text | 是 | — | — | 孔位编号；同一孔位的循环必须连续且不重复 |
| `cycle` | integer | 是 | — | ≥ 1 | 循环数，从 1 开始 |
| `fluorescence` | number | 是 | — | 任意有限实数 | 该循环的荧光读数（原始值，勿做归一化） |
| `target_name` | text | 否 | — | 可留空 | 靶标名，可省略 |
| `sample_name` | text | 否 | — | 可留空 | 样本名，可省略 |

* `position` + `cycle` 组合必须唯一。
* 不要提交已经做过基线校正或归一化的荧光值；软件会自己做并记录方法。

示例：

```csv
position,cycle,fluorescence,target_name,sample_name
A1,1,1023.5,GAPDH,Control_1
A1,2,1011.2,GAPDH,Control_1
A1,3,1044.8,GAPDH,Control_1
```

### `dose_response` — 剂量-反应数据表（long 格式）

一个化合物一组长格式数据：每一行是一对（浓度, 响应）。含 `compound` 列时一个文件可放多个化合物，用于批量 IC50/EC50 与相对效价比较。

* 布局：`tidy`　|　分隔符：`','`　|　编码：`utf-8`　|　最少数据行：5

| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `compound` | text | 是 | — | — | 化合物名称；同一化合物的行构成一条曲线 |
| `concentration` | number | 是 | — | ≥ 1e-15 | 浓度，必须严格大于 0（曲线在 log10 空间拟合） |
| `response` | number | 是 | — | 任意有限实数 | 响应值，任意单位（吸光、荧光、活率百分比等） |
| `replicate` | text | 否 | — | 可留空 | 重复标签；提供后可做孔内精密度检查（POT-010） |

* 浓度必须 > 0：log10(0) 未定义；未处理/溶剂对照响应请通过界面字段或 CLI `--control-response` 传入。
* 建议至少 5 个不同浓度水平、跨度不少于 2 个数量级（POT-003 / POT-006）。
* 响应方向由数据决定：随浓度上升为 EC50，随浓度下降为 IC50；可显式指定。
* 如需百分比化，请自行先归一化；软件会按原始响应拟合并同时报告相对与绝对效价。

示例：

```csv
compound,concentration,response,replicate
CompoundA,0.01,98.2,1
CompoundA,0.1,96.4,1
CompoundA,1,88.1,1
CompoundA,10,52.3,1
CompoundA,100,14.7,1
CompoundA,1000,5.2,1
CompoundB,0.01,99.0,1
CompoundB,0.1,97.8,1
CompoundB,1,95.2,1
CompoundB,10,81.6,1
CompoundB,100,40.4,1
CompoundB,1000,8.9,1
```

### `dose_response_matrix` — 剂量-反应矩阵表（wide 格式）

第一列为浓度，其余每一列是一个化合物（表头即化合物名）。这是酶标仪导出的常见形状，软件会自动转成长格式再拟合。

* 布局：`matrix`　|　分隔符：`','`　|　编码：`utf-8`　|　最少数据行：5

| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `concentration` | number | 是 | — | ≥ 1e-15<br>别名：`conc`、`dose`、`concentration_uM` | 浓度；该列之后的每一列都是响应值 |

* 声明列之后可继续追加**任意数量**的数值列；每个追加列的表头即为该数据系列（化合物）的名称。

* 与 `dose_response` 二选一即可；自动识别时若存在 `compound` 列，则优先按 long 格式解析。
* 矩阵中的空格视为缺失，不会当作 0 参与拟合。
* 所有化合物列共用同一浓度序列。

示例：

```csv
concentration,CompoundA,CompoundB
0.01,98.2,99.0
0.1,96.4,97.8
1,88.1,95.2
10,52.3,81.6
100,14.7,40.4
1000,5.2,8.9
```

### `enzyme_kinetics` — 酶动力学数据表

`[S]` 与初速度的配对表。带 `inhibitor_concentration` 列时同一文件可同时用于抑制类型判别与全局模型选择。

* 布局：`tidy`　|　分隔符：`','`　|　编码：`utf-8`　|　最少数据行：5

| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `substrate_concentration` | number | 是 | 同 Km 单位 | ≥ 0 | 底物浓度 [S] |
| `velocity` | number | 是 | — | ≥ 0 | 初速度 v（单位/时间） |
| `inhibitor_concentration` | number | 否 | — | ≥ 0 | 抑制剂浓度 [I]；抑制剂判别必需，且需至少两个不同取值 |

* 至少需要 5 个不同 [S] 点，且应同时覆盖 Km 以下与以上（ENZ-002）。
* `inhibitor_concentration` 至少需要 2 个不同取值，其中至少一个大于 0。

示例：

```csv
substrate_concentration,velocity,inhibitor_concentration
0.1,1.12,0
0.5,4.31,0
2.0,8.94,0
10.0,12.05,0
0.5,2.11,5
2.0,5.32,5
```

### `fermentation_kinetics` — 发酵动力学时间序列

时间、生物量与底物浓度的三元组，用于 Monod 参数拟合与可辨识性分析。

* 布局：`tidy`　|　分隔符：`','`　|　编码：`utf-8`　|　最少数据行：6

| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `time` | number | 是 | h | ≥ 0 | 培养时间 |
| `biomass` | number | 是 | g/L | ≥ 0 | 生物量浓度 X |
| `substrate` | number | 是 | g/L | ≥ 0 | 限制性底物浓度 S |

* 时间必须严格递增且不得重复。
* 至少 6 个时间点，否则参数不可辨识。

示例：

```csv
time,biomass,substrate
0,0.10,10.00
2,0.21,9.42
4,0.48,8.15
6,1.02,6.40
8,1.94,4.12
10,3.05,2.10
```

### `qpcr_standard_curve` — qPCR 标准曲线与检出限数据

每行一个标准品重复，用于拟合扩增效率、Cq 预测区间，并从重复检测率与Cq 精密度评估经验 LOD、LLOQ、ULOQ 和动态范围。

* 布局：`tidy`　|　分隔符：`','`　|　编码：`utf-8`　|　最少数据行：3

| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `concentration` | number | 是 | — | ≥ 1e-300 | 标准品浓度或拷贝数；全表必须使用同一单位且严格大于 0 |
| `cq` | number | 是 | — | ≥ 0<br>可留空 | 测得的 Cq；未检出必须留空或写 Undetermined |
| `replicate` | text | 否 | — | 可留空 | 重复编号，仅用于追溯 |

* 每个浓度水平建议至少 3 个重复；未检出也必须保留为一行，否则检测率会被高估。
* LOD 判据默认检测率 ≥95%；LLOQ 还要求重复 Cq 标准差 ≤0.5，两项阈值均可按实验室 SOP 调整并随结果记录。
* 至少提供 3 个不同且有有限 Cq 的浓度水平，才能拟合标准曲线。

示例：

```csv
concentration,cq,replicate
1,,1
1,Undetermined,2
1,35.2,3
10,31.8,1
10,31.7,2
10,31.9,3
100,28.5,1
100,28.4,2
100,28.6,3
1000,25.1,1
1000,25.2,2
1000,25.0,3
```

### `qpcr_tidy` — qPCR 整理型（Level A）数据表

一行一口（one row per well）的 qPCR 结果表，是 MIQE 验证、相对定量与分组比较的唯一必需输入。列名不区分大小写，空格/连字符/下划线等价。

* 布局：`tidy`　|　分隔符：`','`　|　编码：`utf-8`　|　最少数据行：1

| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `position` | text | 是 | — | — | 孔位编号，全表唯一（如 A1、A2、P1-1） |
| `sample_name` | text | 是 | — | — | 样本名；NTC / NO_TEMPLATE / BLANK 会被识别为无模板对照 |
| `target_name` | text | 是 | — | — | 靶标（基因）名 |
| `cq` | number | 是 | — | ≥ 0<br>可留空 | 定量循环值；未检出可留空或写 Undetermined |
| `replicate_type` | text | 否 | — | `technical` / `biological` / `single`<br>可留空 | 重复类型 |
| `replicate_group` | text | 否 | — | 可留空 | 重复组标签 |
| `cq_status` | text | 否 | — | `determined` / `undetermined` / `no_template`<br>可留空 | Cq 状态；留空时按 cq 与样本名自动推断 |
| `sample_group` | text | 否 | — | 可留空 | 分组名，用于分组比较 |
| `efficiency` | number | 否 | — | 任意有限实数<br>可留空 | 扩增效率（1.0 = 100%） |
| `r2` | number | 否 | — | 0 ~ 1<br>可留空 | 标准曲线 R² |
| `slope` | number | 否 | — | 任意有限实数<br>可留空 | 标准曲线斜率 |
| `is_reference` | boolean | 否 | — | 可留空 | 是否内参基因；留空时按 reference_targets 判断 |
| `primer_forward` | text | 否 | — | 可留空 | 正向引物序列 |
| `primer_reverse` | text | 否 | — | 可留空 | 反向引物序列 |
| `probe` | text | 否 | — | 可留空 | 探针序列 |
| `lod_cq` | number | 否 | — | ≥ 0<br>可留空 | 检出限 Cq |
| `lloq_cq` | number | 否 | — | ≥ 0<br>可留空 | 定量限 Cq |

* `position` 必须全表唯一；重复孔位会被拒绝。
* 未检出（Undetermined）应留空或写 `Undetermined`，不要写 0，否则会被当成真实的极低 Cq。
* `lloq_cq` 必须小于等于 `lod_cq`。
* 同一 `target_name` 的效率/R²/引物等元数据必须一致，冲突会报 `CONFLICTING_ASSAY_METADATA`。

示例：

```csv
position,sample_name,target_name,cq,replicate_type,replicate_group,cq_status,sample_group,efficiency,r2,slope,is_reference,primer_forward,primer_reverse,probe,lod_cq,lloq_cq
A1,Control_1,GAPDH,18.42,technical,Control_1_rep1,determined,control,0.98,0.999,-3.36,true,GACCAGGTTCTGCTTGTC,CTTGCTCTCAGTATCCTCG,CGAGAAGCTGTGCTACGTC,37.0,35.0
A3,Control_1,IL6,25.13,technical,Control_1_rep1,determined,control,,,,false,AGACAGCCACTCACCTCTTC,GTTCCTGAGAGACTTGGCTG,,,
B1,Treated_1,GAPDH,18.66,technical,Treated_1_rep1,determined,treated,0.98,0.999,-3.36,true,GACCAGGTTCTGCTTGTC,CTTGCTCTCAGTATCCTCG,CGAGAAGCTGTGCTACGTC,37.0,35.0
B3,Treated_1,IL6,22.87,technical,Treated_1_rep1,determined,treated,,,,false,AGACAGCCACTCACCTCTTC,GTTCCTGAGAGACTTGGCTG,,,
C1,NTC,IL6,,single,,no_template,,,,,false,AGACAGCCACTCACCTCTTC,GTTCCTGAGAGACTTGGCTG,,,
```

## 在软件中自查

命令行：

```bash
bioeng-qpcr schema --list                  # 列出全部格式
bioeng-qpcr schema --format dose_response  # 查看某个格式的完整说明
bioeng-qpcr preview my_data.csv            # 预览任意 CSV 并自动匹配格式
bioeng-qpcr schema --format qpcr_tidy --write-template my_template.csv
```

图形界面与网页界面中，「数据预览」页会显示与 `preview` 相同的内容，并可直接
下载对应格式的模板文件。
