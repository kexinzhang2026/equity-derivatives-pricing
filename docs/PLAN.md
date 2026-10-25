# 分阶段计划

核心叙事（简历和面试里反复用的一句话）：
> 同一个产品用三种独立方法定价、互相验证；用真实 SPX 数据校准 Heston/SABR；用校准好的模型给 autocallable 定价，量化模型风险；最后用对冲回测拆解对冲误差的来源。

原则：每个阶段结束时，都有一组**能跑通的测试**和**一张能放进报告的图**。每个公式都有对应的推导文档（`docs/derivations/`），代码注释指向推导，方便面试时逐行讲。

---

## 阶段 0　地基（已完成）
- 项目结构：`src/derivpricing/`（models / engines / products / calibration / hedging / data）、`tests/`、`notebooks/`、`reports/`、`docs/`
- Black-Scholes 解析解 + 解析 Greeks + 隐含波动率反解
- 测试：Hull 教材数值、put-call parity、无套利边界、σ→0 极限、Greeks vs 有限差分、IV 往返
- 一个已经能讲的点：深度实值+低波动时 IV 反解是病态的（vega≈1e-9），所以市场数据一律用虚值一侧反解

## 阶段 1　定价引擎：三法互验（已完成）
| 方法 | 实现 | 验证结果 |
|---|---|---|
| 半解析 | Heston 特征函数（little Heston trap 形式）+ COS 方法 | 与 Lewis 积分（独立方法）一致到 1e-9；BS 下 N≈48 达到机器精度；ξ→0 时误差按 ξ² 收敛到 BS |
| PDE | Crank–Nicolson + Rannacher 起步 + 收益单元平均；美式用罚函数法 | 实测收敛斜率 −2.00；Rannacher 把 strike 处 gamma 误差从 0.2 降到 1e-5；美式看跌与二叉树一致到 1e-3 |
| 蒙特卡洛 | GBM 精确离散、对偶变量、控制变量；Heston QE 格式；Longstaff–Schwartz（样本外规则） | 控制变量方差缩减约 8 倍；QE 在 Feller 条件不满足时与 COS 相差 0.2 个标准误；LSM 4.4758 低于 PDE 美式 4.4867（理论上应为下界） |

结果表与四张收敛图：`reports/stage1_crossvalidation.md`，`reports/figures/`（`python scripts/stage1_crossval.py` 重新生成）。
推导：`docs/derivations/02`–`05`。

面试可以讲的发现：
- Longstaff–Schwartz 论文表 1 的有限差分基准 4.478 其实偏低，精确值约 4.4867（PDE 与 2 万步二叉树一致）
- Fang–Oosterlee 论文给的 Heston 参考值 5.785155450 与两种独立方法的结果 5.7851554344 相差 2e-8
- 2D Heston PDE（ADI）暂不做：Heston 由 COS 和 MC 两种方法互验；如果阶段 3 需要 PDE 定价 autocallable，再补

## 阶段 2　真实数据校准（已完成，报告：reports/stage2_calibration.md）
**已完成（合成数据上验证）**：数据清洗与隐含远期；Heston 全曲面校准（参数找回到 1e-4，噪声下停在噪声水平）；SABR 逐到期日（Hagan 与 SDE 模拟一致到 0.3 vol pt）；κ 的剖面分析（短端曲面上 κ 难以识别，ξ 随 κ 补偿）；Dupire 局部波动率（SVI → LV → PDE 复现 Heston 普通期权，误差 0.03 vol pt）。推导：06–08。
**数据状况**：10/23 的 Yahoo 快照在非交易时段抓取，93% 报价为 0，不可用；抓取已改为 15:30 ET 且加了检查。历史数据改用 optionsDX 免费 SPX 日终数据（2023 全年 + 2020Q1），读取接口已写好，等 Eve 注册下载。
**真实数据结果**：optionsDX 2020Q1 + 2023。Heston 单日 0.89 vol pt（短端 1.48），SABR 0.43；局部波动率 skew 是隐含 skew 的 1.95 倍；59 次周度校准中 ρ 最稳定（CV 3%），κ 与 ξ 的周变化相关系数 0.96（与合成数据的可识别性分析一致）；2020 年 3 月拟合误差升到 2.7 vol pt。

原计划：
- 数据：SPX 期权链 → 清洗（剔除零买价、宽价差、违反无套利的报价）→ 虚值一侧反解 IV → 从 put-call parity 隐含出远期和贴现率
- **Heston**：全曲面校准（COS 定价很快，适合放进优化循环），vega 加权误差，多起点避免局部最优
- **SABR**：逐到期日校准（Hagan 近似公式），对比短端/长端拟合
- 分析：拟合残差热力图；**参数稳定性**（多日快照重复校准，看 κ、θ、ρ 是否跳动，讨论 κ–θ 的可识别性问题）
- 产出图：3D 波动率曲面、市场 vs 模型微笑、参数时间序列

## 阶段 3　奇异产品与模型风险（已完成，报告：reports/stage3_model_risk.md）
**结果**：障碍期权解析解 / PDE（网格对齐后二阶收敛）/ 日度 MC（BGK 修正）一致；3 年 autocallable PDE 97.45 vs MC 97.43 ± 0.04。同一 SPX 曲面下公允票息：BS 14.0%、Heston 15.2%、局部波动率 16.0%；60% 敲入看跌：BS 4.9%、局部波动率 8.4%、Heston 9.3%（占名义本金）。原因：BS 没有 skew；Heston 与局部波动率普通期权价格一致，但远期微笑不同。推导：09。

原计划：
- 先做**下跌敲入障碍期权**：BS 下有解析解，可以验证 PDE/MC 的障碍处理（连续 vs 离散监测，Broadie–Glasserman–Kou 修正）
- 主产品 **autocallable**（高盛结构化产品核心）：观察日自动赎回 + 票息 + 下跌敲入，PDE 和 MC 两种实现
- 局部波动率：从校准后的隐含波动率曲面用 Dupire 公式构造（阶段 2 末尾，约 1 周）
- 模型风险：同一组市场数据下，BS（平值波动率）/ Heston / 局部波动率（Dupire）给出的价格差异，以及差异来自哪里（远期偏斜、波动率的波动率）
- 产出：模型间价格差异表 + 一段你自己的结论

## 阶段 4　Greeks 与对冲回测（约 1.5 周）
- Greeks：解析 / 有限差分（PDE 网格直接读）/ MC pathwise 与 likelihood ratio
- 对冲回测：模拟路径和真实 SPX 历史路径上做 delta、delta-vega 对冲
- 误差拆解：离散对冲误差（∝ gamma × 调仓间隔）、模型误设（实现波动率 ≠ 隐含波动率）、交易成本
- 产出：P&L 分布图、误差 vs 调仓频率的 log-log 图

## 阶段 5　报告与包装（约 1 周）
- README（英文，面向招聘方）、技术报告 PDF、CI（GitHub Actions 跑测试）、简历要点 3 条

---

## 需要你决定的事
1. **数据来源**（脚本已写好：`scripts/fetch_spx_chain.py`，OptionMetrics 读取接口在 `src/derivpricing/data/loaders.py`）。云端环境访问不了 Yahoo/CBOE，抓取需要在你自己电脑上运行。
   - 如果你有学校的 **WRDS / OptionMetrics** 权限：最好，有历史数据，可以做参数稳定性的时间序列
   - 否则：用 yfinance 每天抓一次 SPX 期权链快照，积累 2–4 周数据
2. ~~奇异产品~~ 已决定：障碍期权（做验证）+ autocallable（主产品）
3. **局部波动率模型**放不放进模型风险对比？加上它故事更完整（Heston vs LV 对 autocallable 定价差异是业界经典问题），但多 1 周工作量
4. **GitHub 仓库**：需要你在 GitHub 上新建一个空仓库（建议名 `derivatives-pricing-engine`），告诉我名字后我来推送
