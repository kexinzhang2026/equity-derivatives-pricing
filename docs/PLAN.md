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

## 阶段 1　定价引擎：三法互验（约 2 周）
| 方法 | 内容 | 关键测试 |
|---|---|---|
| 半解析 | Heston 特征函数 + **COS 方法**（Fang–Oosterlee 2008） | Heston→BS 极限（ξ→0）；N 增大时指数收敛 |
| PDE | **Crank–Nicolson**，对数价格网格；Rannacher 起步平滑非光滑收益 | 网格加密后**二阶收敛**（log-log 斜率≈2）；美式看跌的提前行权边界 |
| 蒙特卡洛 | GBM 精确离散 + Heston（QE 格式, Andersen 2008）；对偶变量、控制变量；美式用 **Longstaff–Schwartz** | 误差按 1/√N 下降；方差缩减倍数；LSM vs PDE 美式价格 |

产出图：三方法价格对比表、PDE 收敛阶图、MC 置信区间 vs 路径数、COS 收敛图。
你的数值背景在这一阶段最有发挥空间：CN 的稳定性/振荡、Rannacher 为什么有效、COS 截断区间选择，都是可以深挖的面试点。

## 阶段 2　真实数据校准（约 2 周）
- 数据：SPX 期权链 → 清洗（剔除零买价、宽价差、违反无套利的报价）→ 虚值一侧反解 IV → 从 put-call parity 隐含出远期和贴现率
- **Heston**：全曲面校准（COS 定价很快，适合放进优化循环），vega 加权误差，多起点避免局部最优
- **SABR**：逐到期日校准（Hagan 近似公式），对比短端/长端拟合
- 分析：拟合残差热力图；**参数稳定性**（多日快照重复校准，看 κ、θ、ρ 是否跳动，讨论 κ–θ 的可识别性问题）
- 产出图：3D 波动率曲面、市场 vs 模型微笑、参数时间序列

## 阶段 3　奇异产品与模型风险（约 2 周）
- 先做**下跌敲入障碍期权**：BS 下有解析解，可以验证 PDE/MC 的障碍处理（连续 vs 离散监测，Broadie–Glasserman–Kou 修正）
- 主产品 **autocallable**（高盛结构化产品核心）：观察日自动赎回 + 票息 + 下跌敲入，PDE 和 MC 两种实现
- 模型风险：同一组市场数据下，BS（平值波动率）/ Heston / 局部波动率（Dupire，可选）给出的价格差异，以及差异来自哪里（远期偏斜、波动率的波动率）
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
1. **数据来源**：云端环境访问不了 Yahoo/CBOE，所以数据抓取脚本需要在你自己电脑上运行。
   - 如果你有学校的 **WRDS / OptionMetrics** 权限：最好，有历史数据，可以做参数稳定性的时间序列
   - 否则：用 yfinance 每天抓一次 SPX 期权链快照，积累 2–4 周数据
2. **奇异产品**：建议障碍期权（做验证）+ autocallable（主产品），还是只做其中一个？
3. **局部波动率模型**放不放进模型风险对比？加上它故事更完整（Heston vs LV 对 autocallable 定价差异是业界经典问题），但多 1 周工作量
4. **GitHub 仓库**：需要你在 GitHub 上新建一个空仓库（建议名 `derivatives-pricing-engine`），告诉我名字后我来推送
