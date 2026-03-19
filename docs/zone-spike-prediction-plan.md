# Zone-Level RTM Spike Prediction — 调研与实施规划

> 基于 Lanxin 的讨论文档 (2025-12-14 晚高峰 LZ_CPS spike 案例)
> 调研日期: 2026-03-19

---

## 一、现状审计：我们有什么，缺什么

### ✅ 已有数据

| 数据 | 表/来源 | 粒度 | 时间范围 | 覆盖 |
|------|---------|------|---------|------|
| DAM LMP | `dam_lmp_hist` | 小时 | 2015-01 ~ 2026-03 | 15 SPs |
| RTM LMP | `rtm_lmp_hist` | 15min (4 intervals/hr) | 2015-01 ~ 2026-03 | 15 SPs |
| RTM LMP + Components | `rtm_lmp_api` | 5min | 2026-02-09 ~ now | 1092 SPs (含 resource nodes) |
| DAM Ancillary Services | `dam_asmcpc_hist` | 小时 | 2015-01 ~ 2026-02 | RegDn/Up, RRS, NSPIN, ECRS |
| Fuel Mix | `fuel_mix_hist` | 15min | 2007-01 ~ 2024-12 | Wind, Solar, Gas, Gas-CC, Coal, Nuclear, Hydro... |

### 关键观察

1. **RTM congestion component 只有 5 周数据** (`rtm_lmp_api` 从 2026-02-09 开始)。文档要求的 zone-vs-hub spread 及其导数，用 `rtm_lmp_hist` 可以算（LMP 差值），但无法分离 energy/congestion/loss 三个分量。
2. **Fuel mix 数据断在 2024-12-31**，缺最近 3 个月。需要恢复或补爬。
3. **2025-12-14 案例日**：RTM hist 有数据，DAM hist 有数据。LZ_CPS spike 确认存在（Hour 21: $686.2 max, $630.8 avg）。Houston 同期仅 $97.7。区域性特征极明显。

### ❌ 完全缺失的数据

| 文档要求 | ERCOT 数据产品 | 状态 | 获取难度 |
|---------|---------------|------|---------|
| **RT Reserve Margin (PRC)** | NP6-323-CD (实时), NP6-792-ER (历史周报) | 未爬 | 🟡 中等 — ERCOT Public API 有，需写爬虫 |
| **ORDC Price Adders** | NP6-323-CD / NP6-792-ER | 未爬 | 🟡 同上（同一 report） |
| **Binding Constraints / Shadow Prices** | NP6-86-CD (实时 SCED 级) | 未爬 | 🟡 中等 — 公开数据，但数据量大（每 SCED interval ~5min） |
| **Wind Forecast (STWPF)** | NP4-732-CD | 未爬 | 🟡 中等 — 每小时更新，含 48h 历史 + 168h 预测 |
| **Wind Forecast Error** | 需要 NP4-732-CD 的 forecast vs fuel_mix 的 actual | 计算得出 | 🟢 有 actual（fuel_mix），缺 forecast |
| **Zone-level 天气数据** | NOAA HRRR (3km 分辨率) 或 Open-Meteo API | 未接入 | 🟢 容易 — Herbie 库 (`pip install herbie-data`) 或 Open-Meteo 免费 API |
| **System Load Forecast vs Actual** | NP3-233-CD (DAM load forecast) / NP3-565-CD (7-day STLF) | 未爬 | 🟡 中等 |
| **Net Load** | 计算得出：Total Load - Wind - Solar | 🟢 有原料 | 从 fuel_mix 计算 |

### ❌ 模型架构缺失

| 文档要求 | 当前状态 |
|---------|---------|
| CfC / LTC / Neural ODE | 完全没有。当前只有 CatBoost/LightGBM |
| Zone-level Regime Switching | 没有。当前 Spike 是 system-wide binary |
| 三层标签 (SpikeEvent / LeadSpike / Regime) | 没有。当前标签是简单阈值 |
| 5min 级预测频率 | 没有。当前是小时级 |

---

## 二、数据获取计划

### Phase 0: 补全已有数据 (1 天)

1. **Fuel Mix 2025-01 ~ 2026-03**: 补爬 15 个月断档
2. **RTM congestion components 历史**: `rtm_lmp_api` 只有 5 周。检查 ERCOT 是否提供历史 5min LMP with components (可能需要从 ERCOT MIS 下载 CSV archives)

### Phase 1: 新数据源爬虫 (3-5 天)

每个数据源需要: 理解 API 格式 → 写爬虫 → 历史回填 → 存入 SQLite → 写质量检查。

| 优先级 | 数据源 | ERCOT Product | 为什么重要 | 工作量 |
|--------|--------|--------------|-----------|-------|
| P0 | **RT Reserves + ORDC** | NP6-792-ER (历史), NP6-323-CD (实时) | 文档说的 "预测 regime 而非价格" — reserve margin 是最直接的 regime 信号 | 2 天 |
| P0 | **Zone-level 天气** | Open-Meteo API (免费) 或 HRRR via Herbie | 温度异常、降温速度、冷锋指标 — 对 LZ_CPS 尤其关键 | 1 天 |
| P1 | **Wind Forecast** | NP4-732-CD | 计算 forecast error (surprise) — 文档说的 "比预期更没风" | 1 天 |
| P1 | **Binding Constraints** | NP6-86-CD | shadow price 直接解释 congestion-driven spike | 2 天 |
| P2 | **System Load Forecast** | NP3-565-CD | load forecast error → demand surprise | 1 天 |

### 天气数据方案对比

| 方案 | 优势 | 劣势 | 推荐 |
|------|------|------|------|
| **Open-Meteo API** | 免费、REST、历史 archive、JSON | 分辨率 ~11km（GFS）| ✅ MVP 阶段用这个 |
| **HRRR via Herbie** | 3km 分辨率、官方 NOAA | 需下载 GRIB2 大文件、计算资源 | 后续升级 |
| **NOAA Weather API** | 官方、稳定 | 只有 forecast 不好拿 historical | ❌ |

**天气测站选择** (对应文档提到的 Zone-level)：
- LZ_CPS → San Antonio (29.42°N, 98.49°W)
- LZ_WEST → Midland/Odessa (31.95°N, 102.18°W)
- LZ_HOUSTON → Houston (29.76°N, 95.37°W)
- HB_NORTH → Dallas/Fort Worth (32.78°N, 96.80°W)
- HB_SOUTH → Corpus Christi (27.80°N, 97.40°W)
- System-wide → 加权平均或选 Austin (30.27°N, 97.74°W)

---

## 三、标签体系重构

### 当前标签 (过于简单)
```python
spike = rtm_lmp > max(100, 3 * rolling_24h_mean)
```

### 文档提出的三层标签

#### Layer 1: SpikeEvent_z(t)
```python
# 价格条件 (二选一)
price_cond = (P_z >= P_hi) | (P_z >= rolling_Q99_30d)

# 约束主导条件 (至少一个)
spread_z_hub = P_z - P_hub
spread_z_hou = P_z - P_houston
constraint_cond = (spread_z_hub >= S_hi) | (spread_z_hou >= S_cps_hou_hi)

# 持续时间条件
raw = price_cond & constraint_cond
spike_event = rolling_min_count(raw, m=3)  # 5min 粒度: 连续 15min
```

初始阈值: P_hi=400, S_hi=50, S_cps_hou_hi=80 (后续用滚动分位数替代)

#### Layer 2: LeadSpike_z(t, H)
```python
# 未来 H 分钟内是否会出现 SpikeEvent
lead_spike = spike_event.rolling(H//dt, min_periods=1).max().shift(-H//dt)
```
H=60 或 90 分钟。这是训练目标。

#### Layer 3: Regime_z(t)
```python
regime = 'Normal'   # default
regime[(P_z >= P_mid) | (spread_z_hub >= S_mid)] = 'Tight'
regime[spike_event] = 'Scarcity'
```
P_mid=150, S_mid=20

### 关键注意：数据粒度

文档要求 **5min** 粒度。我们的历史数据:
- `rtm_lmp_hist`: 15min (4 intervals/hr) — 11 年
- `rtm_lmp_api`: 5min — 仅 5 周

**决策**: 先用 15min 构建训练集（覆盖 11 年），rtm_lmp_api 的 5min 数据做近期验证。生产环境用 5min 推理。

---

## 四、特征工程计划

### 文档要求的特征清单 vs 数据可用性

#### 电力系统侧

| 特征 | 需要的数据 | 我们有吗 | 计划 |
|------|-----------|---------|------|
| Net Load | Total Load - Wind - Solar | ✅ 可从 fuel_mix 计算 | Phase 0 |
| ΔNet Load / Δt | 同上 | ✅ | Phase 0 |
| Δ²Net Load / Δt² | 同上 | ✅ | Phase 0 |
| Zone-Hub Spread | LZ_z LMP - Hub LMP | ✅ rtm_lmp_hist 有 | Phase 0 |
| d/dt(Spread) | 同上 | ✅ | Phase 0 |
| Online Gas Capacity vs Ramp | 需要 unit-level data | ❌ | Phase 2+ (可用 fuel_mix 的 Gas 变化率近似) |
| RT Reserve Margin (PRC) | NP6-792-ER | ❌ | Phase 1 爬虫 |
| Storage Net Output | fuel_mix 有 storage 类型? | 🟡 需确认 | Phase 0 检查 |
| ORDC Price Adder | NP6-792-ER | ❌ | Phase 1 爬虫 |
| Binding Constraints | NP6-86-CD | ❌ | Phase 1 爬虫 |

#### 天气侧 (Zone-level)

| 特征 | 数据源 | 公式 | 计划 |
|------|--------|------|------|
| T_anom_z(t) | Open-Meteo / HRRR | T_z(t) - T_norm_z(t) (30d rolling mean) | Phase 1 |
| ΔT_z(t) | 同上 | T_z(t) - T_z(t-1h) | Phase 1 |
| Wind Chill WC_z(t) | 需 T + WindSpeed | 标准 NWS 公式 | Phase 1 |
| Cold Front CF_z(t) | 需 T + WindDir | ΔT < -3°C/h AND wind shift to N | Phase 1 |

#### 风电侧

| 特征 | 数据源 | 公式 | 计划 |
|------|--------|------|------|
| W(t) | fuel_mix_hist | Wind generation MW | ✅ 有 |
| W_anom(t) | 同上 | W(t) - rolling_mean_30d(W) | ✅ 可计算 |
| ΔW(t) | 同上 | W(t) - W(t-1) | ✅ |
| CF_w(t) | 需 capacity data | W(t) / W_cap | 🟡 hardcode capacity |
| e_w(t) | NP4-732-CD forecast | W_actual - W_forecast | ❌ Phase 1 |

---

## 五、模型架构计划

### 阶段性方案

| 阶段 | 模型 | 输入 | 输出 | 目标 |
|------|------|------|------|------|
| **V1: GBM Baseline** | LightGBM / CatBoost | 新特征集 (15min) | LeadSpike_z(t, 60min) binary | PR-AUC baseline |
| **V2: CfC / LTC** | CfC-RNN (PyTorch) | 同上 + 不规则时间步 | LeadSpike + Regime 概率 | 超越 GBM |
| **V3: 时空模型** | CfC + GNN (zone 图结构) | 多 zone 联合 | 联合 zone regime | 最终形态 |

### V1 为什么先做 GBM

1. 文档说 "baseline XGBoost/LightGBM 作为对照" — 这是正确的
2. GBM 在表格数据上通常是强 baseline，且我们有经验
3. 用 GBM 快速验证特征有效性，再上 CfC
4. 如果 GBM 就能达到 reasonable AUC (>0.85)，说明特征工程到位了

### V2 CfC/LTC 实现方案

```
pip install ncps  # Neural Circuit Policies — CfC/LTC 的 PyTorch 实现
```

- 来自 MIT (Hasani et al., Nature Machine Intelligence 2022)
- PyTorch 原生支持
- 天然处理不规则时间步 (文档核心需求)
- 20-50 hidden units 足够 (参数少，不容易过拟合)

---

## 六、数据获取优先级路线图

```
Week 1: Data Foundation
├─ Day 1-2: 补全 fuel_mix (2025-01 ~ now)
│           写 zone-spread 计算 + Net Load 计算
│           验证 2025-12-14 案例可复现
├─ Day 3-4: Open-Meteo 天气数据接入
│           6 个城市, 2015~now, hourly
│           算 T_anom, ΔT, Wind Chill, Cold Front
├─ Day 5:   ERCOT RT Reserves (NP6-792-ER) 爬虫
│           历史周报 → SQLite

Week 2: Labels + Features + GBM Baseline
├─ Day 1-2: 三层标签实现 (SpikeEvent/LeadSpike/Regime)
│           per zone, 15min 粒度
├─ Day 3-4: 特征管道 v2 — 整合所有新特征
│           电力系统 + 天气 + 风电 + spread
├─ Day 5:   GBM baseline 训练 + 评估
│           per zone, LeadSpike(60min) 预测
│           评估: PR-AUC, event-level recall

Week 3: Advanced Model + Strategy
├─ Day 1-3: CfC/LTC 模型实现
│           PyTorch, ncps 库
│           对比 GBM baseline
├─ Day 4-5: BESS Hold-SOC 策略 v1
│           三态调度 (Normal/Tight/Scarcity)
│           2025-12-14 反事实回测

Week 4: Integration + Validation
├─ Day 1-2: Wind forecast (NP4-732-CD) 爬虫
│           Binding constraints (NP6-86-CD) 爬虫
├─ Day 3-4: 特征扩展 + 模型重训
├─ Day 5:   API 集成 — /predictions/spike/zone-regime
│           5min 推理频率
│           输出: {regime, p_hat, recommended_power_cap}
```

---

## 七、反事实验证计划 (2025-12-14 案例)

在任何模型上线前，必须通过这个 litmus test：

### 数据
- RTM 15min LMP: LZ_CPS, LZ_WEST, LZ_HOUSTON, HB_HUBAVG (✅ 有)
- Fuel Mix: Wind, Gas, Solar (✅ 有到 2024-12)
- 天气: San Antonio 当日温度 (Phase 1 获取)

### 预期模型行为
1. **16:30**: Net Load 拐点检测 → Regime 从 Normal → Tight
2. **18:00**: Spread(CPS-Houston) 开始扩大 → 维持 Tight
3. **19:30**: Reserve tightening + spread 加速 → Tight → Scarcity
4. **20:00-22:00**: LeadSpike = 1 → Hold-SOC 策略 → 全功率放电

### 评估
- 传统策略 (16:30-20:00 放电): 在 $100-$325 区间释放 SOC
- 改进策略 (Hold-SOC 到 20:00): 在 $400-$686 区间释放 SOC
- 预期收益差: 30-60%+ 提升

---

## 八、技术栈

| 组件 | 工具 | 备注 |
|------|------|------|
| 数据爬虫 | Python + ERCOT Public API | 复用现有 ercot_client.py |
| 天气数据 | Open-Meteo API (免费, REST) | pip install openmeteo-requests |
| HRRR (未来) | Herbie library | pip install herbie-data |
| 特征存储 | SQLite (现有) | 新建表 |
| GBM 模型 | LightGBM + CatBoost + Optuna | 复用现有管道 |
| CfC/LTC 模型 | PyTorch + ncps | pip install ncps |
| API 集成 | FastAPI (现有) | 新增 zone-regime 端点 |

---

## 九、风险与开放问题

1. **RTM congestion components 历史数据**: `rtm_lmp_api` 只有 5 周。ERCOT 是否提供历史 5min LMP with components? 如果没有，只能用 LMP spread 近似 congestion（精度较低）。

2. **天气数据对齐**: Open-Meteo 的历史天气是否准确覆盖 2015-2026 全部时间范围? 需要验证。

3. **CfC 模型在 ERCOT 场景的有效性**: CfC 论文主要在自动驾驶和 Physionet 验证。电力市场 regime switching 是新场景，需要实验。

4. **15min → 5min 粒度过渡**: 训练用 15min (11 年) 还是 5min (5 周)? 建议 15min 训练 + 5min 推理 (fine-tune)。

5. **Binding constraints 数据量**: NP6-86-CD 每 SCED interval (~5min) 一条，10 年的数据量可能很大。

---

*这份规划不打折。每个阶段都基于对实际数据的调查，不假设任何我们没有的东西。*
