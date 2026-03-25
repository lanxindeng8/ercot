# TrueFlux — 项目路线图

> ERCOT 电力市场预测 + 智能调度平台
> 最后更新: 2026-03-25

---

## 总览

TrueFlux 从 ERCOT 实时/DAM 数据出发，训练 ML 模型预测价格、风电、负荷和价格尖峰，并据此生成矿场开停机信号和电池套利策略。最终目标：SaaS 信号服务。

## 完成进度

| Sprint | 内容 | 状态 | 日期 |
|--------|------|------|------|
| 1 | 基础设施：数据审计、Gap 回填、训练管道 | ✅ 完成 | 03-17 |
| 2 | 模型训练：DAM v2, RTM, Delta-Spread, Spike + 质量修复 | ✅ 完成 | 03-17~18 |
| 3 | 产品化：Prediction API v3→v4, Dashboard | ✅ 完成 | 03-18 |
| 4 | Wind/Load/BESS 模型 + 全栈集成 | ✅ 完成 | 03-18 |
| 5 | 部署：LaunchAgent, SQLite 迁移, 准确率追踪 | ✅ 完成 | 03-18 |
| 6 | 商业化 MVP：矿场调度, BESS 套利, 客户 Dashboard | ✅ 完成 | 03-18 |
| 7.1 | API Key 管理 + Rate Limiting | ✅ 完成 | 03-18 |
| Phase 0 | Spike 数据获取基础设施 (weather/reserves/wind) | ✅ 完成 | 03-19 |
| Phase 1 | Zone-level Spike V2 (labels + features + 14 SP models + Optuna + API) | ✅ 完成 | 03-19~22 |
| Ops | RTM retrain 80 features (15 SP × 3 horizons) | ✅ 完成 | 03-22 |
| Ops | 数据补全 (reserves 2016-2025, wind 2022-2026) | ✅ 完成 | 03-22~23 |
| Ops | LaunchAgent 重构 (9→7 jobs, API/CDR 分离) | ✅ 完成 | 03-23 |
| Ops | ERCOT 429 修复 (退避策略, 文档) | ✅ 完成 | 03-23 |

---

## 当前待办

### 高优先级

| # | 任务 | 说明 |
|---|------|------|
| 1 | **RTM 模型回退修复** | 80-feature retrain 后 lcra/hubavg/houston 24h MAE 翻倍。方案: 增加 Optuna trials (50+) 或 horizon-aware feature selection |
| 2 | **验证新 LaunchAgent 架构** | dam-pipeline 首次运行待验证 (每天 14:00), rtm-lmp-api 429 退避效果待观察 |

### 中优先级

| # | 任务 | 说明 |
|---|------|------|
| 3 | **Phase 2: 多 horizon spike prediction** | 15min/30min/4h spike 预测，利用已完成的数据基础设施 |
| 4 | **实时管道优化** | CDR → zone aggregation，提升实时特征质量 |

### 低优先级 / 暂停

| # | 任务 | 说明 |
|---|------|------|
| 5 | Sprint 7.2 多租户隔离 | 按 API key 隔离数据视图 |
| 6 | Sprint 7.3 生产部署 | HTTPS, 域名, 公网 |
| 7 | Sprint 8 模型增强 | 自动重训, drift 检测, A/B 测试 |
| 8 | Sprint 9 客户获取 | Landing page, Stripe |

---

## 关键指标

| 模型 | 核心指标 | 值 | 更新日期 |
|------|---------|-----|---------|
| DAM v2 | MAE (best/worst SP) | 5.37 / 7.50 | 03-19 |
| RTM 1h | MAE (best/worst SP) | 10.9 / 20.2 | 03-22 |
| Delta-Spread | Sharpe | 34.86 | 03-18 |
| Spike V2 | PR-AUC improvement | +0.22 avg | 03-22 |
| Wind | MAE | 14.3 MW | 03-18 |
| Load | MAPE | 0.81% | 03-18 |

## 数据基础设施

| 数据 | 行数 | 覆盖 | 状态 |
|------|------|------|------|
| RTM 价格 | 17.2M | 2015–实时 | ✅ 完整 |
| DAM 价格 | 1.4M | 2015–实时 | ✅ 完整 |
| RT Reserves/ORDC | 1.1M | 2016–2025 | ✅ 完整 |
| Weather (6站) | 590K | 2015–2026 | ✅ 完整 |
| Wind Forecast | 114K | 2022-12–2026-03 | ✅ 完整 |
| Spike Labels | 5.3M | 2015–2026, 15 SP | ✅ 完整 |
| 测试 | 323 | — | ✅ 全部通过 |

## 技术债

- [x] ~~main.py 1882 行~~ → 拆为 119 行 main.py + 6 router 文件 + schemas.py + helpers.py (`807a80d`)
- [x] ~~RTM 24h 模型回退~~ → 50 trials 修复 5/6 回退 (`5fadea8`)。hb_houston/24h 仍 19.07 vs 原始 12.97（需 feature analysis）
- [x] ~~SQLite 缺 UNIQUE 约束~~ → 4 表全加 UNIQUE index，去重 26,743 行
- [x] ~~`data/__init__.py` 的 `__getattr__`~~ → 已移除 (`47357a3`)
- [x] ~~废弃 LaunchAgent plist~~ → 5 个旧 plist 移至 .archived/
- [x] ~~InfluxDB 迁移~~ → 不迁移。InfluxDB Cloud 免费 tier，时序查询给前端比 SQLite 更合适，双写成本低
- [ ] hb_houston/24h MAE 回退 (12.97→19.07)，可能需要 horizon-aware feature selection
