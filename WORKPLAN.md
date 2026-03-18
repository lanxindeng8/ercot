# ERCOT 开发工作计划

## 项目现状摘要 (更新于 2026-03-18)

| 组件 | 状态 | 备注 |
|------|------|------|
| **Scraper** | ✅ 运行中 | RTM 实时到 2026-03-18, DAM 到 03-19, LaunchAgent 自动 |
| **SQLite 数据** | ✅ 丰富 | RTM 1070万条, DAM 117万+142万条, 燃料混合537万条 |
| **RTM 模型** | ✅ 完成 | 多 horizon (1h/4h/24h), LightGBM, MAE=12.1 (1h) |
| **DAM v2 模型** | ✅ 完成 | LightGBM, 5 settlement points, MAE=5.88 (HB_WEST) |
| **Delta-Spread** | ✅ 完成 | LightGBM Sharpe 34.86, win rate 94.6%, 质量修复已应用 |
| **Spike** | ✅ 完成 | CatBoost, P=0.886, R=1.0, F1=0.939, 数据泄漏修复 |
| **Wind** | ✅ 完成 | GBM + Ensemble, MAE=14.3MW, Ramp POD=1.0 |
| **Load** | ✅ 完成 | CatBoost + LightGBM + Ensemble, MAPE=0.81% |
| **Battery-Strategy** | ✅ 集成 | LP 优化器接入 API, 接受 DAM 预测输入 |
| **Prediction API** | ✅ v4.0 | 7 预测端点 + 7 模型信息 + 健康检查, port 8011 |
| **Viewer** | ✅ 升级 | 7 面板 Dashboard, 暗色主题, 5min 自动刷新 |

## 数据时间范围
- RTM/DAM 历史: 2015-01-01 ~ 2026-03-18 (11年+, gap 已回填)
- RTM CDR 实时: 持续更新到当前
- DAM 实时: 持续更新，含次日预测
- Wind/Load: 从 fuel_mix_hist 提取

---

## Sprint 完成记录

### Sprint 1: 基础设施加固 ✅ (2026-03-17)
- 数据审计 + Gap 回填 (RTM +26K, DAM +6.5K)
- 统一训练数据管道 (42 features, 5 settlement points)
- 2 轮 Codex review (DST 处理, lag 对齐, fuel 去重)
- Commits: 1946303, 77639b3, 5e3136b, 5f13ceb

### Sprint 2: 模型训练 + 质量 ✅ (2026-03-17 ~ 03-18)
- DAM v2: 5 settlement points, MAE=5.88
- RTM: 多 horizon (1h/4h/24h)
- Delta-Spread: Sharpe 34.86, $57.63 avg PnL/trade
- Spike: F1=0.939, $797K revenue impact
- 质量修复: Sharpe 计算, spike 数据泄漏, metrics 处理
- Commits: 06f42da, 501ec6a, fe6f474, f19235b, 63bd71f

### Sprint 3: 产品化 ✅ (2026-03-18)
- Prediction API v3.0 → v4.0: 10+ 端点
- Predictions Dashboard: 暗色主题, 自动刷新
- Codex review: 输入校验, fetcher 泄漏, 假信号→503
- Commits: 1b08539, 4a288e7, 1b1eca5

### Sprint 4: Wind/Load/BESS ✅ (2026-03-18)
- Wind: GBM + Ensemble, MAE=14.3MW, 58 features
- Load: CatBoost + LightGBM + Ensemble, MAPE=0.81%, 87K records
- BESS: LP 优化器集成 API, DAM 价格输入
- 全栈集成: API v4.0, 7 面板 Dashboard
- Codex review: load 推理, wind bounds, BESS fallback
- Commits: bc09be1, cae56c9, 5c773de, fad1324

---

## 待办 Sprint

### Sprint 5: 部署 + 实时管道 (Priority: P0)

**目标**: 生产环境部署, 实时预测循环

1. **部署**
   - API 服务 LaunchAgent (自动重启)
   - Viewer 部署到 Vercel (或 local)
   - 环境变量 + InfluxDB 连接配置

2. **实时预测管道**
   - RTM: 每 5 分钟自动预测
   - DAM: 每天 HE10 前出预测
   - Spike: 持续监控, push alert
   - Wind/Load: 每小时更新
   - 写回 InfluxDB 供 viewer 消费

3. **预测 vs 实际追踪**
   - 自动对比预测与实际价格
   - 模型准确率仪表板
   - 漂移检测 + 重训触发

### Sprint 6: 商业化 MVP (Priority: P1)

**目标**: 可交付的信号服务

1. **矿场智能调度**
   - 开停机信号 API
   - DAM 购电量推荐
   - Telegram/Email alert

2. **BESS 套利信号**
   - 充放电时间表推荐
   - 每日 PnL 追踪
   - 历史回测报告

3. **客户 Dashboard**
   - 多租户隔离
   - 自定义 settlement point
   - API key 管理

---

## 执行策略

- 每个任务: CC 写代码 → Codex review → 迭代 → commit
- `.dev-state.json` 跟踪进度
- 最大化并行: 不相互依赖的任务同时派 CC
- 每完成一个 Sprint 给 Lanxin 汇报
