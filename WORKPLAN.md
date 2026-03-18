# ERCOT 开发工作计划

## 项目现状摘要

| 组件 | 状态 | 备注 |
|------|------|------|
| **Scraper** | ✅ 运行中 | RTM 实时到 2026-03-18, DAM 到 03-19, LaunchAgent 自动 |
| **SQLite 数据** | ✅ 丰富 | RTM 1070万条, DAM 117万+142万条, 燃料混合537万条 |
| **RTM 模型** | ⚠️ 基础版 | 15分钟预测，需要重训 |
| **DAM v2 模型** | ⚠️ 训练完但空目录 | 只有 metrics CSV，代码在 prediction/src/ |
| **Delta-Spread** | ⚠️ 有代码无部署 | 套利信号，backtesting 存在 |
| **Spike** | ⚠️ 设计完整 | 文档好，需要训练+集成 |
| **Wind** | ⚠️ 完整框架 | HRRR 集成，多模型，需训练 |
| **Load** | 🔴 仅 Jupyter | 5个 notebook，未产品化 |
| **Battery-Strategy** | ✅ 可用 | LP 优化器 + 测试 |
| **Prediction API** | ⚠️ 基础 | FastAPI，只接了 delta-spread 和 dam_v2 |
| **Viewer** | ⚠️ 基础 | Next.js + Recharts + InfluxDB, Vercel 部署 |

## 数据时间范围
- RTM/DAM 历史: 2015-01-01 ~ 2026-02-14 (11年)
- RTM CDR 实时: 持续更新到当前
- DAM 实时: 持续更新，含次日预测

---

## Sprint 计划

### Sprint 1: 基础设施加固 (Priority: P0)

**目标**: 数据管道可靠性 + 模型训练基础

1. **数据管道审计 + 修复**
   - 历史数据 gap 检测 (2026-02-14 到现在有 gap)
   - 回填 2026-02-15 ~ 今天的历史数据
   - 统一数据 schema (rtm_lmp_hist vs rtm_lmp_cdr 格式不一致)
   - 添加数据质量监控 (anomaly/null/duplicate 检测)

2. **训练数据管道**
   - 建立统一的 feature engineering pipeline
   - SQLite → Parquet 导出脚本 (大数据训练用)
   - 训练/验证/测试集标准化分割

3. **测试框架**
   - pytest 基础设施
   - 数据质量测试
   - 模型 smoke tests

### Sprint 2: 模型重训 + 评估 (Priority: P0)

**目标**: 所有模型用最新数据重训，建立 baseline

1. **DAM v2 重训**
   - 用 2015-2026 全量数据重训
   - 评估 MAPE/RMSE/方向准确率
   - 对比 naive baseline (昨天价格)

2. **RTM 模型升级**
   - 从 15min 扩展到多 horizon (5min, 1h, 4h)
   - CatBoost/LightGBM 对比
   - 加入实时特征 (当前价格、trending)

3. **Delta-Spread 模型验证**
   - 用最新数据 backtest
   - 计算真实 PnL (含交易成本)
   - Sharpe ratio / max drawdown 分析

4. **Spike 模型训练**
   - 完成数据准备 pipeline
   - 训练 spike 分类器
   - 评估 precision/recall tradeoff

### Sprint 3: 产品化 (Priority: P1)

**目标**: 信号可消费，有 API 和可视化

1. **Prediction API 扩展**
   - 接入所有训练好的模型
   - 添加 /forecast/rtm, /forecast/dam, /signal/spread, /alert/spike
   - 模型版本管理
   - 健康检查 + 指标

2. **实时预测管道**
   - RTM: 每 5 分钟自动预测
   - DAM: 每天 HE10 前出预测
   - Spike: 持续监控，push alert
   - 写回 InfluxDB 供 viewer 消费

3. **Viewer 升级**
   - 预测 vs 实际对比图
   - Spread 信号仪表板
   - Spike 预警面板
   - 模型准确率追踪

### Sprint 4: BESS 策略优化 (Priority: P1)

**目标**: 储能调度决策引擎

1. **Battery-Strategy 集成实时数据**
   - 接入 DAM 预测 (而非历史价格)
   - 加入 RTM 不确定性
   - Spike 事件 hold-SOC 策略

2. **Wind 模型训练**
   - HRRR 数据获取自动化
   - 训练风电出力预测
   - 风电-价格关联分析

3. **Load 模型产品化**
   - Notebook → 可训练脚本
   - 集成到 Prediction API

### Sprint 5: 商业化准备 (Priority: P2)

**目标**: MVP 产品

1. **矿场智能调度 MVP**
   - 开停机信号 API
   - DAM 购电量推荐
   - Telegram/Email alert

2. **BESS 套利信号服务**
   - 充放电时间表推荐
   - 每日 PnL 追踪
   - 历史回测报告

---

## 执行策略

- 每个任务: CC 写代码 → Codex review → 迭代 → commit
- `.dev-state.json` 跟踪进度
- 每完成一个 Sprint 任务给 Lanxin 汇报
- 当前启动: **Sprint 1, Task 1 (数据管道审计)**
