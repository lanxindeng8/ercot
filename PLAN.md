# TrueFlux — 项目路线图

> ERCOT 电力市场预测 + 智能调度平台
> 最后更新: 2026-03-18

---

## 总览

TrueFlux 从 ERCOT 实时/DAM 数据出发，训练 7 个 ML 模型预测价格、风电、负荷和价格尖峰，并据此生成矿场开停机信号和电池套利策略。最终目标：SaaS 信号服务。

## 完成进度

| Sprint | 内容 | 状态 | 日期 |
|--------|------|------|------|
| 1 | 基础设施：数据审计、Gap 回填、训练管道 | ✅ 完成 | 03-17 |
| 2 | 模型训练：DAM v2, RTM, Delta-Spread, Spike + 质量修复 | ✅ 完成 | 03-17 ~ 03-18 |
| 3 | 产品化：Prediction API v3→v4, Dashboard | ✅ 完成 | 03-18 |
| 4 | Wind/Load/BESS 模型 + 全栈集成 | ✅ 完成 | 03-18 |
| 5 | 部署：LaunchAgent, SQLite 迁移, 准确率追踪 | ✅ 完成 | 03-18 |
| 6 | 商业化 MVP：矿场调度, BESS 套利, 客户 Dashboard | ✅ 完成 | 03-18 |
| 7.1 | API Key 管理 + Rate Limiting | ✅ 完成 | 03-18 |

## 当前 Sprint: 7 — 多租户 + 部署

### 7.1 API Key 管理 ✅
- SQLite key 存储, 三级 tier (free/pro/enterprise)
- Rate limiting middleware
- 管理 CLI (`manage_keys.py`)
- Admin 端点 (`/admin/keys/*`)

### 7.2 多租户隔离（待启动）
- [ ] 按 API key 隔离数据视图
- [ ] 每客户 settlement point 配置
- [ ] 用量追踪 + 计费基础

### 7.3 生产部署
- [ ] HTTPS (nginx reverse proxy / Caddy)
- [ ] 域名 + SSL
- [ ] Dashboard 公网访问
- [ ] 监控告警 (uptime + model drift)

---

## 未来 Sprint

### Sprint 8: 模型增强
- 特征扩展 (ancillary services, congestion, fuel mix breakdown → 80+ features)
- 自动重训管道 (drift 检测 → 触发)
- 模型 A/B 测试框架

### Sprint 9: 客户获取
- Landing page
- 试用 tier（免费 100 calls/day）
- Stripe 接入
- API 文档 (Swagger/Redoc 已有基础)

### Sprint 10: 规模化
- 多节点预测 (全 ERCOT settlement points)
- 实时 WebSocket 推送
- 历史回测 API
- 移动端告警 (push notification)

---

## 关键指标

| 模型 | 核心指标 | 值 |
|------|---------|-----|
| DAM v2 | MAE (HB_WEST) | 5.88 |
| RTM 1h | MAE | 12.1 |
| Delta-Spread | Sharpe | 34.86 |
| Spike | F1 | 0.939 |
| Wind | MAE | 14.3 MW |
| Load | MAPE | 0.81% |
| BESS | 策略类型 | LP 优化器 |

## 技术债

- [ ] `prediction/ROADMAP.md` 过时 — 已被本文件替代
- [ ] `README.md` 仍描述旧 InfluxDB-only 架构
- [ ] main.py 1700+ 行，需拆分 router
- [ ] 测试覆盖率可提升 (当前 160 tests)
- [ ] 旧 SQLite DB 缺 UNIQUE 约束，新建的有
