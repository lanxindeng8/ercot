# ERCOT API Rate Limit — 经验文档

**最后更新**: 2026-03-23  
**适用**: ERCOT Public API (`api.ercot.com`)

## 账号信息

- 只有一套 ERCOT API 凭证（truetest86@gmail.com）
- 所有 endpoint、所有 job 共享同一个 `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY` 的 rate limit 配额
- 凭证存储在 LaunchAgent plist `EnvironmentVariables` 中，不在 `.env` 文件里

## Rate Limit 行为观察

### 限流类型：小时带宽限制
429 响应体明确说明：
```json
{"statusCode": 429, "error": "BandwidthQuotaExceeded",
 "message": "You have exceeded the hourly bandwidth limit. Your quota will reset within the hour."}
```

- **限制的是带宽（数据传输量），不是请求次数**
- 配额 **每小时重置**（整点或滚动窗口）
- 所有 endpoint 共享配额：rtm-lmp、dam-lmp、wind-forecast、archive 下载都算在内
- ERCOT API **不返回 `Retry-After` header**

### 经验数据
- Wind forecast 每月 ~160K 行 (~10MB)，连续 3-5 个月查询后触发
- Archive 下载每文件 ~16MB，连续 4 个可以过（间隔 90s），第 5 个大概率 429
- 429 触发后，同一小时内的后续请求基本都会被拒

### 两类请求的差异
| 类型 | Endpoint 示例 | 特点 |
|---|---|---|
| **Archive 下载** | `np6-792-ER` (reserves) | 一次下载整年 XLSX (~16MB)，总请求数少 |
| **实时查询** | `np4-732-cd` (wind forecast) | 按月分页查询，每月 ~160K 行分 4 页，请求密集 |

Archive 下载更容易成功，因为单文件包含整年数据（4 次请求 = 4 年）。  
实时查询每月需 1-4 次分页请求，12 个月 = 12-48 次请求，极易触发限流。

### 踩过的坑

1. **urllib3 `Retry(status_forcelist=[429])` 是灾难**  
   urllib3 对 429 做快速重试（秒级），反而加速耗尽配额。  
   **修复**: 从 `status_forcelist` 移除 429，只用自定义退避逻辑处理。  
   Commit: `6ffd9f3`

2. **多 job 并发 = 加速撞限流**  
   之前 rtm-scraper + prediction-runner 每 5 分钟同时触发，加上手动 backfill，三路并发秒触 429。  
   **规则**: 任何时候最多一个进程访问 ERCOT API。

3. **Archive 文件命名与内容不匹配**  
   `RTM_ORDC_..._2023.xlsx` 实际包含 2022 年数据（ERCOT 以发布年份命名，不是数据年份）。  
   **规则**: 入库后必须验证 `sced_timestamp` 的实际年份范围。

## 当前退避策略 (`scraper/src/ercot_client.py`)

```
初始延迟: 10 秒
退避方式: 指数 × 2 + jitter (±50%)
最大延迟: 300 秒 (5 分钟)
最大重试: 5 次
序列示例: 10s → 20s → 40s → 80s → 160s (加随机抖动)
```

## 数据获取策略（推荐）

### 原则
1. **串行，不并发** — 同一时间只有一个 ERCOT API 调用者
2. **Archive 优先** — 如果数据有 yearly archive 格式（如 reserves），优先下载 archive
3. **间隔 ≥ 90 秒** — 每次成功请求后等 90 秒再发下一个
4. **失败不急** — 被 429 后至少等 5 分钟再试

### 批量 Backfill
- **Archive 类型（reserves 等）**: 直接跑脚本，间隔 90 秒，每小时 ≤4 个文件
- **查询类型（wind forecast 等）**: 用 cron 每小时触发，顺序拉取直到 429，下次继续
- **大批量（>20 请求）**: 分多天完成，或找 archive 替代方案
- **脚本模式**: 触发限流就立即放弃，不做长时间退避重试（浪费时间，配额要到下一小时才重置）

### LaunchAgent 运行时
- RTM scraper: 每 5 分钟（最高频请求者）
- DAM scraper: 每天 1 次
- Prediction runner: 不直接调 ERCOT API（只调本地 prediction-api）
- **建议**: RTM scraper 降频到每 15 分钟，或与其他 job 错开

## 数据补全现状 (2026-03-23)

### RT Reserves ✅ 完整
| 年份 | 行数 | 状态 |
|---|---|---|
| 2016-2025 | 1,056,444 | 每年 ~106K 行，连续完整 |

### Wind Forecast — 还缺 7 个月
| 有数据 | 缺失 |
|---|---|
| 33 个月 (2022-12 → 2025-11) | `2024-04, 2024-05, 2024-07, 2025-12, 2026-01, 2026-02, 2026-03` |

**补全方案（已实施）**:
- OpenClaw cron job `wind-forecast-backfill`，每 1 小时触发
- 脚本 `prediction/scripts/fetch_wind_one_month.py`：找第一个缺失月 → 拉一个月 → 429 就放弃等下次
- 全部补完后自动删除 cron job
- 预计 7 小时内补完（每小时 1 个月）

## Cron Jobs

### wind-forecast-backfill
- **频率**: 每 1 小时
- **脚本**: `prediction/scripts/fetch_wind_one_month.py`
- **逻辑**: 查 DB 找所有缺失月 → 按顺序逐月拉取 → 429 立即停止等下次 cron → 全部补完后删除 cron
- **每次运行**尽量多拉（在配额允许内），不止拉一个月
- **状态**: 运行中（2026-03-23 创建）

## 未来优化方向

- [ ] 获取第二个 subscription key（如果 ERCOT 允许多 key）
- [ ] 实现请求配额池：所有 job 共享一个令牌桶，防并发超限
- [ ] 检查 NP4-732-CD 是否有 archive endpoint（待 429 配额恢复后查）
- [ ] RTM scraper 降频到 15 分钟 + 加退避
- [ ] 退避策略调整：429 后不做长退避，直接放弃等下个小时（匹配 hourly quota reset）
