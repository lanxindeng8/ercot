# ERCOT API Rate Limit — 经验文档

**最后更新**: 2026-03-23  
**适用**: ERCOT Public API (`api.ercot.com`)

## 账号信息

- 只有一套 ERCOT API 凭证（truetest86@gmail.com）
- 所有 endpoint、所有 job 共享同一个 `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY` 的 rate limit 配额
- 凭证存储在 LaunchAgent plist `EnvironmentVariables` 中，不在 `.env` 文件里

## Rate Limit 行为观察

### 触发规律
- **连续 3-5 个请求后触发 429**，无论请求间隔（即使间隔 10-15 秒）
- 429 触发后，**冷却时间至少 2-5 分钟**，60 秒退避通常不够
- ERCOT API **不返回 `Retry-After` header**，只能靠经验估计冷却时间
- 所有 endpoint 共享配额：rtm-lmp、dam-lmp、wind-forecast、archive 下载都算在内

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
- **小批量（≤5 请求）**: 直接跑脚本，间隔 90 秒
- **中批量（5-20 请求）**: 用 OpenClaw cron 调度，每 10-15 分钟触发一个请求
- **大批量（>20 请求）**: 分多天完成，或找 archive 替代方案

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

**补全方案**:
1. 检查 NP4-732-CD 是否有 archive 格式（一次下载整年）
2. 如果没有，用 cron 滴灌：每 15 分钟取一个月，7 个月 ≈ 2 小时完成
3. 2026-01~03 如果 API 没有数据（太新），可能需要等 ERCOT 发布

## 未来优化方向

- [ ] 获取第二个 subscription key（如果 ERCOT 允许多 key）
- [ ] 实现请求配额池：所有 job 共享一个令牌桶，防并发超限
- [ ] 对 NP4-732-CD 检查 archive endpoint 是否可用
- [ ] RTM scraper 降频 + 加退避
