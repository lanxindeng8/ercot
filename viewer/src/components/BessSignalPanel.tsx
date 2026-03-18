"use client";

import React, { useState } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  BarChart,
  Area,
  AreaChart,
} from "recharts";
import { fetchPredictionJson, isRecord, asNumber, asString } from "@/components/predictions";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface HourSignal {
  hour_ending: string;
  action: string;
  power_mw: number;
  soc_pct: number;
  dam_price: number;
  rtm_volatility: number;
  spike_probability: number;
  revenue_estimate: number;
  risk_flag: string;
  mining_curtail: boolean;
}

interface SignalSummary {
  total_revenue_estimate: number;
  risk_adjusted_revenue: number;
  charge_hours: number;
  discharge_hours: number;
  idle_hours: number;
  peak_discharge_price: number;
  avg_charge_price: number;
  spike_hold_hours: number[];
}

interface SignalResponse {
  status: string;
  settlement_point: string;
  date: string;
  generated_at: string;
  summary: SignalSummary;
  signals: HourSignal[];
}

interface PnLDay {
  date: string;
  projected_revenue: number;
  actual_revenue: number | null;
  charge_cost: number;
  discharge_revenue: number;
  cycles: number;
  degradation_cost: number;
  net_pnl: number;
}

interface PnLResponse {
  status: string;
  summary: {
    total_pnl: number;
    total_cycles: number;
    avg_daily_pnl: number;
  };
  daily: PnLDay[];
}

interface RiskResponse {
  status: string;
  risk: {
    days: number;
    total_pnl: number;
    avg_daily_pnl: number;
    var_95: number;
    max_drawdown: number;
    win_rate: number;
    sharpe_ratio: number;
    best_day: number;
    worst_day: number;
    volatility: number;
  };
}

interface ChartPoint {
  hour: number;
  hour_label: string;
  power_mw: number;
  soc_pct: number;
  dam_price: number;
  revenue: number;
  action: string;
  spike_prob: number;
  risk_flag: string;
  mining_curtail: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ACTION_COLORS: Record<string, string> = {
  charge: "#6c5ce7",
  discharge: "#00b894",
  idle: "#636e72",
};

const RISK_BADGES: Record<string, { label: string; color: string }> = {
  normal: { label: "Normal", color: "#00b894" },
  spike_hold: { label: "Spike Hold", color: "#e17055" },
  high_volatility: { label: "High Vol", color: "#ffd93d" },
};

function isSignalResponse(v: unknown): v is SignalResponse {
  if (!isRecord(v)) return false;
  if (asString(v.status) !== "success") return false;
  if (!isRecord(v.summary) || !Array.isArray(v.signals)) return false;
  return asNumber(v.summary.total_revenue_estimate) !== null;
}

function isPnLResponse(v: unknown): v is PnLResponse {
  if (!isRecord(v)) return false;
  if (asString(v.status) !== "success") return false;
  return isRecord(v.summary) && Array.isArray(v.daily);
}

function isRiskResponse(v: unknown): v is RiskResponse {
  if (!isRecord(v)) return false;
  if (asString(v.status) !== "success") return false;
  return isRecord(v.risk) && asNumber(v.risk.var_95) !== null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface BessSignalPanelProps {
  refreshKey: number;
}

export default function BessSignalPanel({ refreshKey }: BessSignalPanelProps) {
  const [signals, setSignals] = useState<SignalResponse | null>(null);
  const [pnl, setPnl] = useState<PnLResponse | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [chartData, setChartData] = useState<ChartPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchAll() {
      setLoading(true);
      setError(null);
      try {
        const [sigRes, pnlRes, riskRes] = await Promise.all([
          fetchPredictionJson<unknown>("/api/dispatch/bess/daily-signals", {
            signal: controller.signal,
          }),
          fetchPredictionJson<unknown>("/api/dispatch/bess/pnl?days=7", {
            signal: controller.signal,
          }).catch(() => null),
          fetchPredictionJson<unknown>("/api/dispatch/bess/risk?days=30", {
            signal: controller.signal,
          }).catch(() => null),
        ]);
        if (controller.signal.aborted) return;

        if (!isSignalResponse(sigRes)) {
          throw new Error("BESS signal response was malformed");
        }
        setSignals(sigRes);

        const points: ChartPoint[] = sigRes.signals.map((s, i) => ({
          hour: i + 1,
          hour_label: s.hour_ending,
          power_mw: s.action === "charge" ? -s.power_mw : s.power_mw,
          soc_pct: s.soc_pct,
          dam_price: s.dam_price,
          revenue: s.revenue_estimate,
          action: s.action,
          spike_prob: s.spike_probability,
          risk_flag: s.risk_flag,
          mining_curtail: s.mining_curtail,
        }));
        setChartData(points);

        if (isPnLResponse(pnlRes)) setPnl(pnlRes);
        if (isRiskResponse(riskRes)) setRisk(riskRes);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load BESS signals");
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchAll();
    return () => controller.abort();
  }, [refreshKey]);

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>BESS Arbitrage Signals</h3>
        {signals && (
          <span className="dash-subtitle">
            {signals.settlement_point} | {signals.date}
          </span>
        )}
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !signals && <div className="dash-loading">Loading BESS signals...</div>}

      {signals && chartData && (
        <>
          {/* Summary stats */}
          <div className="bess-summary">
            <div className="bess-stat">
              <span className="bess-stat-label">Est. Revenue</span>
              <span className="bess-stat-value bess-revenue">
                ${signals.summary.total_revenue_estimate.toFixed(2)}
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Risk-Adj Revenue</span>
              <span className="bess-stat-value">
                ${signals.summary.risk_adjusted_revenue.toFixed(2)}
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Charge / Discharge</span>
              <span className="bess-stat-value">
                {signals.summary.charge_hours}h / {signals.summary.discharge_hours}h
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Spread</span>
              <span className="bess-stat-value">
                ${(signals.summary.peak_discharge_price - signals.summary.avg_charge_price).toFixed(2)}
              </span>
            </div>
            {signals.summary.spike_hold_hours.length > 0 && (
              <div className="bess-stat">
                <span className="bess-stat-label">Spike Holds</span>
                <span className="bess-stat-value" style={{ color: "#e17055" }}>
                  {signals.summary.spike_hold_hours.map((h) => `HE${h}`).join(", ")}
                </span>
              </div>
            )}
          </div>

          {/* Timeline chart */}
          <div className="dash-chart-wrapper" style={{ height: 340 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="hour"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 11 }}
                  tickFormatter={(h) => `${h}:00`}
                  interval={2}
                />
                <YAxis
                  yAxisId="power"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 11 }}
                  label={{ value: "MW", angle: -90, position: "insideLeft", fill: "#888" }}
                />
                <YAxis
                  yAxisId="soc"
                  orientation="right"
                  stroke="#ffd93d"
                  tick={{ fill: "#ffd93d", fontSize: 11 }}
                  domain={[0, 100]}
                  label={{ value: "SoC %", angle: 90, position: "insideRight", fill: "#ffd93d" }}
                />
                <YAxis yAxisId="price" orientation="right" stroke="#ff6b6b" hide />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  labelFormatter={(h) => `Hour Ending ${h}:00`}
                  formatter={(value, name) => {
                    const v = Number(value ?? 0);
                    if (name === "Power") return [`${v.toFixed(3)} MW`, ""];
                    if (name === "SoC") return [`${v.toFixed(1)}%`, ""];
                    if (name === "DAM Price") return [`$${v.toFixed(2)}`, ""];
                    if (name === "Revenue") return [`$${v.toFixed(2)}`, ""];
                    return [v, ""];
                  }}
                />
                <Legend />
                <ReferenceLine yAxisId="power" y={0} stroke="#555" strokeDasharray="3 3" />
                <Bar yAxisId="power" dataKey="power_mw" name="Power" radius={[2, 2, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={index} fill={ACTION_COLORS[entry.action] || ACTION_COLORS.idle} />
                  ))}
                </Bar>
                <Line
                  yAxisId="soc"
                  type="monotone"
                  dataKey="soc_pct"
                  name="SoC"
                  stroke="#ffd93d"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="dam_price"
                  name="DAM Price"
                  stroke="#ff6b6b"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Risk flags legend */}
          <div className="bess-legend">
            <span className="bess-legend-item">
              <span className="bess-legend-dot" style={{ background: "#6c5ce7" }} /> Charge
            </span>
            <span className="bess-legend-item">
              <span className="bess-legend-dot" style={{ background: "#00b894" }} /> Discharge
            </span>
            <span className="bess-legend-item">
              <span className="bess-legend-dot" style={{ background: "#636e72" }} /> Idle
            </span>
            {chartData.some((p) => p.mining_curtail) && (
              <span className="bess-legend-item">
                <span className="bess-legend-dot" style={{ background: "#d63031" }} /> Mining Curtail
              </span>
            )}
          </div>

          {/* Revenue chart */}
          <h4 style={{ color: "#ccc", margin: "20px 0 8px", fontSize: 14 }}>Hourly Revenue</h4>
          <div className="dash-chart-wrapper" style={{ height: 180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="hour"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 10 }}
                  tickFormatter={(h) => `${h}`}
                  interval={2}
                />
                <YAxis
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 10 }}
                  tickFormatter={(v) => `$${v}`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  labelFormatter={(h) => `HE ${h}`}
                  formatter={(v) => [`$${Number(v ?? 0).toFixed(2)}`, "Revenue"]}
                />
                <ReferenceLine y={0} stroke="#555" />
                <Bar dataKey="revenue" name="Revenue" radius={[2, 2, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={index} fill={entry.revenue >= 0 ? "#00b894" : "#d63031"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Risk gauge */}
      {risk && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ color: "#ccc", margin: "0 0 8px", fontSize: 14 }}>Risk Metrics (30d)</h4>
          <div className="bess-summary">
            <div className="bess-stat">
              <span className="bess-stat-label">VaR 95%</span>
              <span className="bess-stat-value" style={{ color: risk.risk.var_95 < 0 ? "#d63031" : "#00b894" }}>
                ${risk.risk.var_95.toFixed(2)}
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Max Drawdown</span>
              <span className="bess-stat-value" style={{ color: "#d63031" }}>
                ${risk.risk.max_drawdown.toFixed(2)}
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Sharpe</span>
              <span className="bess-stat-value">{risk.risk.sharpe_ratio.toFixed(2)}</span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Win Rate</span>
              <span className="bess-stat-value">{(risk.risk.win_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Daily Vol</span>
              <span className="bess-stat-value">${risk.risk.volatility.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}

      {/* PnL rolling */}
      {pnl && pnl.daily.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ color: "#ccc", margin: "0 0 8px", fontSize: 14 }}>
            Rolling PnL (7d) &mdash; Total: ${pnl.summary.total_pnl.toFixed(2)}
          </h4>
          <div className="dash-chart-wrapper" style={{ height: 160 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={pnl.daily.map((d) => ({
                  date: d.date.slice(5),
                  pnl: d.net_pnl,
                  cumulative: 0,
                }))}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="date" stroke="#888" tick={{ fill: "#888", fontSize: 10 }} />
                <YAxis stroke="#888" tick={{ fill: "#888", fontSize: 10 }} tickFormatter={(v) => `$${v}`} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  formatter={(v) => [`$${Number(v ?? 0).toFixed(2)}`, "Net PnL"]}
                />
                <ReferenceLine y={0} stroke="#555" />
                <Area
                  type="monotone"
                  dataKey="pnl"
                  stroke="#6c5ce7"
                  fill="#6c5ce744"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
