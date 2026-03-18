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
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";
import { fetchPredictionJson, isRecord, asNumber, asString } from "@/components/predictions";

interface HourSignal {
  hour_ending: string;
  action: string;
  power_mw: number;
  soc_pct: number;
  dam_price: number;
  revenue_estimate: number;
}

interface SignalSummary {
  total_revenue_estimate: number;
  risk_adjusted_revenue: number;
  charge_hours: number;
  discharge_hours: number;
  idle_hours: number;
  peak_discharge_price: number;
  avg_charge_price: number;
}

interface SignalResponse {
  status: string;
  settlement_point: string;
  date: string;
  summary: SignalSummary;
  signals: HourSignal[];
}

interface ChartPoint {
  hour: number;
  power_mw: number;
  soc_pct: number;
  dam_price: number;
  revenue: number;
  action: string;
}

interface BessArbitrageCardProps {
  refreshKey: number;
}

const ACTION_COLORS: Record<string, string> = {
  charge: "#6c5ce7",
  discharge: "#00b894",
  idle: "#636e72",
};

function isSignalResponse(v: unknown): v is SignalResponse {
  if (!isRecord(v)) return false;
  if (asString(v.status) !== "success") return false;
  if (!isRecord(v.summary) || !Array.isArray(v.signals)) return false;
  return asNumber(v.summary.total_revenue_estimate) !== null;
}

export default function BessArbitrageCard({ refreshKey }: BessArbitrageCardProps) {
  const [data, setData] = useState<SignalResponse | null>(null);
  const [chartData, setChartData] = useState<ChartPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>("/api/dispatch/bess/daily-signals", {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!isSignalResponse(result)) {
          throw new Error("BESS signal response was malformed");
        }
        setData(result);

        const points: ChartPoint[] = result.signals.map((s, i) => ({
          hour: i + 1,
          power_mw: s.action === "charge" ? -s.power_mw : s.power_mw,
          soc_pct: s.soc_pct,
          dam_price: s.dam_price,
          revenue: s.revenue_estimate,
          action: s.action,
        }));
        setChartData(points);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load BESS signals");
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchData();
    return () => controller.abort();
  }, [refreshKey]);

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>BESS Arbitrage</h3>
        {data && <span className="dash-subtitle">{data.settlement_point} | {data.date}</span>}
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading BESS schedule...</div>}

      {data && chartData && (
        <>
          {/* Summary stats */}
          <div className="bess-summary">
            <div className="bess-stat">
              <span className="bess-stat-label">Expected Revenue</span>
              <span className="bess-stat-value bess-revenue">
                ${data.summary.total_revenue_estimate.toFixed(2)}
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Risk-Adjusted</span>
              <span className="bess-stat-value">
                ${data.summary.risk_adjusted_revenue.toFixed(2)}
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Charge / Discharge</span>
              <span className="bess-stat-value">
                {data.summary.charge_hours}h / {data.summary.discharge_hours}h
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Price Spread</span>
              <span className="bess-stat-value">
                ${(data.summary.peak_discharge_price - data.summary.avg_charge_price).toFixed(2)}
              </span>
            </div>
          </div>

          {/* Timeline chart */}
          <div className="dash-chart-wrapper" style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="hour"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 10 }}
                  tickFormatter={(h) => `${h}`}
                  interval={2}
                />
                <YAxis
                  yAxisId="power"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 10 }}
                  label={{ value: "MW", angle: -90, position: "insideLeft", fill: "#888", fontSize: 10 }}
                />
                <YAxis
                  yAxisId="soc"
                  orientation="right"
                  stroke="#ffd93d"
                  tick={{ fill: "#ffd93d", fontSize: 10 }}
                  domain={[0, 100]}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  labelFormatter={(h) => `HE ${h}`}
                  formatter={(value, name) => {
                    const v = Number(value ?? 0);
                    if (name === "Power") return [`${v.toFixed(3)} MW`, ""];
                    if (name === "SoC") return [`${v.toFixed(1)}%`, ""];
                    return [v, ""];
                  }}
                />
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
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Legend */}
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
            <span className="bess-legend-item">
              <span className="bess-legend-dot" style={{ background: "#ffd93d" }} /> SoC %
            </span>
          </div>
        </>
      )}
    </div>
  );
}
