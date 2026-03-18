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
} from "recharts";
import { asNumber, asString, fetchPredictionJson, isRecord } from "@/components/predictions";

interface BessEntry {
  hour_ending: string;
  action: string;
  power_mw: number;
  soc_pct: number;
  dam_price: number;
}

interface BessOptimization {
  status: string;
  total_revenue: number;
  solve_time_s: number;
  battery_config: Record<string, number>;
}

interface BessResponse {
  status: string;
  model: string;
  settlement_point: string;
  generated_at: string;
  optimization: BessOptimization;
  schedule: BessEntry[];
}

interface ChartPoint {
  hour: number;
  hour_label: string;
  power_mw: number;
  soc_pct: number;
  dam_price: number;
  action: string;
}

interface BessSchedulePanelProps {
  refreshKey: number;
}

function isBessResponse(value: unknown): value is BessResponse {
  if (!isRecord(value)) return false;
  if (asString(value.model) === null || !isRecord(value.optimization)) return false;
  if (!Array.isArray(value.schedule)) return false;
  if (asNumber(value.optimization.total_revenue) === null) return false;

  return value.schedule.every((e) => {
    if (!isRecord(e)) return false;
    return (
      asString(e.hour_ending) !== null &&
      asString(e.action) !== null &&
      asNumber(e.power_mw) !== null &&
      asNumber(e.soc_pct) !== null &&
      asNumber(e.dam_price) !== null
    );
  });
}

const ACTION_COLORS: Record<string, string> = {
  charge: "#6c5ce7",
  discharge: "#00b894",
  idle: "#636e72",
};

export default function BessSchedulePanel({ refreshKey }: BessSchedulePanelProps) {
  const [data, setData] = useState<BessResponse | null>(null);
  const [chartData, setChartData] = useState<ChartPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>("/api/predictions/bess", {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!isBessResponse(result)) {
          throw new Error("BESS response was malformed");
        }

        setData(result);

        const points: ChartPoint[] = result.schedule.map((e, i) => ({
          hour: i + 1,
          hour_label: e.hour_ending,
          power_mw: e.power_mw,
          soc_pct: e.soc_pct,
          dam_price: e.dam_price,
          action: e.action,
        }));
        setChartData(points);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load BESS schedule");
          setData(null);
          setChartData(null);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchData();
    return () => controller.abort();
  }, [refreshKey]);

  const chargeHours = chartData?.filter((p) => p.action === "charge").length ?? 0;
  const dischargeHours = chartData?.filter((p) => p.action === "discharge").length ?? 0;

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>BESS Schedule</h3>
        {data && (
          <span className="dash-subtitle">
            {data.settlement_point} | {data.optimization.status}
          </span>
        )}
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading BESS schedule...</div>}

      {data && chartData && (
        <>
          <div className="bess-summary">
            <div className="bess-stat">
              <span className="bess-stat-label">Est. Revenue</span>
              <span className="bess-stat-value bess-revenue">
                ${data.optimization.total_revenue.toFixed(2)}
              </span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Charge Hours</span>
              <span className="bess-stat-value">{chargeHours}</span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Discharge Hours</span>
              <span className="bess-stat-value">{dischargeHours}</span>
            </div>
            <div className="bess-stat">
              <span className="bess-stat-label">Battery</span>
              <span className="bess-stat-value">
                {data.optimization.battery_config.E_max_mwh} MWh / {data.optimization.battery_config.P_max_mw} MW
              </span>
            </div>
          </div>

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
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 11 }}
                  domain={[0, 100]}
                  label={{ value: "SoC %", angle: 90, position: "insideRight", fill: "#888" }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  labelFormatter={(h) => `Hour Ending ${h}:00`}
                  formatter={(value, name) => {
                    const v = Number(value);
                    if (name === "Power") return [`${v.toFixed(3)} MW`, ""];
                    if (name === "SoC") return [`${v.toFixed(1)}%`, ""];
                    if (name === "DAM Price") return [`$${v.toFixed(2)}`, ""];
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
                  yAxisId="power"
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
          </div>
        </>
      )}
    </div>
  );
}
