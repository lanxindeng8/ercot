"use client";

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { asNumber, asString, fetchPredictionJson, isRecord } from "@/components/predictions";

interface AccuracyMetrics {
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  directional_accuracy: number | null;
  count: number;
}

interface HourlyMetrics {
  mae: number;
  rmse: number;
  mape: number | null;
  count: number;
}

interface Comparison {
  target_time: string;
  settlement_point: string | null;
  predicted: number;
  actual: number;
  error: number;
  abs_error: number;
  pct_error: number | null;
}

interface AccuracyResponse {
  status: string;
  model: string;
  days: number;
  metrics: AccuracyMetrics;
  hourly: Record<string, HourlyMetrics>;
  recent_comparisons: Comparison[];
}

interface HourlyChartPoint {
  hour: number;
  hour_label: string;
  mae: number;
  count: number;
}

interface AccuracyPanelProps {
  refreshKey: number;
}

function isAccuracyResponse(value: unknown): value is AccuracyResponse {
  if (!isRecord(value)) return false;
  if (asString(value.status) === null) return false;
  if (asString(value.model) === null) return false;
  if (!isRecord(value.metrics)) return false;
  if (asNumber(value.metrics.count) === null) return false;
  return true;
}

const MODEL_COLORS: Record<string, string> = {
  dam: "#4ecdc4",
  rtm: "#ff6b6b",
};

export default function AccuracyPanel({ refreshKey }: AccuracyPanelProps) {
  const [selectedModel, setSelectedModel] = useState<"dam" | "rtm">("dam");
  const [days, setDays] = useState(7);
  const [data, setData] = useState<AccuracyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>(
          `/api/accuracy?model=${selectedModel}&days=${days}`,
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        if (!isAccuracyResponse(result)) {
          throw new Error("Accuracy response was malformed");
        }
        setData(result);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load accuracy data");
          setData(null);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchData();
    return () => controller.abort();
  }, [refreshKey, selectedModel, days]);

  const hourlyData: HourlyChartPoint[] = data?.hourly
    ? Object.entries(data.hourly)
        .map(([hour, m]) => ({
          hour: parseInt(hour, 10),
          hour_label: `${parseInt(hour, 10)}:00`,
          mae: m.mae,
          count: m.count,
        }))
        .sort((a, b) => a.hour - b.hour)
    : [];

  const metrics = data?.metrics;

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>Prediction Accuracy</h3>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value as "dam" | "rtm")}
            style={{
              background: "#1e1e3a",
              color: "#ccc",
              border: "1px solid #444",
              borderRadius: 4,
              padding: "4px 8px",
              fontSize: 12,
            }}
          >
            <option value="dam">DAM</option>
            <option value="rtm">RTM</option>
          </select>
          <select
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value, 10))}
            style={{
              background: "#1e1e3a",
              color: "#ccc",
              border: "1px solid #444",
              borderRadius: 4,
              padding: "4px 8px",
              fontSize: 12,
            }}
          >
            <option value={1}>1 day</option>
            <option value={3}>3 days</option>
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
          </select>
        </div>
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading accuracy data...</div>}

      {data && metrics && (
        <>
          {/* Summary metrics */}
          <div className="load-summary">
            <div className="load-stat">
              <span className="load-stat-label">MAE</span>
              <span className="load-stat-value">
                {metrics.mae !== null ? `$${metrics.mae.toFixed(2)}` : "N/A"}
              </span>
              <span className="load-stat-detail">$/MWh</span>
            </div>
            <div className="load-stat">
              <span className="load-stat-label">RMSE</span>
              <span className="load-stat-value">
                {metrics.rmse !== null ? `$${metrics.rmse.toFixed(2)}` : "N/A"}
              </span>
              <span className="load-stat-detail">$/MWh</span>
            </div>
            <div className="load-stat">
              <span className="load-stat-label">MAPE</span>
              <span className="load-stat-value">
                {metrics.mape !== null ? `${metrics.mape.toFixed(1)}%` : "N/A"}
              </span>
            </div>
            <div className="load-stat">
              <span className="load-stat-label">Direction</span>
              <span className="load-stat-value">
                {metrics.directional_accuracy !== null
                  ? `${(metrics.directional_accuracy * 100).toFixed(0)}%`
                  : "N/A"}
              </span>
              <span className="load-stat-detail">{metrics.count} scored</span>
            </div>
          </div>

          {/* MAE per hour bar chart */}
          {hourlyData.length > 0 && (
            <div className="dash-chart-wrapper">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={hourlyData} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    dataKey="hour"
                    stroke="#888"
                    tick={{ fill: "#888", fontSize: 11 }}
                    tickFormatter={(h) => `${h}:00`}
                    interval={2}
                  />
                  <YAxis
                    stroke="#888"
                    tick={{ fill: "#888", fontSize: 11 }}
                    label={{ value: "MAE ($/MWh)", angle: -90, position: "insideLeft", fill: "#888", fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                    labelFormatter={(h) => `Hour ${h}:00`}
                    formatter={(value) => [`$${Number(value ?? 0).toFixed(2)}`, "MAE"]}
                  />
                  <Bar
                    dataKey="mae"
                    name="MAE"
                    fill={MODEL_COLORS[selectedModel] || "#4ecdc4"}
                    radius={[2, 2, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Recent comparisons table */}
          {data.recent_comparisons.length > 0 && (
            <div style={{ overflowX: "auto", marginTop: 12 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #444" }}>
                    <th style={{ padding: "6px 8px", textAlign: "left", color: "#888" }}>Time</th>
                    <th style={{ padding: "6px 8px", textAlign: "left", color: "#888" }}>SP</th>
                    <th style={{ padding: "6px 8px", textAlign: "right", color: "#888" }}>Predicted</th>
                    <th style={{ padding: "6px 8px", textAlign: "right", color: "#888" }}>Actual</th>
                    <th style={{ padding: "6px 8px", textAlign: "right", color: "#888" }}>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent_comparisons.slice(0, 12).map((c, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #333" }}>
                      <td style={{ padding: "4px 8px", color: "#ccc" }}>
                        {c.target_time.replace("T", " ").slice(0, 16)}
                      </td>
                      <td style={{ padding: "4px 8px", color: "#aaa" }}>
                        {c.settlement_point || "—"}
                      </td>
                      <td style={{ padding: "4px 8px", textAlign: "right", color: "#ccc" }}>
                        ${c.predicted.toFixed(2)}
                      </td>
                      <td style={{ padding: "4px 8px", textAlign: "right", color: "#ccc" }}>
                        ${c.actual.toFixed(2)}
                      </td>
                      <td
                        style={{
                          padding: "4px 8px",
                          textAlign: "right",
                          color: c.error > 0 ? "#ff6b6b" : "#4ecdc4",
                          fontWeight: 500,
                        }}
                      >
                        {c.error > 0 ? "+" : ""}
                        ${c.error.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {metrics.count === 0 && (
            <div style={{ textAlign: "center", color: "#888", padding: "24px 0" }}>
              No scored predictions yet. Run <code>score_predictions.py</code> to generate accuracy data.
            </div>
          )}
        </>
      )}
    </div>
  );
}
