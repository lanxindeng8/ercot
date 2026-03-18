"use client";

import React, { useState } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { asNumber, asString, fetchPredictionJson, isRecord } from "@/components/predictions";

interface WindPrediction {
  hour_ending: string;
  predicted_mw: number;
  lower_bound_mw: number;
  upper_bound_mw: number;
}

interface WindResponse {
  status: string;
  model: string;
  generated_at: string;
  predictions: WindPrediction[];
}

interface ChartPoint {
  hour: number;
  hour_label: string;
  predicted: number;
  lower: number;
  upper: number;
  bandBase: number;
  bandSize: number;
}

interface WindForecastChartProps {
  refreshKey: number;
}

function isWindResponse(value: unknown): value is WindResponse {
  if (!isRecord(value)) return false;
  if (asString(value.model) === null) return false;
  if (!Array.isArray(value.predictions)) return false;

  return value.predictions.every((p) => {
    if (!isRecord(p)) return false;
    return (
      asString(p.hour_ending) !== null &&
      asNumber(p.predicted_mw) !== null &&
      asNumber(p.lower_bound_mw) !== null &&
      asNumber(p.upper_bound_mw) !== null
    );
  });
}

export default function WindForecastChart({ refreshKey }: WindForecastChartProps) {
  const [data, setData] = useState<ChartPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>("/api/predictions/wind", {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!isWindResponse(result)) {
          throw new Error("Wind response was malformed");
        }

        const points: ChartPoint[] = result.predictions.map((p, i) => ({
          hour: i + 1,
          hour_label: p.hour_ending,
          predicted: p.predicted_mw,
          lower: p.lower_bound_mw,
          upper: p.upper_bound_mw,
          bandBase: p.lower_bound_mw,
          bandSize: Math.max(p.upper_bound_mw - p.lower_bound_mw, 0),
        }));

        setData(points);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load wind forecast");
          setData(null);
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
        <h3>Wind Generation Forecast</h3>
        <span className="dash-subtitle">24-Hour Ahead | Q10/Q50/Q90</span>
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading wind forecast...</div>}

      {data && (
        <div className="dash-chart-wrapper">
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
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
                label={{ value: "MW", angle: -90, position: "insideLeft", fill: "#888" }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                labelFormatter={(h) => `Hour Ending ${h}:00`}
                formatter={(value) => [`${Number(value).toFixed(0)} MW`, ""]}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="bandBase"
                name="Q10 Base"
                stackId="wind-band"
                stroke="none"
                fillOpacity={0}
                legendType="none"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="bandSize"
                name="Uncertainty Band"
                stackId="wind-band"
                stroke="none"
                fill="#00b894"
                fillOpacity={0.18}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                name="Forecast (Q50)"
                stroke="#00b894"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="upper"
                name="Upper (Q90)"
                stroke="#00b89466"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="lower"
                name="Lower (Q10)"
                stroke="#00b89466"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
