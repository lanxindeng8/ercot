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
  Cell,
} from "recharts";
import { asNumber, asString, fetchPredictionJson, isRecord } from "@/components/predictions";

interface LoadPrediction {
  hour_ending: string;
  predicted_load_mw: number;
}

interface LoadResponse {
  status: string;
  model: string;
  generated_at: string;
  predictions: LoadPrediction[];
}

interface ChartPoint {
  hour: number;
  hour_label: string;
  load_mw: number;
}

interface LoadForecastChartProps {
  refreshKey: number;
}

function isLoadResponse(value: unknown): value is LoadResponse {
  if (!isRecord(value)) return false;
  if (asString(value.model) === null) return false;
  if (!Array.isArray(value.predictions)) return false;

  return value.predictions.every((p) => {
    if (!isRecord(p)) return false;
    return asString(p.hour_ending) !== null && asNumber(p.predicted_load_mw) !== null;
  });
}

const getBarColor = (load: number, max: number): string => {
  const ratio = load / max;
  if (ratio >= 0.9) return "#d63031";
  if (ratio >= 0.75) return "#e17055";
  if (ratio >= 0.5) return "#ffd93d";
  return "#6c5ce7";
};

export default function LoadForecastChart({ refreshKey }: LoadForecastChartProps) {
  const [data, setData] = useState<ChartPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>("/api/predictions/load", {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!isLoadResponse(result)) {
          throw new Error("Load response was malformed");
        }

        const points: ChartPoint[] = result.predictions.map((p, i) => ({
          hour: i + 1,
          hour_label: p.hour_ending,
          load_mw: p.predicted_load_mw,
        }));

        setData(points);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load demand forecast");
          setData(null);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchData();
    return () => controller.abort();
  }, [refreshKey]);

  const maxLoad = data ? Math.max(...data.map((d) => d.load_mw)) : 1;

  // Summary stats
  const peakLoad = data ? Math.max(...data.map((d) => d.load_mw)) : 0;
  const minLoad = data ? Math.min(...data.map((d) => d.load_mw)) : 0;
  const avgLoad = data ? data.reduce((sum, d) => sum + d.load_mw, 0) / data.length : 0;
  const peakHour = data ? data.find((d) => d.load_mw === peakLoad)?.hour ?? 0 : 0;

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>System Load Forecast</h3>
        <span className="dash-subtitle">24-Hour Demand</span>
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading demand forecast...</div>}

      {data && (
        <>
          <div className="load-summary">
            <div className="load-stat">
              <span className="load-stat-label">Peak</span>
              <span className="load-stat-value">{(peakLoad / 1000).toFixed(1)} GW</span>
              <span className="load-stat-detail">HE {peakHour}:00</span>
            </div>
            <div className="load-stat">
              <span className="load-stat-label">Min</span>
              <span className="load-stat-value">{(minLoad / 1000).toFixed(1)} GW</span>
            </div>
            <div className="load-stat">
              <span className="load-stat-label">Average</span>
              <span className="load-stat-value">{(avgLoad / 1000).toFixed(1)} GW</span>
            </div>
          </div>

          <div className="dash-chart-wrapper">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
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
                  tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  labelFormatter={(h) => `Hour Ending ${h}:00`}
                  formatter={(value) => [`${Number(value).toLocaleString()} MW`, "Load"]}
                />
                <Bar dataKey="load_mw" name="Predicted Load" radius={[2, 2, 0, 0]}>
                  {data.map((entry, index) => (
                    <Cell key={index} fill={getBarColor(entry.load_mw, maxLoad)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
