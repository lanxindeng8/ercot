"use client";

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface RtmPrediction {
  horizon: string;
  hours_ahead: number;
  predicted_price: number;
}

interface RtmResponse {
  status: string;
  model: string;
  settlement_point: string;
  generated_at: string;
  predictions: RtmPrediction[];
}

const HORIZON_COLORS: Record<string, string> = {
  "1h": "#4ecdc4",
  "4h": "#ffd93d",
  "24h": "#ff6b6b",
};

interface RtmHorizonChartProps {
  refreshKey: number;
}

export default function RtmHorizonChart({ refreshKey }: RtmHorizonChartProps) {
  const [data, setData] = useState<RtmPrediction[] | null>(null);
  const [meta, setMeta] = useState<{ model: string; point: string; time: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("/api/predictions/rtm?horizons=1h,4h,24h");
        if (!response.ok) throw new Error("Failed to fetch RTM predictions");
        const result: RtmResponse = await response.json();
        if (cancelled) return;
        setData(result.predictions);
        setMeta({
          model: result.model,
          point: result.settlement_point,
          time: result.generated_at,
        });
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load RTM predictions");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchData();
    return () => { cancelled = true; };
  }, [refreshKey]);

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>RTM Multi-Horizon Forecast</h3>
        {meta && <span className="dash-subtitle">{meta.point}</span>}
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading RTM predictions...</div>}

      {data && (
        <>
          <div className="dash-chart-wrapper">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="horizon"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 12 }}
                  tickFormatter={(h) => `${h} ahead`}
                />
                <YAxis
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 11 }}
                  label={{ value: "$/MWh", angle: -90, position: "insideLeft", fill: "#888" }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  formatter={(value) => [`$${Number(value).toFixed(2)}`, "Predicted Price"]}
                  labelFormatter={(h) => `${h} Ahead`}
                />
                <Legend />
                <Bar dataKey="predicted_price" name="Predicted Price" radius={[6, 6, 0, 0]}>
                  {data.map((entry) => (
                    <Cell
                      key={entry.horizon}
                      fill={HORIZON_COLORS[entry.horizon] || "#4ecdc4"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rtm-metrics">
            {data.map((pred) => (
              <div key={pred.horizon} className="rtm-metric-card" style={{ borderLeftColor: HORIZON_COLORS[pred.horizon] }}>
                <div className="rtm-metric-horizon">{pred.horizon} Ahead</div>
                <div className="rtm-metric-price">${pred.predicted_price.toFixed(2)}</div>
                <div className="rtm-metric-label">predicted $/MWh</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
