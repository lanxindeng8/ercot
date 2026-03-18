"use client";

import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const SETTLEMENT_POINTS = ["hb_west", "hb_north", "hb_south", "hb_houston", "hb_busavg"];
const SP_COLORS: Record<string, string> = {
  hb_west: "#4ecdc4",
  hb_north: "#ff6b6b",
  hb_south: "#ffd93d",
  hb_houston: "#6c5ce7",
  hb_busavg: "#a8e6cf",
};

interface DamPrediction {
  hour_ending: string;
  predicted_price: number;
}

interface DamResponse {
  status: string;
  model: string;
  settlement_point: string;
  delivery_date: string;
  generated_at: string;
  predictions: DamPrediction[];
}

interface ChartPoint {
  hour: number;
  hour_label: string;
  [key: string]: number | string;
}

interface DamForecastChartProps {
  refreshKey: number;
}

export default function DamForecastChart({ refreshKey }: DamForecastChartProps) {
  const [data, setData] = useState<ChartPoint[] | null>(null);
  const [deliveryDate, setDeliveryDate] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPoints, setSelectedPoints] = useState<Set<string>>(
    new Set(SETTLEMENT_POINTS)
  );

  React.useEffect(() => {
    let cancelled = false;

    async function fetchAll() {
      setLoading(true);
      setError(null);
      try {
        const responses = await Promise.all(
          SETTLEMENT_POINTS.map((sp) =>
            fetch(`/api/predictions/dam/next-day?settlement_point=${sp}`).then(
              (r) => {
                if (!r.ok) throw new Error(`DAM fetch failed for ${sp}`);
                return r.json() as Promise<DamResponse>;
              }
            )
          )
        );

        if (cancelled) return;

        const merged: ChartPoint[] = [];
        for (let i = 0; i < 24; i++) {
          const point: ChartPoint = {
            hour: i + 1,
            hour_label: responses[0].predictions[i]?.hour_ending || `${i + 1}:00`,
          };
          for (const resp of responses) {
            const sp = resp.settlement_point.toLowerCase();
            point[sp] = resp.predictions[i]?.predicted_price ?? 0;
          }
          merged.push(point);
        }

        setData(merged);
        setDeliveryDate(responses[0].delivery_date);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load DAM forecasts");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchAll();
    return () => { cancelled = true; };
  }, [refreshKey]);

  const togglePoint = (sp: string) => {
    setSelectedPoints((prev) => {
      const next = new Set(prev);
      if (next.has(sp)) next.delete(sp);
      else next.add(sp);
      return next;
    });
  };

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>DAM 24-Hour Forecast</h3>
        {deliveryDate && <span className="dash-subtitle">Delivery: {deliveryDate}</span>}
      </div>

      <div className="sp-toggles">
        {SETTLEMENT_POINTS.map((sp) => (
          <button
            key={sp}
            className={`sp-toggle ${selectedPoints.has(sp) ? "active" : ""}`}
            style={{
              borderColor: SP_COLORS[sp],
              backgroundColor: selectedPoints.has(sp) ? SP_COLORS[sp] : "transparent",
              color: selectedPoints.has(sp) ? "#1a1a2e" : SP_COLORS[sp],
            }}
            onClick={() => togglePoint(sp)}
          >
            {sp.toUpperCase().replace("_", " ")}
          </button>
        ))}
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading DAM forecasts...</div>}

      {data && (
        <div className="dash-chart-wrapper">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
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
                label={{ value: "$/MWh", angle: -90, position: "insideLeft", fill: "#888" }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                labelFormatter={(h) => `Hour Ending ${h}:00`}
                formatter={(value) => [`$${Number(value).toFixed(2)}`, ""]}
              />
              <Legend />
              {SETTLEMENT_POINTS.filter((sp) => selectedPoints.has(sp)).map((sp) => (
                <Line
                  key={sp}
                  type="monotone"
                  dataKey={sp}
                  name={sp.toUpperCase().replace("_", " ")}
                  stroke={SP_COLORS[sp]}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
