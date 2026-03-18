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
  Area,
} from "recharts";
import { fetchPredictionJson, isRecord, asNumber, asString } from "@/components/predictions";

interface DamPoint {
  hour_ending: string;
  predicted_price: number;
}

interface RtmPrediction {
  horizon: string;
  hours_ahead: number;
  predicted_price: number;
}

interface SpikeAlert {
  is_spike: boolean;
  spike_probability: number;
  confidence: string;
}

interface ChartPoint {
  hour: number;
  dam_price: number;
  spike_prob: number;
}

interface PriceForecastCardProps {
  refreshKey: number;
}

export default function PriceForecastCard({ refreshKey }: PriceForecastCardProps) {
  const [chartData, setChartData] = useState<ChartPoint[] | null>(null);
  const [rtmPrices, setRtmPrices] = useState<RtmPrediction[] | null>(null);
  const [spike, setSpike] = useState<SpikeAlert | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchAll() {
      setLoading(true);
      setError(null);
      try {
        const [damRes, rtmRes, spikeRes] = await Promise.all([
          fetchPredictionJson<unknown>("/api/predictions/dam/next-day?settlement_point=HB_WEST", {
            signal: controller.signal,
          }),
          fetchPredictionJson<unknown>("/api/predictions/rtm?horizons=1h,4h,24h", {
            signal: controller.signal,
          }).catch(() => null),
          fetchPredictionJson<unknown>("/api/predictions/spike", {
            signal: controller.signal,
          }).catch(() => null),
        ]);
        if (controller.signal.aborted) return;

        // Parse DAM
        if (!isRecord(damRes) || !Array.isArray(damRes.predictions)) {
          throw new Error("DAM forecast response was malformed");
        }
        const damPreds = damRes.predictions as DamPoint[];

        // Parse spike for overlay
        let spikeProb = 0;
        if (isRecord(spikeRes) && isRecord(spikeRes.alert)) {
          const a = spikeRes.alert;
          if (typeof a.is_spike === "boolean" && asNumber(a.spike_probability) !== null) {
            setSpike({
              is_spike: a.is_spike as boolean,
              spike_probability: a.spike_probability as number,
              confidence: asString(a.confidence) ?? "low",
            });
            spikeProb = a.spike_probability as number;
          }
        }

        // Parse RTM
        if (isRecord(rtmRes) && Array.isArray(rtmRes.predictions)) {
          setRtmPrices(rtmRes.predictions as RtmPrediction[]);
        }

        const points: ChartPoint[] = damPreds.map((p, i) => ({
          hour: i + 1,
          dam_price: p.predicted_price,
          spike_prob: spikeProb * 100, // uniform for now — single spike value
        }));
        setChartData(points);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load price forecast");
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchAll();
    return () => controller.abort();
  }, [refreshKey]);

  const getHighPrice = (): number => {
    if (!chartData) return 0;
    return Math.max(...chartData.map((p) => p.dam_price));
  };

  const getLowPrice = (): number => {
    if (!chartData) return 0;
    return Math.min(...chartData.map((p) => p.dam_price));
  };

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>Price Forecast</h3>
        <span className="dash-subtitle">Next 24h DAM + RTM + Spike</span>
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !chartData && <div className="dash-loading">Loading price forecast...</div>}

      {chartData && (
        <>
          {/* RTM quick metrics */}
          <div className="price-metrics">
            {rtmPrices && rtmPrices.map((r) => (
              <div key={r.horizon} className="price-metric">
                <span className="price-metric-label">RTM {r.horizon}</span>
                <span className="price-metric-value">${r.predicted_price.toFixed(2)}</span>
              </div>
            ))}
            <div className="price-metric">
              <span className="price-metric-label">DAM High</span>
              <span className="price-metric-value" style={{ color: "#ff6b6b" }}>
                ${getHighPrice().toFixed(2)}
              </span>
            </div>
            <div className="price-metric">
              <span className="price-metric-label">DAM Low</span>
              <span className="price-metric-value" style={{ color: "#4ecdc4" }}>
                ${getLowPrice().toFixed(2)}
              </span>
            </div>
            {spike && (
              <div className="price-metric">
                <span className="price-metric-label">Spike Risk</span>
                <span
                  className="price-metric-value"
                  style={{ color: spike.spike_probability >= 0.5 ? "#d63031" : "#00b894" }}
                >
                  {(spike.spike_probability * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>

          {/* DAM price chart */}
          <div className="dash-chart-wrapper" style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="hour"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 10 }}
                  tickFormatter={(h) => `HE${h}`}
                  interval={2}
                />
                <YAxis
                  yAxisId="price"
                  stroke="#888"
                  tick={{ fill: "#888", fontSize: 10 }}
                  tickFormatter={(v) => `$${v}`}
                />
                <YAxis
                  yAxisId="spike"
                  orientation="right"
                  stroke="#d63031"
                  tick={{ fill: "#d63031", fontSize: 10 }}
                  domain={[0, 100]}
                  hide
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e1e3a", border: "1px solid #444", borderRadius: 6 }}
                  labelFormatter={(h) => `Hour Ending ${h}`}
                  formatter={(value, name) => {
                    const v = Number(value ?? 0);
                    if (name === "DAM Price") return [`$${v.toFixed(2)}`, ""];
                    if (name === "Spike %") return [`${v.toFixed(0)}%`, ""];
                    return [v, ""];
                  }}
                />
                <Area
                  yAxisId="spike"
                  type="monotone"
                  dataKey="spike_prob"
                  name="Spike %"
                  fill="#d6303122"
                  stroke="#d6303155"
                  strokeWidth={0}
                />
                <Bar
                  yAxisId="price"
                  dataKey="dam_price"
                  name="DAM Price"
                  radius={[2, 2, 0, 0]}
                  fill="#4ecdc4"
                >
                  {chartData.map((entry, index) => {
                    const max = getHighPrice();
                    const ratio = entry.dam_price / (max || 1);
                    const color = ratio > 0.85 ? "#ff6b6b" : ratio > 0.6 ? "#ffd93d" : "#4ecdc4";
                    return <Bar key={index} fill={color} />;
                  })}
                </Bar>
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="dam_price"
                  name="DAM Price"
                  stroke="#4ecdc4"
                  strokeWidth={2}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
