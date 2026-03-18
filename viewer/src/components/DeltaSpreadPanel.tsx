"use client";

import React, { useState } from "react";
import { asNumber, asString, fetchPredictionJson, isRecord } from "@/components/predictions";

interface SpreadPrediction {
  target_hour: number;
  target_date: string;
  spread_value: number;
  direction: string;
  direction_probability: number;
  spread_interval: number;
  spread_interval_label: string;
  interval_probabilities: number[];
  signal: string;
}

interface SpreadResponse {
  status: string;
  model: string;
  settlement_point: string;
  forecast_horizon_hours: number;
  timestamp: string;
  predictions: SpreadPrediction[];
}

const SIGNAL_STYLES: Record<string, { bg: string; color: string; label: string }> = {
  STRONG_LONG: { bg: "#00b894", color: "#fff", label: "STRONG LONG" },
  LONG: { bg: "#00cec9", color: "#1a1a2e", label: "LONG" },
  HOLD: { bg: "#636e72", color: "#fff", label: "HOLD" },
  SHORT: { bg: "#e17055", color: "#fff", label: "SHORT" },
  STRONG_SHORT: { bg: "#d63031", color: "#fff", label: "STRONG SHORT" },
};

interface DeltaSpreadPanelProps {
  refreshKey: number;
}

function isSpreadResponse(value: unknown): value is SpreadResponse {
  if (!isRecord(value)) return false;
  if (asString(value.settlement_point) === null || asNumber(value.forecast_horizon_hours) === null) return false;
  if (!Array.isArray(value.predictions)) return false;

  return value.predictions.every((prediction) => {
    if (!isRecord(prediction)) return false;
    return (
      asNumber(prediction.target_hour) !== null &&
      asString(prediction.target_date) !== null &&
      asNumber(prediction.spread_value) !== null &&
      asString(prediction.direction) !== null &&
      asNumber(prediction.direction_probability) !== null &&
      asNumber(prediction.spread_interval) !== null &&
      asString(prediction.spread_interval_label) !== null &&
      Array.isArray(prediction.interval_probabilities) &&
      asString(prediction.signal) !== null
    );
  });
}

export default function DeltaSpreadPanel({ refreshKey }: DeltaSpreadPanelProps) {
  const [data, setData] = useState<SpreadResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>("/api/predictions/delta-spread", {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!isSpreadResponse(result)) {
          throw new Error("Delta spread response was malformed");
        }
        setData(result);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load spread predictions");
          setData(null);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchData();
    return () => controller.abort();
  }, [refreshKey]);

  const totalPnl = data?.predictions.reduce((sum, p) => {
    if (p.signal === "LONG" || p.signal === "STRONG_LONG") return sum + Math.abs(p.spread_value);
    if (p.signal === "SHORT" || p.signal === "STRONG_SHORT") return sum + Math.abs(p.spread_value);
    return sum;
  }, 0) ?? 0;

  const signalCounts = data?.predictions.reduce(
    (acc, p) => {
      acc[p.signal] = (acc[p.signal] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  ) ?? {};

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>Delta Spread & Trading Signals</h3>
        {data && <span className="dash-subtitle">{data.settlement_point} | {data.predictions[0]?.target_date}</span>}
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading spread predictions...</div>}

      {data && (
        <>
          <div className="spread-summary">
            <div className="spread-stat">
              <div className="spread-stat-label">Potential PnL</div>
              <div className="spread-stat-value positive">${totalPnl.toFixed(2)}/MWh</div>
            </div>
            <div className="spread-stat">
              <div className="spread-stat-label">Active Signals</div>
              <div className="spread-stat-value">
                {(signalCounts["LONG"] || 0) + (signalCounts["STRONG_LONG"] || 0) + (signalCounts["SHORT"] || 0) + (signalCounts["STRONG_SHORT"] || 0)} / {data.predictions.length}
              </div>
            </div>
            <div className="signal-badges">
              {Object.entries(signalCounts).map(([signal, count]) => {
                const style = SIGNAL_STYLES[signal] || SIGNAL_STYLES.HOLD;
                return (
                  <span key={signal} className="signal-badge" style={{ backgroundColor: style.bg, color: style.color }}>
                    {style.label}: {count}
                  </span>
                );
              })}
            </div>
          </div>

          <div className="spread-table-wrapper">
            <table className="spread-table">
              <thead>
                <tr>
                  <th>Hour</th>
                  <th>Spread</th>
                  <th>Direction</th>
                  <th>Prob</th>
                  <th>Interval</th>
                  <th>Signal</th>
                </tr>
              </thead>
              <tbody>
                {data.predictions.map((pred) => {
                  const sigStyle = SIGNAL_STYLES[pred.signal] || SIGNAL_STYLES.HOLD;
                  return (
                    <tr key={pred.target_hour}>
                      <td className="spread-hour">HE {pred.target_hour}</td>
                      <td className={`spread-value ${pred.spread_value >= 0 ? "positive" : "negative"}`}>
                        ${pred.spread_value.toFixed(2)}
                      </td>
                      <td className="spread-direction">{pred.direction}</td>
                      <td className="spread-prob">{(pred.direction_probability * 100).toFixed(0)}%</td>
                      <td className="spread-interval">{pred.spread_interval_label}</td>
                      <td>
                        <span className="signal-pill" style={{ backgroundColor: sigStyle.bg, color: sigStyle.color }}>
                          {sigStyle.label}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
