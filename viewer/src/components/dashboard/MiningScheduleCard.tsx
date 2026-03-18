"use client";

import React, { useState } from "react";
import { fetchPredictionJson, isRecord, asNumber, asString } from "@/components/predictions";

interface ScheduleHour {
  hour_ending: string;
  dam_price: number;
  action: string;
  reason: string;
  spike_probability: number;
  bess_action: string;
}

interface ScheduleSummary {
  hours_to_run: number;
  hours_to_curtail: number;
  expected_cost_savings: number;
  peak_price: number;
  avg_on_price: number;
  spike_hours: number;
}

interface ScheduleResponse {
  status: string;
  settlement_point: string;
  date: string;
  schedule: ScheduleHour[];
  summary: ScheduleSummary;
}

interface MiningScheduleCardProps {
  refreshKey: number;
}

function isScheduleResponse(v: unknown): v is ScheduleResponse {
  if (!isRecord(v)) return false;
  if (asString(v.status) !== "success") return false;
  if (!Array.isArray(v.schedule) || !isRecord(v.summary)) return false;
  return asNumber(v.summary.hours_to_run) !== null;
}

const ACTION_COLORS: Record<string, string> = {
  ON: "#00b894",
  OFF: "#d63031",
  CURTAIL: "#e17055",
};

export default function MiningScheduleCard({ refreshKey }: MiningScheduleCardProps) {
  const [data, setData] = useState<ScheduleResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>("/api/dispatch/mining/schedule", {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!isScheduleResponse(result)) {
          throw new Error("Mining schedule response was malformed");
        }
        setData(result);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load mining schedule");
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
        <h3>Mining Operations</h3>
        {data && <span className="dash-subtitle">{data.settlement_point} | {data.date}</span>}
      </div>

      {error && (
        <div className="dash-error">
          {error}
          <button className="retry-btn" onClick={() => setError(null)}>Retry</button>
        </div>
      )}
      {loading && !data && <div className="dash-loading">Loading mining schedule...</div>}

      {data && (
        <>
          {/* Summary stats */}
          <div className="mining-summary">
            <div className="mining-stat">
              <span className="mining-stat-label">Run Hours</span>
              <span className="mining-stat-value" style={{ color: "#00b894" }}>
                {data.summary.hours_to_run}h
              </span>
            </div>
            <div className="mining-stat">
              <span className="mining-stat-label">Curtail Hours</span>
              <span className="mining-stat-value" style={{ color: "#e17055" }}>
                {data.summary.hours_to_curtail}h
              </span>
            </div>
            <div className="mining-stat">
              <span className="mining-stat-label">Est. Savings</span>
              <span className="mining-stat-value" style={{ color: "#4ecdc4" }}>
                ${data.summary.expected_cost_savings.toFixed(0)}
              </span>
            </div>
            <div className="mining-stat">
              <span className="mining-stat-label">Avg ON Price</span>
              <span className="mining-stat-value">
                ${data.summary.avg_on_price.toFixed(2)}
              </span>
            </div>
          </div>

          {/* Timeline bar */}
          <div className="mining-timeline">
            <div className="mining-timeline-label">24h Schedule</div>
            <div className="mining-timeline-bar">
              {data.schedule.map((h, i) => {
                const action = h.action.toUpperCase();
                const color = ACTION_COLORS[action] || ACTION_COLORS.OFF;
                return (
                  <div
                    key={i}
                    className="mining-timeline-cell"
                    style={{ background: color }}
                    title={`HE${i + 1}: ${h.action} - $${h.dam_price.toFixed(2)} (${h.reason})`}
                  />
                );
              })}
            </div>
            <div className="mining-timeline-hours">
              <span>1</span>
              <span>6</span>
              <span>12</span>
              <span>18</span>
              <span>24</span>
            </div>
          </div>

          {/* Legend */}
          <div className="mining-legend">
            <span className="mining-legend-item">
              <span className="mining-legend-dot" style={{ background: "#00b894" }} /> ON
            </span>
            <span className="mining-legend-item">
              <span className="mining-legend-dot" style={{ background: "#d63031" }} /> OFF
            </span>
            {data.summary.spike_hours > 0 && (
              <span className="mining-legend-item" style={{ color: "#ffd93d" }}>
                {data.summary.spike_hours} spike hour{data.summary.spike_hours > 1 ? "s" : ""}
              </span>
            )}
          </div>

          {/* Schedule table */}
          <div className="mining-table-wrapper">
            <table className="mining-table">
              <thead>
                <tr>
                  <th>HE</th>
                  <th>Action</th>
                  <th>DAM Price</th>
                  <th>Reason</th>
                  <th>Spike %</th>
                  <th>BESS</th>
                </tr>
              </thead>
              <tbody>
                {data.schedule.map((h, i) => (
                  <tr key={i}>
                    <td className="mining-he">{i + 1}</td>
                    <td>
                      <span
                        className="mining-action-pill"
                        style={{
                          background: ACTION_COLORS[h.action.toUpperCase()] || "#636e72",
                        }}
                      >
                        {h.action.toUpperCase()}
                      </span>
                    </td>
                    <td className="mining-price">${h.dam_price.toFixed(2)}</td>
                    <td className="mining-reason">{h.reason}</td>
                    <td style={{ color: h.spike_probability > 0.5 ? "#d63031" : "#888" }}>
                      {(h.spike_probability * 100).toFixed(0)}%
                    </td>
                    <td className="mining-bess">{h.bess_action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
