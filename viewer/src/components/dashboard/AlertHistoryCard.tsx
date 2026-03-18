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

interface BessSignal {
  hour_ending: string;
  action: string;
  power_mw: number;
  revenue_estimate: number;
  risk_flag: string;
  mining_curtail: boolean;
}

interface AlertEvent {
  time: string;
  type: "mining" | "bess" | "spike";
  action: string;
  detail: string;
  severity: "info" | "warning" | "critical";
}

interface AlertHistoryCardProps {
  refreshKey: number;
}

function getCurrentHE(): number {
  const now = new Date();
  const ct = new Date(now.toLocaleString("en-US", { timeZone: "America/Chicago" }));
  return ct.getHours() + 1;
}

export default function AlertHistoryCard({ refreshKey }: AlertHistoryCardProps) {
  const [events, setEvents] = useState<AlertEvent[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchAll() {
      setLoading(true);
      setError(null);
      try {
        const [miningRes, bessRes] = await Promise.all([
          fetchPredictionJson<unknown>("/api/dispatch/mining/schedule", {
            signal: controller.signal,
          }).catch(() => null),
          fetchPredictionJson<unknown>("/api/dispatch/bess/daily-signals", {
            signal: controller.signal,
          }).catch(() => null),
        ]);
        if (controller.signal.aborted) return;

        const alerts: AlertEvent[] = [];
        const currentHE = getCurrentHE();

        // Mining dispatch changes
        if (isRecord(miningRes) && Array.isArray(miningRes.schedule)) {
          const schedule = miningRes.schedule as ScheduleHour[];
          for (let i = 0; i < schedule.length; i++) {
            const h = schedule[i];
            const prevAction = i > 0 ? schedule[i - 1].action : null;
            // Only show transitions and notable events
            if (prevAction && h.action !== prevAction) {
              alerts.push({
                time: `HE${i + 1}`,
                type: "mining",
                action: h.action.toUpperCase(),
                detail: `Mining ${h.action.toUpperCase()} - ${h.reason}`,
                severity: h.action.toLowerCase() === "off" ? "warning" : "info",
              });
            }
            if (h.spike_probability > 0.5) {
              alerts.push({
                time: `HE${i + 1}`,
                type: "spike",
                action: "SPIKE RISK",
                detail: `${(h.spike_probability * 100).toFixed(0)}% spike probability - $${h.dam_price.toFixed(2)}`,
                severity: h.spike_probability > 0.8 ? "critical" : "warning",
              });
            }
          }
        }

        // BESS notable events
        if (isRecord(bessRes) && Array.isArray(bessRes.signals)) {
          const signals = bessRes.signals as BessSignal[];
          for (let i = 0; i < signals.length; i++) {
            const s = signals[i];
            if (s.risk_flag && s.risk_flag !== "normal") {
              alerts.push({
                time: `HE${i + 1}`,
                type: "bess",
                action: s.risk_flag.toUpperCase().replace("_", " "),
                detail: `BESS ${s.action} ${s.power_mw.toFixed(1)}MW - $${s.revenue_estimate.toFixed(2)}`,
                severity: "warning",
              });
            }
            if (s.mining_curtail) {
              alerts.push({
                time: `HE${i + 1}`,
                type: "mining",
                action: "CURTAIL",
                detail: `Mining curtailment recommended`,
                severity: "warning",
              });
            }
          }
        }

        // Sort by hour (extract number from HE string)
        alerts.sort((a, b) => {
          const ha = parseInt(a.time.replace("HE", ""));
          const hb = parseInt(b.time.replace("HE", ""));
          return ha - hb;
        });

        setEvents(alerts);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load alerts");
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchAll();
    return () => controller.abort();
  }, [refreshKey]);

  const SEVERITY_COLORS: Record<string, string> = {
    critical: "#d63031",
    warning: "#ffd93d",
    info: "#4ecdc4",
  };

  const TYPE_ICONS: Record<string, string> = {
    mining: "M",
    bess: "B",
    spike: "S",
  };

  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3>Alert History</h3>
        <span className="dash-subtitle">Today&apos;s dispatch events</span>
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !events && <div className="dash-loading">Loading alerts...</div>}

      {events && events.length === 0 && (
        <div className="alert-empty">No notable events today</div>
      )}

      {events && events.length > 0 && (
        <div className="alert-timeline">
          {events.map((evt, i) => (
            <div key={i} className={`alert-item alert-${evt.severity}`}>
              <div className="alert-time">{evt.time}</div>
              <div
                className="alert-icon"
                style={{ background: SEVERITY_COLORS[evt.severity] || "#4ecdc4" }}
              >
                {TYPE_ICONS[evt.type] || "?"}
              </div>
              <div className="alert-content">
                <span className="alert-action">{evt.action}</span>
                <span className="alert-detail">{evt.detail}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
