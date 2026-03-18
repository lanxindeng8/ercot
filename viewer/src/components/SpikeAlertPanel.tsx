"use client";

import React, { useState } from "react";
import { asNumber, asString, fetchPredictionJson, isRecord } from "@/components/predictions";

interface SpikeAlert {
  is_spike: boolean;
  spike_probability: number;
  confidence: string;
  threshold: number;
  lookahead: string;
}

interface SpikeResponse {
  status: string;
  model: string;
  settlement_point: string;
  generated_at: string;
  alert: SpikeAlert;
}

interface SpikeAlertPanelProps {
  refreshKey: number;
}

const CONFIDENCE_CLASS: Record<string, string> = {
  high: "spike-conf-high",
  medium: "spike-conf-medium",
  low: "spike-conf-low",
};

function isSpikeResponse(value: unknown): value is SpikeResponse {
  if (!isRecord(value)) return false;
  if (asString(value.settlement_point) === null || !isRecord(value.alert)) return false;
  return (
    typeof value.alert.is_spike === "boolean" &&
    asNumber(value.alert.spike_probability) !== null &&
    asString(value.alert.confidence) !== null &&
    asNumber(value.alert.threshold) !== null &&
    asString(value.alert.lookahead) !== null
  );
}

export default function SpikeAlertPanel({ refreshKey }: SpikeAlertPanelProps) {
  const [data, setData] = useState<SpikeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchPredictionJson<unknown>("/api/predictions/spike", {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!isSpikeResponse(result)) {
          throw new Error("Spike response was malformed");
        }
        setData(result);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load spike alert");
          setData(null);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchData();
    return () => controller.abort();
  }, [refreshKey]);

  const probability = data?.alert.spike_probability ?? 0;
  const isSpike = data?.alert.is_spike ?? false;
  const confidence = data?.alert.confidence ?? "low";
  const confidenceClass = CONFIDENCE_CLASS[confidence] ?? CONFIDENCE_CLASS.low;

  // SVG gauge parameters
  const gaugeRadius = 80;
  const gaugeStroke = 12;
  const centerX = 100;
  const centerY = 95;
  const startAngle = -210;
  const endAngle = 30;
  const totalAngle = endAngle - startAngle;
  const circumference = (totalAngle / 360) * 2 * Math.PI * gaugeRadius;
  const filledLength = probability * circumference;

  const getGaugeColor = (prob: number): string => {
    if (prob >= 0.8) return "#d63031";
    if (prob >= 0.5) return "#e17055";
    if (prob >= 0.3) return "#ffd93d";
    return "#00b894";
  };

  const polarToCartesian = (cx: number, cy: number, r: number, angleDeg: number) => {
    const angleRad = (angleDeg * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(angleRad),
      y: cy + r * Math.sin(angleRad),
    };
  };

  const describeArc = (cx: number, cy: number, r: number, startAng: number, endAng: number) => {
    const start = polarToCartesian(cx, cy, r, endAng);
    const end = polarToCartesian(cx, cy, r, startAng);
    const largeArc = endAng - startAng > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
  };

  const valueAngle = startAngle + probability * totalAngle;

  return (
    <div className={`dash-card spike-card ${isSpike ? "spike-active" : "spike-clear"}`}>
      <div className="dash-card-header">
        <h3>Spike Alert</h3>
        {data && <span className="dash-subtitle">{data.settlement_point} | {data.alert.lookahead}</span>}
      </div>

      {error && <div className="dash-error">{error}</div>}
      {loading && !data && <div className="dash-loading">Loading spike alert...</div>}

      {data && (
        <div className="spike-content">
          <div className="spike-gauge">
            <svg viewBox="0 0 200 130" width="100%" height="130" preserveAspectRatio="xMidYMid meet">
              {/* Background arc */}
              <path
                d={describeArc(centerX, centerY, gaugeRadius, startAngle, endAngle)}
                fill="none"
                stroke="#333"
                strokeWidth={gaugeStroke}
                strokeLinecap="round"
              />
              {/* Filled arc */}
              {probability > 0 && (
                <path
                  d={describeArc(centerX, centerY, gaugeRadius, startAngle, valueAngle)}
                  fill="none"
                  stroke={getGaugeColor(probability)}
                  strokeWidth={gaugeStroke}
                  strokeLinecap="round"
                />
              )}
              {/* Center text */}
              <text x={centerX} y={centerY - 10} textAnchor="middle" fill="#fff" fontSize="28" fontWeight="bold">
                {(probability * 100).toFixed(0)}%
              </text>
              <text x={centerX} y={centerY + 12} textAnchor="middle" fill="#888" fontSize="11">
                spike probability
              </text>
            </svg>
          </div>

          <div className="spike-status">
            <div className={`spike-badge ${isSpike ? "spike-danger" : "spike-safe"}`}>
              {isSpike ? "SPIKE ALERT" : "NO SPIKE"}
            </div>
            <div className="spike-meta">
              <div className="spike-meta-row">
                <span className="spike-meta-label">Confidence</span>
                <span className={`spike-confidence ${confidenceClass}`}>
                  {confidence.toUpperCase()}
                </span>
              </div>
              <div className="spike-meta-row">
                <span className="spike-meta-label">Threshold</span>
                <span>{(data.alert.threshold * 100).toFixed(0)}%</span>
              </div>
              <div className="spike-meta-row">
                <span className="spike-meta-label">Lookahead</span>
                <span>{data.alert.lookahead}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
