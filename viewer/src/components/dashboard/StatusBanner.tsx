"use client";

import React, { useState } from "react";
import { fetchPredictionJson, isRecord, asNumber, asString } from "@/components/predictions";

interface SpikeData {
  is_spike: boolean;
  spike_probability: number;
  confidence: string;
}

interface LoadPoint {
  hour_ending: string;
  predicted_load_mw: number;
}

interface WindPoint {
  hour_ending: string;
  predicted_mw: number;
}

interface DamPoint {
  hour_ending: string;
  predicted_price: number;
}

interface StatusData {
  load: { current: number; peak: number } | null;
  wind: { current: number } | null;
  price: { current: number; avg: number } | null;
  spike: SpikeData | null;
}

interface StatusBannerProps {
  refreshKey: number;
}

function getCurrentHE(): number {
  const now = new Date();
  const ct = new Date(now.toLocaleString("en-US", { timeZone: "America/Chicago" }));
  return ct.getHours() + 1; // hour-ending
}

export default function StatusBanner({ refreshKey }: StatusBannerProps) {
  const [data, setData] = useState<StatusData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();

    async function fetchAll() {
      setLoading(true);
      setError(null);
      try {
        const [loadRes, windRes, damRes, spikeRes] = await Promise.all([
          fetchPredictionJson<unknown>("/api/predictions/load", { signal: controller.signal }).catch(() => null),
          fetchPredictionJson<unknown>("/api/predictions/wind", { signal: controller.signal }).catch(() => null),
          fetchPredictionJson<unknown>("/api/predictions/dam/next-day?settlement_point=HB_WEST", { signal: controller.signal }).catch(() => null),
          fetchPredictionJson<unknown>("/api/predictions/spike", { signal: controller.signal }).catch(() => null),
        ]);
        if (controller.signal.aborted) return;

        const currentHE = getCurrentHE();
        const status: StatusData = { load: null, wind: null, price: null, spike: null };

        // Parse load
        if (isRecord(loadRes) && Array.isArray(loadRes.predictions)) {
          const preds = loadRes.predictions as LoadPoint[];
          const current = preds.find((p) => {
            const he = parseInt(p.hour_ending?.replace(/[^0-9]/g, "") || "0");
            return he === currentHE;
          });
          const peak = preds.reduce((max, p) => Math.max(max, p.predicted_load_mw || 0), 0);
          status.load = {
            current: current?.predicted_load_mw ?? preds[0]?.predicted_load_mw ?? 0,
            peak,
          };
        }

        // Parse wind
        if (isRecord(windRes) && Array.isArray(windRes.predictions)) {
          const preds = windRes.predictions as WindPoint[];
          const current = preds.find((p) => {
            const he = parseInt(p.hour_ending?.replace(/[^0-9]/g, "") || "0");
            return he === currentHE;
          });
          status.wind = {
            current: current?.predicted_mw ?? preds[0]?.predicted_mw ?? 0,
          };
        }

        // Parse DAM price
        if (isRecord(damRes) && Array.isArray(damRes.predictions)) {
          const preds = damRes.predictions as DamPoint[];
          const current = preds.find((p) => {
            const he = parseInt(p.hour_ending?.replace(/[^0-9]/g, "") || "0");
            return he === currentHE;
          });
          const avg = preds.reduce((s, p) => s + (p.predicted_price || 0), 0) / (preds.length || 1);
          status.price = {
            current: current?.predicted_price ?? preds[0]?.predicted_price ?? 0,
            avg,
          };
        }

        // Parse spike
        if (isRecord(spikeRes) && isRecord(spikeRes.alert)) {
          const a = spikeRes.alert;
          if (typeof a.is_spike === "boolean" && asNumber(a.spike_probability) !== null) {
            status.spike = {
              is_spike: a.is_spike as boolean,
              spike_probability: a.spike_probability as number,
              confidence: asString(a.confidence) ?? "low",
            };
          }
        }

        setData(status);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Failed to load status");
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    fetchAll();
    return () => controller.abort();
  }, [refreshKey]);

  const getPriceLevel = (price: number): { label: string; color: string } => {
    if (price >= 100) return { label: "HIGH", color: "#d63031" };
    if (price >= 50) return { label: "ELEVATED", color: "#e17055" };
    if (price >= 25) return { label: "NORMAL", color: "#00b894" };
    return { label: "LOW", color: "#4ecdc4" };
  };

  const getSpikeColor = (prob: number): string => {
    if (prob >= 0.7) return "#d63031";
    if (prob >= 0.4) return "#ffd93d";
    return "#00b894";
  };

  if (error) return <div className="status-banner status-banner-error">{error}</div>;
  if (loading && !data) {
    return (
      <div className="status-banner">
        <div className="status-banner-loading">Loading market status...</div>
      </div>
    );
  }
  if (!data) return null;

  const priceLevel = data.price ? getPriceLevel(data.price.current) : null;

  return (
    <div className={`status-banner ${data.spike?.is_spike ? "status-banner-alert" : ""}`}>
      <div className="status-item">
        <span className="status-label">System Load</span>
        <span className="status-value">
          {data.load ? `${(data.load.current / 1000).toFixed(1)} GW` : "--"}
        </span>
        {data.load && (
          <span className="status-detail">Peak: {(data.load.peak / 1000).toFixed(1)} GW</span>
        )}
      </div>

      <div className="status-divider" />

      <div className="status-item">
        <span className="status-label">Wind Generation</span>
        <span className="status-value">
          {data.wind ? `${(data.wind.current / 1000).toFixed(1)} GW` : "--"}
        </span>
      </div>

      <div className="status-divider" />

      <div className="status-item">
        <span className="status-label">DAM Price (HB_WEST)</span>
        <span className="status-value" style={{ color: priceLevel?.color }}>
          {data.price ? `$${data.price.current.toFixed(2)}` : "--"}
        </span>
        {priceLevel && (
          <span className="status-badge" style={{ background: priceLevel.color }}>
            {priceLevel.label}
          </span>
        )}
      </div>

      <div className="status-divider" />

      <div className="status-item">
        <span className="status-label">Spike Risk</span>
        <span
          className="status-value"
          style={{ color: data.spike ? getSpikeColor(data.spike.spike_probability) : undefined }}
        >
          {data.spike ? `${(data.spike.spike_probability * 100).toFixed(0)}%` : "--"}
        </span>
        {data.spike?.is_spike && (
          <span className="status-badge status-badge-pulse" style={{ background: "#d63031" }}>
            ALERT
          </span>
        )}
      </div>
    </div>
  );
}
