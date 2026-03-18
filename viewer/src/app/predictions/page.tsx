"use client";

import React, { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import DamForecastChart from "@/components/DamForecastChart";
import RtmHorizonChart from "@/components/RtmHorizonChart";
import DeltaSpreadPanel from "@/components/DeltaSpreadPanel";
import SpikeAlertPanel from "@/components/SpikeAlertPanel";
import WindForecastChart from "@/components/WindForecastChart";
import LoadForecastChart from "@/components/LoadForecastChart";
import BessSchedulePanel from "@/components/BessSchedulePanel";
import AccuracyPanel from "@/components/AccuracyPanel";

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

export default function PredictionsPage() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [countdown, setCountdown] = useState(REFRESH_INTERVAL / 1000);

  const refresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
    setLastRefresh(new Date());
    setCountdown(REFRESH_INTERVAL / 1000);
  }, []);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(refresh, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [refresh]);

  // Countdown timer
  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((c) => (c > 0 ? c - 1 : REFRESH_INTERVAL / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatCountdown = (seconds: number): string => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="dash-container">
      <header className="dash-header">
        <div className="dash-header-left">
          <h1>ERCOT Prediction Dashboard</h1>
          <p>ML-powered price forecasts & trading signals</p>
        </div>
        <nav className="dash-nav">
          <Link href="/" className="dash-nav-link">Market Data</Link>
          <Link href="/predictions" className="dash-nav-link active">Predictions</Link>
        </nav>
        <div className="dash-header-right">
          <div className="dash-refresh-info">
            <span className="dash-countdown">Next refresh: {formatCountdown(countdown)}</span>
            <span className="dash-last-update">
              Updated: {lastRefresh.toLocaleTimeString("en-US", {
                timeZone: "America/Chicago",
                hour: "2-digit",
                minute: "2-digit",
              })} CT
            </span>
          </div>
          <button className="dash-refresh-btn" onClick={refresh}>
            Refresh Now
          </button>
        </div>
      </header>

      <div className="dash-grid">
        <div className="dash-col-wide">
          <DamForecastChart refreshKey={refreshKey} />
        </div>
        <div className="dash-col-narrow">
          <SpikeAlertPanel refreshKey={refreshKey} />
          <RtmHorizonChart refreshKey={refreshKey} />
        </div>
      </div>

      <div className="dash-grid">
        <div className="dash-col-wide">
          <WindForecastChart refreshKey={refreshKey} />
        </div>
        <div className="dash-col-narrow">
          <LoadForecastChart refreshKey={refreshKey} />
        </div>
      </div>

      <div className="dash-full-width">
        <BessSchedulePanel refreshKey={refreshKey} />
      </div>

      <div className="dash-full-width">
        <DeltaSpreadPanel refreshKey={refreshKey} />
      </div>

      <div className="dash-full-width">
        <AccuracyPanel refreshKey={refreshKey} />
      </div>

      <footer className="dash-footer">
        <p>
          Predictions powered by LightGBM, CatBoost & LP optimization trained on ERCOT historical data.
          Not financial advice.
        </p>
      </footer>
    </div>
  );
}
