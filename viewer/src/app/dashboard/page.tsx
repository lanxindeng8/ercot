"use client";

import React, { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import StatusBanner from "@/components/dashboard/StatusBanner";
import MiningScheduleCard from "@/components/dashboard/MiningScheduleCard";
import BessArbitrageCard from "@/components/dashboard/BessArbitrageCard";
import PriceForecastCard from "@/components/dashboard/PriceForecastCard";
import AlertHistoryCard from "@/components/dashboard/AlertHistoryCard";

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

export default function DashboardPage() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [countdown, setCountdown] = useState(REFRESH_INTERVAL / 1000);

  const refresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
    setLastRefresh(new Date());
    setCountdown(REFRESH_INTERVAL / 1000);
  }, []);

  useEffect(() => {
    const interval = setInterval(refresh, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [refresh]);

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
          <h1>Operations Dashboard</h1>
          <p>Unified dispatch signals & market intelligence</p>
        </div>
        <nav className="dash-nav">
          <Link href="/dashboard" className="dash-nav-link active">Dashboard</Link>
          <Link href="/" className="dash-nav-link">Market Data</Link>
          <Link href="/predictions" className="dash-nav-link">Predictions</Link>
          <Link href="/dispatch" className="dash-nav-link">Dispatch</Link>
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

      {/* Status Banner */}
      <StatusBanner refreshKey={refreshKey} />

      {/* Main grid: left column (operations), right column (alerts) */}
      <div className="cust-grid">
        <div className="cust-col-main">
          <MiningScheduleCard refreshKey={refreshKey} />
          <BessArbitrageCard refreshKey={refreshKey} />
        </div>
        <div className="cust-col-side">
          <PriceForecastCard refreshKey={refreshKey} />
          <AlertHistoryCard refreshKey={refreshKey} />
        </div>
      </div>

      <footer className="dash-footer">
        <p>
          Dispatch signals powered by ML forecasts, LP optimization & risk analysis.
          Not financial advice.
        </p>
      </footer>
    </div>
  );
}
