"use client";

import React, { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import BessSignalPanel from "@/components/BessSignalPanel";

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

export default function DispatchPage() {
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
          <h1>Dispatch Dashboard</h1>
          <p>BESS arbitrage signals & mining coordination</p>
        </div>
        <nav className="dash-nav">
          <Link href="/dashboard" className="dash-nav-link">Dashboard</Link>
          <Link href="/" className="dash-nav-link">Market Data</Link>
          <Link href="/predictions" className="dash-nav-link">Predictions</Link>
          <Link href="/dispatch" className="dash-nav-link active">Dispatch</Link>
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

      <div className="dash-full-width">
        <BessSignalPanel refreshKey={refreshKey} />
      </div>

      <footer className="dash-footer">
        <p>
          Dispatch signals powered by LP optimization, spike detection & risk analysis.
          Not financial advice.
        </p>
      </footer>
    </div>
  );
}
