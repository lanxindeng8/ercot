"use client";

import React, { useEffect, useState, useCallback } from "react";
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

interface ChartDataPoint {
  hour: number;
  rtmActual: number | null;
  damActual: number | null;
  damPred: number | null;
}

interface ChartResponse {
  date: string;
  settlementPoint: string;
  data: ChartDataPoint[];
  lastUpdated: string;
}

export default function PriceChart() {
  const [data, setData] = useState<ChartResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/chart-data");
      if (!response.ok) {
        throw new Error("Failed to fetch chart data");
      }
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const formatHour = (hour: number): string => {
    if (hour === 0) return "12AM";
    if (hour === 12) return "12PM";
    if (hour < 12) return `${hour}AM`;
    return `${hour - 12}PM`;
  };

  const lastUpdatedFormatted = data?.lastUpdated
    ? new Date(data.lastUpdated).toLocaleString("en-US", {
        timeZone: "America/Chicago",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      }) + " CST"
    : "";

  return (
    <div className="chart-container">
      <div className="chart-header">
        <h2>LZ_WEST Price Chart - {data?.date || "Today"}</h2>
        <div className="chart-controls">
          <button onClick={fetchData} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
        {data && (
          <div className="last-updated">Last Updated: {lastUpdatedFormatted}</div>
        )}
      </div>

      {error && <div className="error">Error: {error}</div>}

      {loading && !data && <div className="loading">Loading chart...</div>}

      {data && (
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={350}>
            <LineChart
              data={data.data}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis
                dataKey="hour"
                tickFormatter={formatHour}
                stroke="#888"
                tick={{ fill: "#888" }}
                interval={2}
              />
              <YAxis
                stroke="#888"
                tick={{ fill: "#888" }}
                label={{
                  value: "$/MWh",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#888",
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1a1a",
                  border: "1px solid #333",
                  borderRadius: "4px",
                }}
                labelFormatter={(hour) => `Hour: ${formatHour(hour as number)}`}
                formatter={(value) => {
                  if (value === null || value === undefined) return ["N/A", ""];
                  return [`$${Number(value).toFixed(2)}`, ""];
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="rtmActual"
                name="RTM Actual"
                stroke="#ff6b6b"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="damActual"
                name="DAM Actual"
                stroke="#4ecdc4"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="damPred"
                name="DAM Predicted"
                stroke="#4ecdc4"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
