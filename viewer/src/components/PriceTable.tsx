"use client";

import React, { useEffect, useState, useCallback } from "react";

interface PivotedRow {
  time: string;
  prices: Record<string, number | null>;
  predictions?: Record<string, number | null>;
}

interface TableData {
  date: string;
  settlementPoints: string[];
  data: PivotedRow[];
  lastUpdated: string;
}

interface PriceTableProps {
  title: string;
  apiEndpoint: string;
  timeColumnLabel: string;
  defaultDate?: string;
}

export default function PriceTable({
  title,
  apiEndpoint,
  timeColumnLabel,
  defaultDate,
}: PriceTableProps) {
  const [data, setData] = useState<TableData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDate, setSelectedDate] = useState(defaultDate || "");

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const url = selectedDate
        ? `${apiEndpoint}?date=${selectedDate}`
        : apiEndpoint;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error("Failed to fetch data");
      }

      const result = await response.json();
      setData(result);
      if (!selectedDate && result.date) {
        setSelectedDate(result.date);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [apiEndpoint, selectedDate]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const formatPrice = (price: number | null): string => {
    if (price === null) return "";
    return price.toFixed(2);
  };

  const lastUpdatedFormatted = data?.lastUpdated
    ? new Date(data.lastUpdated).toLocaleString("en-US", {
        timeZone: "America/Chicago",
        month: "short",
        day: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }) + " CST"
    : "";

  return (
    <div className="table-container">
      <div className="table-header">
        <h2>{title}</h2>
        <div className="controls">
          <label>
            Operating Day:{" "}
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            />
          </label>
          <button onClick={fetchData} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
        {data && (
          <div className="last-updated">Last Updated: {lastUpdatedFormatted}</div>
        )}
      </div>

      {error && <div className="error">Error: {error}</div>}

      {loading && !data && <div className="loading">Loading data...</div>}

      {data && data.data.length > 0 && (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th rowSpan={2}>{timeColumnLabel}</th>
                {data.settlementPoints.map((point) => (
                  <th key={point} colSpan={2} className="point-header">
                    {point}
                  </th>
                ))}
              </tr>
              <tr>
                {data.settlementPoints.map((point) => (
                  <React.Fragment key={point}>
                    <th className="sub-header">Actual</th>
                    <th className="sub-header pred-header">Pred</th>
                  </React.Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.data.map((row, idx) => (
                <tr key={idx}>
                  <td className="time-cell">{row.time}</td>
                  {data.settlementPoints.map((point) => (
                    <React.Fragment key={point}>
                      <td className="price-cell">
                        {formatPrice(row.prices[point])}
                      </td>
                      <td className="price-cell pred-cell">
                        {formatPrice(row.predictions?.[point] ?? null)}
                      </td>
                    </React.Fragment>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && data.data.length === 0 && (
        <div className="no-data">No data available for the selected date.</div>
      )}
    </div>
  );
}
