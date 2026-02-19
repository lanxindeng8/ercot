import { NextRequest, NextResponse } from "next/server";
import { queryRtmLmpData, queryDamLmpData, queryDamPredictions } from "@/lib/influxdb";
import { getToday } from "@/lib/utils";

const CHART_SETTLEMENT_POINT = "LZ_WEST";

interface ChartDataPoint {
  hour: number;
  rtmActual: number | null;
  damActual: number | null;
  damPred: number | null;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get("date") || getToday();

    // Fetch RTM, DAM, and DAM predictions in parallel
    const [rtmRecords, damRecords, damPredictions] = await Promise.all([
      queryRtmLmpData(date, [CHART_SETTLEMENT_POINT]),
      queryDamLmpData(date, [CHART_SETTLEMENT_POINT]),
      queryDamPredictions(date, [CHART_SETTLEMENT_POINT]),
    ]);

    // Initialize 24 hours of data
    const chartData: ChartDataPoint[] = Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      rtmActual: null,
      damActual: null,
      damPred: null,
    }));

    // Aggregate RTM data by hour (average of 5-min intervals)
    const rtmByHour: Map<number, number[]> = new Map();
    for (const record of rtmRecords) {
      // Convert UTC to Central Time (UTC-6)
      const centralTime = new Date(record.time.getTime() - 6 * 60 * 60 * 1000);
      const hour = centralTime.getUTCHours();
      if (!rtmByHour.has(hour)) {
        rtmByHour.set(hour, []);
      }
      rtmByHour.get(hour)!.push(record.lmp);
    }

    // Calculate hourly averages for RTM
    for (const [hour, prices] of rtmByHour) {
      if (hour >= 0 && hour < 24) {
        const avg = prices.reduce((a, b) => a + b, 0) / prices.length;
        chartData[hour].rtmActual = Math.round(avg * 100) / 100;
      }
    }

    // DAM data is already hourly
    for (const record of damRecords) {
      // Convert UTC to Central Time (UTC-6)
      const centralTime = new Date(record.time.getTime() - 6 * 60 * 60 * 1000);
      const hour = centralTime.getUTCHours();
      if (hour >= 0 && hour < 24) {
        chartData[hour].damActual = Math.round(record.lmp * 100) / 100;
      }
    }

    // DAM predictions
    for (const pred of damPredictions) {
      const hour = pred.hourEnding - 1; // hourEnding 1-24 -> index 0-23
      if (hour >= 0 && hour < 24) {
        chartData[hour].damPred = Math.round(pred.predictedPrice * 100) / 100;
      }
    }

    return NextResponse.json({
      date,
      settlementPoint: CHART_SETTLEMENT_POINT,
      data: chartData,
      lastUpdated: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error fetching chart data:", error);
    return NextResponse.json(
      { error: "Failed to fetch chart data" },
      { status: 500 }
    );
  }
}
