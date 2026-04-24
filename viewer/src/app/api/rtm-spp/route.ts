import { NextRequest, NextResponse } from "next/server";
import { queryRtmLmpData, queryRtmPredictions } from "@/lib/influxdb";
import { SETTLEMENT_POINTS } from "@/lib/constants";
import { pivotRtmData, getToday } from "@/lib/utils";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get("date") || getToday();

    // Fetch actual prices and predictions in parallel
    const [records, predictionRecords] = await Promise.all([
      queryRtmLmpData(date, SETTLEMENT_POINTS),
      queryRtmPredictions(date, SETTLEMENT_POINTS),
    ]);

    const pivoted = pivotRtmData(records, SETTLEMENT_POINTS);

    // Build predictions map: time (HH:00) -> { settlementPoint -> predictedPrice }
    const predictionsMap = new Map<string, Map<string, number>>();
    for (const pred of predictionRecords) {
      const timeKey = `${String(pred.hourEnding).padStart(2, "0")}:00`;
      if (!predictionsMap.has(timeKey)) {
        predictionsMap.set(timeKey, new Map());
      }
      predictionsMap.get(timeKey)!.set(pred.settlementPoint, pred.predictedPrice);
    }

    // Merge predictions into pivoted data
    const dataWithPredictions = pivoted.map((row) => {
      const timePreds = predictionsMap.get(row.time);
      const predictions: Record<string, number | null> = {};

      for (const point of SETTLEMENT_POINTS) {
        predictions[point] = timePreds?.get(point) ?? null;
      }

      return {
        ...row,
        predictions,
      };
    });

    return NextResponse.json({
      date,
      settlementPoints: SETTLEMENT_POINTS,
      data: dataWithPredictions,
      lastUpdated: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error fetching RTM SPP data:", error);
    return NextResponse.json(
      { error: "Failed to fetch RTM SPP data" },
      { status: 500 }
    );
  }
}
