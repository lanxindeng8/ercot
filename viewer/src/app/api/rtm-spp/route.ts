import { NextRequest, NextResponse } from "next/server";
import { queryRtmLmpData } from "@/lib/influxdb";
import { SETTLEMENT_POINTS } from "@/lib/constants";
import { pivotRtmData, getToday } from "@/lib/utils";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get("date") || getToday();

    const records = await queryRtmLmpData(date, SETTLEMENT_POINTS);
    const pivoted = pivotRtmData(records, SETTLEMENT_POINTS);

    return NextResponse.json({
      date,
      settlementPoints: SETTLEMENT_POINTS,
      data: pivoted,
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
