import { InfluxDBClient } from "@influxdata/influxdb3-client";

let client: InfluxDBClient | null = null;

function getClient(): InfluxDBClient {
  if (!client) {
    const host = process.env.INFLUXDB_URL;
    const token = process.env.INFLUXDB_TOKEN;
    const database = process.env.INFLUXDB_DATABASE;

    if (!host || !token || !database) {
      throw new Error(
        "Missing InfluxDB configuration. Set INFLUXDB_URL, INFLUXDB_TOKEN, and INFLUXDB_DATABASE."
      );
    }

    client = new InfluxDBClient({ host, token, database });
  }
  return client;
}

export interface RtmLmpRecord {
  time: Date;
  settlementPoint: string;
  lmp: number;
}

export interface DamLmpRecord {
  time: Date;
  settlementPoint: string;
  lmp: number;
}

// Convert Central Time date to UTC range
function getUtcRangeForDate(date: string): { start: string; end: string } {
  // Central Time is UTC-6 (CST) or UTC-5 (CDT)
  // For simplicity, use UTC-6 (CST) - adjust if needed for DST
  // date is in format YYYY-MM-DD, representing a day in Central Time
  // Central midnight = 06:00 UTC
  const start = `${date}T06:00:00Z`;

  // Next day at 06:00 UTC
  const nextDate = new Date(date);
  nextDate.setDate(nextDate.getDate() + 1);
  const nextDateStr = nextDate.toISOString().split("T")[0];
  const end = `${nextDateStr}T06:00:00Z`;

  return { start, end };
}

export async function queryRtmLmpData(
  date: string,
  settlementPoints: string[]
): Promise<RtmLmpRecord[]> {
  const client = getClient();
  const pointsFilter = settlementPoints.map((p) => `'${p}'`).join(", ");
  const { start, end } = getUtcRangeForDate(date);

  // Query both tables and merge results
  // rtm_lmp_api has historical data (API, ~6h delay)
  // rtm_lmp_realtime has recent data (CDR, ~5min delay)
  const recordMap = new Map<string, RtmLmpRecord>();

  // First, get API data (historical baseline)
  const apiQuery = `
    SELECT time, settlement_point, lmp
    FROM "rtm_lmp_api"
    WHERE time > '${start}'
      AND time <= '${end}'
      AND settlement_point IN (${pointsFilter})
    ORDER BY time ASC
  `;

  try {
    const result = await client.query(apiQuery);
    for await (const row of result) {
      const key = `${row.time}-${row.settlement_point}`;
      recordMap.set(key, {
        time: new Date(row.time),
        settlementPoint: row.settlement_point,
        lmp: row.lmp,
      });
    }
  } catch (error) {
    console.error("Error querying rtm_lmp_api:", error);
  }

  // Then, overlay with realtime data (newer, overwrites API data for same time)
  const realtimeQuery = `
    SELECT time, settlement_point, lmp
    FROM "rtm_lmp_realtime"
    WHERE time > '${start}'
      AND time <= '${end}'
      AND settlement_point IN (${pointsFilter})
    ORDER BY time ASC
  `;

  try {
    const result = await client.query(realtimeQuery);
    for await (const row of result) {
      const key = `${row.time}-${row.settlement_point}`;
      recordMap.set(key, {
        time: new Date(row.time),
        settlementPoint: row.settlement_point,
        lmp: row.lmp,
      });
    }
  } catch (error) {
    console.error("Error querying rtm_lmp_realtime:", error);
  }

  // Convert map to array and sort by time
  const records = Array.from(recordMap.values());
  records.sort((a, b) => a.time.getTime() - b.time.getTime());

  return records;
}

export async function queryDamLmpData(
  date: string,
  settlementPoints: string[]
): Promise<DamLmpRecord[]> {
  const client = getClient();
  const pointsFilter = settlementPoints.map((p) => `'${p}'`).join(", ");
  const { start, end } = getUtcRangeForDate(date);

  const query = `
    SELECT time, settlement_point, lmp
    FROM "dam_lmp"
    WHERE time >= '${start}'
      AND time < '${end}'
      AND settlement_point IN (${pointsFilter})
    ORDER BY time ASC
  `;

  const records: DamLmpRecord[] = [];

  try {
    const result = await client.query(query);
    for await (const row of result) {
      records.push({
        time: new Date(row.time),
        settlementPoint: row.settlement_point,
        lmp: row.lmp,
      });
    }
  } catch (error) {
    console.error("Error querying DAM LMP data:", error);
    throw error;
  }

  return records;
}

export interface DamPredictionRecord {
  time: Date;
  settlementPoint: string;
  predictedPrice: number;
  hourEnding: number;
}

export async function queryDamPredictions(
  date: string,
  settlementPoints: string[]
): Promise<DamPredictionRecord[]> {
  const client = getClient();
  const pointsFilter = settlementPoints.map((p) => `'${p}'`).join(", ");
  const { start, end } = getUtcRangeForDate(date);

  const query = `
    SELECT time, settlement_point, predicted_price, hour_ending
    FROM "dam_prediction"
    WHERE time >= '${start}'
      AND time < '${end}'
      AND settlement_point IN (${pointsFilter})
    ORDER BY time ASC
  `;

  const records: DamPredictionRecord[] = [];

  try {
    const result = await client.query(query);
    for await (const row of result) {
      records.push({
        time: new Date(row.time),
        settlementPoint: row.settlement_point,
        predictedPrice: row.predicted_price,
        hourEnding: row.hour_ending,
      });
    }
  } catch (error) {
    console.error("Error querying DAM predictions:", error);
    // Don't throw - predictions are optional
  }

  return records;
}
