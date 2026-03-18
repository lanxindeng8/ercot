export async function fetchPredictionJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, init);

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    const contentType = response.headers.get("content-type") ?? "";

    try {
      if (contentType.includes("application/json")) {
        const body = (await response.json()) as { detail?: unknown };
        if (typeof body.detail === "string" && body.detail.trim()) {
          detail = body.detail;
        }
      } else {
        const text = await response.text();
        if (text.trim()) {
          detail = text.trim();
        }
      }
    } catch {
      // Keep the generic HTTP error when the response body is malformed.
    }

    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
