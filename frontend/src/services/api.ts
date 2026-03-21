/**
 * API service — talks to the FastAPI backend.
 */

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

/* ── Types ──────────────────────────────────────────────────────────────── */

export interface SummaryLevel {
  value: number;
  name: string;
  emoji: string;
  description: string;
}

export interface SummaryResponse {
  summary: string;
  word_count: number;
  level_name: string;
  level_emoji: string;
  filename: string;
}

/* ── Endpoints ──────────────────────────────────────────────────────────── */

/**
 * Fetch available summary levels from the backend.
 */
export async function fetchSummaryLevels(): Promise<SummaryLevel[]> {
  const res = await fetch(`${API_BASE}/api/summary-levels`);
  if (!res.ok) throw new Error("Failed to load summary levels");
  return res.json();
}

/**
 * Upload a PDF and get a summary back.
 */
export async function summarizePdf(
  file: File,
  level: number,
): Promise<SummaryResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("level", String(level));

  const res = await fetch(`${API_BASE}/api/summarize`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(body.detail ?? `Server error ${res.status}`);
  }

  return res.json();
}

/**
 * Generate a downloadable summary PDF.
 */
export async function downloadSummaryPdf(
  summary: string,
  levelName: string,
  filename: string,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/download-pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      summary,
      level_name: levelName,
      filename,
    }),
  });

  if (!res.ok) throw new Error("Failed to generate PDF");

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;

  // Extract filename from Content-Disposition or build one
  const cd = res.headers.get("Content-Disposition");
  const match = cd?.match(/filename="?([^"]+)"?/);
  a.download = match?.[1] ?? `${filename}_summary.pdf`;

  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
