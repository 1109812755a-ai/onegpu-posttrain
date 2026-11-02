export async function loadJsonResource(
  url,
  signal,
  onData,
  onError,
  request = fetch,
) {
  try {
    const response = await request(url, { signal });
    if (!response.ok) {
      const error = new Error(String(response.status));
      error.status = response.status;
      throw error;
    }
    const data = await response.json();
    if (!signal.aborted) onData(data);
  } catch (error) {
    if (signal.aborted || error?.name === "AbortError") return;
    onError(error instanceof Error ? error : new Error(String(error)));
  }
}

export function classifyOverviewLoadError(error) {
  if (error?.status === 404 || error?.message === "404") {
    return { kind: "missing" };
  }
  return { kind: "error", message: error?.message ?? String(error) };
}
