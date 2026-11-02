export async function fetchDatasetPreview(datasetId, { fetchImpl = fetch, limit = 3, signal } = {}) {
  const path = `/api/data/datasets/${encodeURIComponent(datasetId)}/preview?limit=${encodeURIComponent(limit)}`;
  const response = await fetchImpl(path, { signal, headers: { Accept: "application/json" } });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.message;
    const error = new Error(message || `样本预览失败 (${response.status})`);
    error.code = typeof detail === "object" ? detail?.code : undefined;
    error.status = response.status;
    throw error;
  }
  return payload;
}

export function datasetPreviewReducer(state, action) {
  switch (action.type) {
    case "loading":
      return { ...state, [action.datasetId]: { status: "loading" } };
    case "success":
      return { ...state, [action.datasetId]: { status: "success", data: action.data } };
    case "error":
      return { ...state, [action.datasetId]: { status: "error", message: action.message } };
    case "close": {
      const next = { ...state };
      delete next[action.datasetId];
      return next;
    }
    default:
      return state;
  }
}
