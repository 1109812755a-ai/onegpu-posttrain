export const MAX_DATASET_IMPORT_BYTES = 256 * 1024 * 1024;

export function datasetImportUrl(metadata) {
  const params = new URLSearchParams({
    stage: metadata.stage,
    source_format: metadata.sourceFormat || "canonical-v1",
    source_name: metadata.sourceName || "本地上传",
    revision_status: metadata.revisionStatus,
    license_status: metadata.licenseStatus,
  });
  for (const [key, value] of [
    ["source_uri", metadata.sourceUri], ["revision", metadata.revision],
    ["license_id", metadata.licenseId], ["attribution", metadata.attribution],
  ]) if (value?.trim()) params.set(key, value.trim());
  return `/api/data/import?${params.toString()}`;
}

export async function previewDatasetImport({ fetchImpl = fetch, stage, sourceFormat, file, signal }) {
  const query = new URLSearchParams({ stage, source_format: sourceFormat });
  const response = await fetchImpl(`/api/data/import/preview?${query}`, {
    method: "POST", headers: { "content-type": "application/x-ndjson" }, body: file, signal,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(datasetImportError(payload, response.status));
  return payload;
}

export function datasetImportError(payload, status) {
  const detail = payload?.detail;
  if (detail && typeof detail === "object" && Array.isArray(detail.errors)) {
    return detail.errors.map((item) => `第 ${item.line} 行${item.field ? ` · ${item.field}` : ""}：${item.message}`).join("；");
  }
  return String(detail || `登记失败 (${status})`);
}

export async function postDatasetImport({ fetchImpl = fetch, url, file, signal }) {
  const response = await fetchImpl(url, {
    method: "POST", headers: { "content-type": "application/x-ndjson" }, body: file, signal,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(datasetImportError(payload, response.status));
  return payload;
}
