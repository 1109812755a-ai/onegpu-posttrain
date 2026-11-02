import test from "node:test";
import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { datasetImportError, datasetImportUrl, MAX_DATASET_IMPORT_BYTES, postDatasetImport } from "../src/datasetImport.mjs";

const workbenchSource = await readFile(new URL("../src/DatasetWorkbench.tsx", import.meta.url), "utf8");
const appSource = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");

test("import URL keeps the declared stage and encoded provenance distinct", () => {
  const url = new URL(datasetImportUrl({
    stage: "preference", sourceName: "团队数据 & 验证", sourceUri: "https://example.test/data?a=1&b=2",
    revision: "rev-42", revisionStatus: "known", licenseId: "CC-BY-4.0",
    licenseStatus: "user_declared_unverified", attribution: "Research group",
  }), "http://localhost");
  assert.equal(url.searchParams.get("stage"), "preference");
  assert.equal(url.searchParams.get("source_name"), "团队数据 & 验证");
  assert.equal(url.searchParams.get("source_uri"), "https://example.test/data?a=1&b=2");
  assert.equal(url.searchParams.get("license_status"), "user_declared_unverified");
  assert.equal(MAX_DATASET_IMPORT_BYTES, 256 * 1024 * 1024);
});

test("bounded validation errors are readable and never require raw record echo", () => {
  assert.equal(datasetImportError({ detail: { errors: [{ line: 3, field: "messages", message: "schema invalid" }] } }, 422),
    "第 3 行 · messages：schema invalid");
  assert.equal(datasetImportError({ detail: "上传超过上限" }, 413), "上传超过上限");
});

test("cancel aborts the active upload request without clearing caller-owned form context", async () => {
  const controller = new AbortController();
  const file = { name: "stage.jsonl" };
  const metadata = { stage: "sft", sourceName: "retained source" };
  const pending = postDatasetImport({
    url: datasetImportUrl({ ...metadata, revisionStatus: "not_provided", licenseStatus: "not_provided" }),
    file, signal: controller.signal,
    fetchImpl: (_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    }),
  });
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
  assert.deepEqual(metadata, { stage: "sft", sourceName: "retained source" });
  assert.equal(file.name, "stage.jsonl");
});

test("retry reuses the selected file and provenance after a recoverable server error", async () => {
  const file = new Blob(["{}\n"], { type: "application/x-ndjson" });
  const metadata = { stage: "sft", sourceName: "same source", revisionStatus: "unknown", licenseStatus: "unknown" };
  const url = datasetImportUrl(metadata);
  const requests = [];
  const fetchImpl = async (requestUrl, options) => {
    requests.push({ requestUrl, body: options.body });
    return requests.length === 1
      ? { ok: false, status: 503, json: async () => ({ detail: "本地服务暂不可用" }) }
      : { ok: true, status: 201, json: async () => ({ dataset_id: "ds-sft-fixture" }) };
  };
  await assert.rejects(postDatasetImport({ fetchImpl, url, file }), /本地服务暂不可用/);
  const result = await postDatasetImport({ fetchImpl, url, file });
  assert.equal(result.dataset_id, "ds-sft-fixture");
  assert.equal(requests.length, 2);
  assert.ok(requests.every((request) => request.requestUrl === url && request.body === file));
  assert.equal(metadata.sourceName, "same source");
});

test("catalog and source capability errors remain local, and quality metrics show Chinese names with denominator/status", () => {
  assert.match(workbenchSource, /loadDataWorkbenchResources/);
  assert.match(workbenchSource, /数据目录暂不可用/);
  assert.match(workbenchSource, /QUALITY_NAMES/);
  assert.match(workbenchSource, /实际分母/);
  assert.match(workbenchSource, /QUALITY_STATUS/);
  assert.match(appSource, /page === "data" && Boolean\(error\)/);
  assert.match(appSource, /数据报告暂不可用/);
});
