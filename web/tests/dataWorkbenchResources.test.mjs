import assert from "node:assert/strict";
import test from "node:test";
import { loadDataWorkbenchResources } from "../src/dataWorkbenchResources.mjs";

function fetchFixture(results) {
  return async (url) => {
    const result = results[url];
    return { ok: result.status >= 200 && result.status < 300, status: result.status, json: async () => result.body };
  };
}

test("data catalog failure does not block source capabilities", async () => {
  const result = await loadDataWorkbenchResources(fetchFixture({
    "/api/data": { status: 503, body: { detail: "unavailable" } },
    "/api/data/capabilities": { status: 200, body: { capabilities: [{ source_id: "local_upload" }] } },
  }));
  assert.equal(result.catalog.status, "rejected");
  assert.equal(result.capabilities.status, "fulfilled");
  assert.equal(result.capabilities.value.capabilities[0].source_id, "local_upload");
});

test("source capability failure does not block the registered dataset catalog", async () => {
  const result = await loadDataWorkbenchResources(fetchFixture({
    "/api/data": { status: 200, body: { builds: [], issues: [] } },
    "/api/data/capabilities": { status: 404, body: { detail: "not enabled" } },
  }));
  assert.equal(result.catalog.status, "fulfilled");
  assert.equal(result.capabilities.status, "rejected");
  assert.deepEqual(result.catalog.value.builds, []);
});
