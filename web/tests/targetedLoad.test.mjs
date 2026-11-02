import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyOverviewLoadError,
  loadJsonResource,
} from "../src/targetedLoad.mjs";

function response(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

function capture() {
  return { data: undefined, error: undefined };
}

test("overview 404 does not prevent capabilities from loading", async () => {
  const overview = capture();
  const capabilities = capture();
  const controller = new AbortController();
  const request = async (url) =>
    url.endsWith("overview")
      ? response(404, {})
      : response(200, { profiles: [{ profile_id: "fake" }] });

  await Promise.all([
    loadJsonResource("/api/targeted/overview", controller.signal,
      (data) => { overview.data = data; }, (error) => { overview.error = error; }, request),
    loadJsonResource("/api/targeted/capabilities", controller.signal,
      (data) => { capabilities.data = data; }, (error) => { capabilities.error = error; }, request),
  ]);

  assert.equal(overview.data, undefined);
  assert.equal(overview.error.message, "404");
  assert.deepEqual(capabilities.data, { profiles: [{ profile_id: "fake" }] });
  assert.equal(capabilities.error, undefined);
});

test("capabilities 404 does not prevent the overview from loading", async () => {
  const overview = capture();
  const capabilities = capture();
  const controller = new AbortController();
  const request = async (url) =>
    url.endsWith("overview")
      ? response(200, { available: true, scenarios: [{ scenario_id: "algebra" }] })
      : response(404, {});

  await Promise.all([
    loadJsonResource("/api/targeted/overview", controller.signal,
      (data) => { overview.data = data; }, (error) => { overview.error = error; }, request),
    loadJsonResource("/api/targeted/capabilities", controller.signal,
      (data) => { capabilities.data = data; }, (error) => { capabilities.error = error; }, request),
  ]);

  assert.deepEqual(overview.data, { available: true, scenarios: [{ scenario_id: "algebra" }] });
  assert.equal(overview.error, undefined);
  assert.equal(capabilities.data, undefined);
  assert.equal(capabilities.error.message, "404");
});

test("both resources load when both endpoints succeed", async () => {
  const overview = capture();
  const capabilities = capture();
  const controller = new AbortController();
  const request = async (url) => response(200, { path: url });

  await Promise.all([
    loadJsonResource("/api/targeted/overview", controller.signal,
      (data) => { overview.data = data; }, (error) => { overview.error = error; }, request),
    loadJsonResource("/api/targeted/capabilities", controller.signal,
      (data) => { capabilities.data = data; }, (error) => { capabilities.error = error; }, request),
  ]);

  assert.deepEqual(overview.data, { path: "/api/targeted/overview" });
  assert.deepEqual(capabilities.data, { path: "/api/targeted/capabilities" });
  assert.equal(overview.error, undefined);
  assert.equal(capabilities.error, undefined);
});

test("overview 404 is classified as a missing report, not a read failure", async () => {
  const controller = new AbortController();
  const result = capture();

  await loadJsonResource(
    "/api/targeted/overview",
    controller.signal,
    (data) => { result.data = data; },
    (error) => { result.error = error; },
    async () => response(404, {}),
  );

  assert.equal(result.error.status, 404);
  assert.deepEqual(classifyOverviewLoadError(result.error), { kind: "missing" });
});

test("other overview errors are retryable read failures", async () => {
  const controller = new AbortController();
  const result = capture();
  let attempts = 0;
  const request = async () => {
    attempts += 1;
    return attempts === 1
      ? response(503, {})
      : response(200, { available: false, scenarios: [] });
  };

  await loadJsonResource(
    "/api/targeted/overview",
    controller.signal,
    (data) => { result.data = data; },
    (error) => { result.error = error; },
    request,
  );
  assert.deepEqual(classifyOverviewLoadError(result.error), {
    kind: "error",
    message: "503",
  });

  result.error = undefined;
  await loadJsonResource(
    "/api/targeted/overview",
    controller.signal,
    (data) => { result.data = data; },
    (error) => { result.error = error; },
    request,
  );
  assert.equal(attempts, 2);
  assert.deepEqual(result.data, { available: false, scenarios: [] });
  assert.equal(result.error, undefined);
});
