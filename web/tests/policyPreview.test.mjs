import assert from "node:assert/strict";
import test from "node:test";
import { canApplyPreviewResponse, createPreviewFingerprint, isPreviewStale, previewPolicyPair } from "../src/policyPreview.mjs";

test("preview is stale after any source, policy, stage or range fingerprint change", () => {
  assert.equal(isPreviewStale(true, "same-draft", "same-draft"), false);
  assert.equal(isPreviewStale(true, "old-source-and-rules", "changed-source-and-rules"), true);
  assert.equal(isPreviewStale(false, undefined, "new-draft"), true);
});

test("late old responses and responses for edited drafts cannot replace current preview", () => {
  assert.equal(canApplyPreviewResponse(1, 2, "old", "new"), false);
  assert.equal(canApplyPreviewResponse(2, 2, "old", "new"), false);
  assert.equal(canApplyPreviewResponse(2, 2, "new", "new"), true);
});

test("A/B previews run sequentially for the single-instance local preview service", async () => {
  const inFlight = new Set();
  const calls = [];
  const result = await previewPolicyPair(async (policy) => {
    assert.equal(inFlight.size, 0, "preview calls must not overlap");
    inFlight.add(policy);
    calls.push(policy);
    await new Promise((resolve) => setTimeout(resolve, 5));
    inFlight.delete(policy);
    return `${policy}-result`;
  }, ["policy-a", "policy-b"]);

  assert.deepEqual(calls, ["policy-a", "policy-b"]);
  assert.deepEqual(result, { a: "policy-a-result", b: "policy-b-result" });
});

test("preview fingerprint follows data inputs and excludes experiment-contract fields", () => {
  const a = { rules: { completeness: "filter" } };
  const b = { rules: { completeness: "report_only" } };
  const original = createPreviewFingerprint("source-1", "sft", a, b, 10);
  assert.equal(createPreviewFingerprint("source-1", "sft", a, b, 10), original);
  assert.notEqual(createPreviewFingerprint("source-2", "sft", a, b, 10), original);
  assert.notEqual(createPreviewFingerprint("source-1", "preference", a, b, 10), original);
  assert.notEqual(createPreviewFingerprint("source-1", "sft", a, b, 11), original);
  assert.notEqual(createPreviewFingerprint("source-1", "sft", a, { rules: { completeness: "filter" } }, 10), original);
});
