export function isPreviewStale(hasBothResults, previewFingerprint, currentFingerprint) {
  return !hasBothResults || previewFingerprint !== currentFingerprint;
}

export function createPreviewFingerprint(sourceId, stage, policyA, policyB, rows) {
  return JSON.stringify({ sourceId, stage, policyA, policyB, rows });
}

export function canApplyPreviewResponse(requestId, latestRequestId, requestFingerprint, currentFingerprint) {
  return requestId === latestRequestId && requestFingerprint === currentFingerprint;
}

export async function previewPolicyPair(previewOne, policies) {
  const results = [];
  for (const policy of policies) results.push(await previewOne(policy));
  return { a: results[0], b: results[1] };
}
