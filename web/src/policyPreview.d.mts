export function isPreviewStale(hasBothResults: boolean, previewFingerprint: string | undefined, currentFingerprint: string): boolean;
export function createPreviewFingerprint<TPolicy>(sourceId: string, stage: string, policyA: TPolicy, policyB: TPolicy, rows: number): string;
export function canApplyPreviewResponse(requestId: number, latestRequestId: number, requestFingerprint: string, currentFingerprint: string): boolean;
export function previewPolicyPair<TPolicy, TResult>(previewOne: (policy: TPolicy) => Promise<TResult>, policies: readonly [TPolicy, TPolicy]): Promise<{ a: TResult; b: TResult }>;
