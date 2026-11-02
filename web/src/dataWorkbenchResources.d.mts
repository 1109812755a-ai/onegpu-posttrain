export function loadDataWorkbenchResources(
  fetchImpl?: typeof fetch,
  signal?: AbortSignal,
): Promise<{
  catalog: PromiseSettledResult<unknown>;
  capabilities: PromiseSettledResult<unknown>;
}>;
