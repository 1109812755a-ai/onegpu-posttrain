export function loadJsonResource<T>(
  url: string,
  signal: AbortSignal,
  onData: (data: T) => void,
  onError: (error: Error) => void,
  request?: typeof fetch,
): Promise<void>;

export type OverviewLoadError =
  | { kind: "missing" }
  | { kind: "error"; message: string };

export function classifyOverviewLoadError(
  error: Error & { status?: number },
): OverviewLoadError;
