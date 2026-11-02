export const MAX_DATASET_IMPORT_BYTES: number;
export function datasetImportUrl(metadata: {
  stage: string;
  sourceFormat?: string;
  sourceName?: string;
  sourceUri?: string;
  revision?: string;
  revisionStatus: string;
  licenseId?: string;
  licenseStatus: string;
  attribution?: string;
}): string;
export function previewDatasetImport<T = Record<string, unknown>>(args: {
  fetchImpl?: typeof fetch;
  stage: string;
  sourceFormat: string;
  file: Blob;
  signal?: AbortSignal;
}): Promise<T>;
export function datasetImportError(payload: unknown, status: number): string;
export function postDatasetImport<T = Record<string, unknown>>(args: {
  fetchImpl?: typeof fetch;
  url: string;
  file: Blob;
  signal?: AbortSignal;
}): Promise<T>;
