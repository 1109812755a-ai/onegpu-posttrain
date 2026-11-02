export interface PreviewSample {
  id: string;
  fields: Record<string, unknown>;
  truncated_fields: string[];
}

export interface DatasetPreview {
  available: true;
  dataset_id: string;
  stage: string;
  canonical_sha256: string;
  integrity: "verified";
  total_records: number;
  returned_records: number;
  truncated: boolean;
  samples: PreviewSample[];
}

export type DatasetPreviewState = Record<string, {
  status: "loading" | "success" | "error";
  data?: DatasetPreview;
  message?: string;
}>;

export type DatasetPreviewAction =
  | { type: "loading"; datasetId: string }
  | { type: "success"; datasetId: string; data: DatasetPreview }
  | { type: "error"; datasetId: string; message: string }
  | { type: "close"; datasetId: string };

export function fetchDatasetPreview(
  datasetId: string,
  options?: { fetchImpl?: typeof fetch; limit?: number; signal?: AbortSignal },
): Promise<DatasetPreview>;

export function datasetPreviewReducer(state: DatasetPreviewState, action: DatasetPreviewAction): DatasetPreviewState;
