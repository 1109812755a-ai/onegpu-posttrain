export interface Artifact {
  id: string;
  path: string;
  label: string;
}
export interface Metric {
  value: number | null;
  unit: string;
  status: string;
  class: string;
  method: string;
  numerator: number | null;
  denominator: number | null;
  limitations: string[];
}
export interface Dataset {
  stage: string;
  dataset_id: string;
  display_name?: string | null;
  count: number;
  target: number | null;
  unit: string;
  tokens: number | null;
  token_target: number | null;
  content_sha256?: string | null;
  mixture_eligible?: boolean;
  mixture_block_reason?: string | null;
  buildId?: string;
  buildName?: string;
  buildPath?: string;
  quality_status?: string | null;
  source?: {
    source_id?: string | null;
    kind: string;
    name?: string | null;
    revision?: string | null;
    revision_status?: string | null;
    license_status?: string | null;
  };
  quality?: {
    metrics: Record<string, Metric>;
    gate: {
      summary_status: string;
      checks: { check_id: string; result: string }[];
    };
    limitations: string[];
  } | null;
  lengths?: Record<string, Record<string, number | null>>;
  source_distribution?: Record<string, number>;
  manifest?: Artifact;
  quality_artifact?: Artifact | null;
}
export interface Step {
  step_id: string;
  order: number;
  operation: string;
  operation_version: string;
  status: string;
  input_count: { value: number | null; unit: string };
  output_count: { value: number | null; unit: string };
  rejected_count: number | null;
  unselected_count: number | null;
  first_rejection_reasons: Record<string, number>;
  unselected_reasons: Record<string, number>;
  metrics_before: Record<string, Metric>;
  metrics_after: Record<string, Metric>;
  limitations: string[];
  details: Record<string, unknown>;
}
export interface BuildSummary {
  id: string;
  name: string;
  path: string;
  source: string;
  revision: string | null;
  mode: string;
  is_demo: boolean;
  raw_count: number | null;
  raw_unit: string;
  created_at: string | null;
  warnings: string[];
  datasets: Dataset[];
}
export interface BuildDetail extends BuildSummary {
  steps: Step[];
  artifacts: Artifact[];
  overlap: Record<string, number>;
  model_effect: { status: string; reason: string };
  related_runs: {
    run_id: string;
    stage: string;
    status: string;
    source: string;
    note: string;
    metrics: {
      name: string;
      value: number | null;
      unit: string;
      samples: number | null;
    }[];
  }[];
}
export interface Catalog {
  builds: BuildSummary[];
  issues: { path: string; message: string }[];
  scanned_at: string;
  read_only: boolean;
  roots: string[];
}

export interface EvaluationMatrixIndexItem {
  matrix_id: string;
  name: string;
  status: "available" | "invalid";
  issue?: string;
  protocol?: string | null;
  baseline_model_id?: string | null;
  model_count?: number;
  comparability_status?: string | null;
}

export interface EvaluationMatrixMetric {
  value: number | null;
  unit: string;
  status: string;
  requested: number | null;
  observed: number | null;
  delta: number | null;
}

export interface EvaluationMatrix {
  matrix_id: string;
  name: string;
  protocol: string | null;
  baseline_model_id: string | null;
  expected_metrics: string[];
  requested_limits: Record<string, number | null>;
  comparability: {
    status: string;
    expected_protocol_version: string | null;
    protocol_versions: string[];
    config_hashes: string[];
    dataset_id_sets: string[][];
    protocol_version_consistent: boolean | null;
    config_hash_consistent: boolean | null;
    dataset_ids_consistent: boolean | null;
    limitations: string[];
  };
  models: {
    model_id: string | null;
    status: string;
    provenance: { status: string; issues: { code: string; message: string }[] };
    metrics: Record<string, EvaluationMatrixMetric>;
    samples: Record<string, { requested: number | null; observed: number | null }>;
    lineage: {
      eval_id: string | null;
      evaluation_run_id: string | null;
      model_artifact_id: string | null;
      dataset_ids: string[];
      protocol_version: string | null;
      config_sha256: string | null;
      training_resolution: {
        resolved: boolean;
        run_id?: string;
        run_type?: string | null;
        status?: string | null;
        parent_model_ids?: string[];
        dataset_ids?: string[];
      };
      evaluation_resolution: {
        resolved: boolean;
        run_id?: string | null;
        run_status?: string | null;
        eval_id?: string | null;
        model_id?: string | null;
        dataset_ids?: string[];
        protocol_version?: string | null;
        config_sha256?: string | null;
      };
    };
    errors_count: number;
  }[];
  truncated_models: boolean;
}
