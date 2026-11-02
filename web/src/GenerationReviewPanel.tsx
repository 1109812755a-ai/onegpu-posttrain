type Preview = {
  records: unknown[];
  count?: number;
  total_records?: number;
  canonical_sha256?: string;
  truncated?: boolean;
};

export function GenerationReviewPanel({
  label,
  canReview,
  busy,
  preview,
  reviewed,
  registeredDatasetId,
  registrationSummary,
  onPreview,
  onReviewedChange,
  onRegister,
}: {
  label: string;
  canReview: boolean;
  busy: boolean;
  preview: Preview | null;
  reviewed: boolean;
  registeredDatasetId?: string;
  registrationSummary?: string;
  onPreview: () => void;
  onReviewedChange: (value: boolean) => void;
  onRegister: () => void;
}) {
  if (!canReview && !registeredDatasetId) return null;

  return <section className="generation-review-panel" aria-label={`${label}候选审核与登记`}>
    {registeredDatasetId ? <div className="dataset-import-success" role="status">
      <div className="success-mark" aria-hidden="true">✓</div>
      <div><strong>数据集已登记</strong><p><code>{registeredDatasetId}</code>{registrationSummary ? ` · ${registrationSummary}` : ""}</p></div>
    </div> : <>
      <div className="import-actions">
        <button className="secondary-button" disabled={busy} onClick={onPreview}>{preview ? "刷新候选预览" : "审核候选样本"}</button>
      </div>
      {preview && <div className="generation-review-preview">
        <strong>样本审核 · {preview.count ?? preview.records.length} 条{preview.canonical_sha256 ? ` · SHA256 ${preview.canonical_sha256}` : ""}</strong>
        <p>请检查内容和格式。完整性核验不代表语义正确性或来源许可已独立验证。</p>
        <div className="dataset-preview-samples">{preview.records.map((record, index) => <article className="dataset-preview-sample" key={`${index}-${typeof record === "object" && record && "id" in record ? String(record.id) : "record"}`}>
          <pre>{JSON.stringify(record, null, 2)}</pre>
        </article>)}</div>
        {preview.truncated && <small>仅显示 {preview.records.length} 条候选样本。</small>}
        <label className="sft-review-check"><input type="checkbox" checked={reviewed} onChange={(event) => onReviewedChange(event.target.checked)} />我已审核预览内容，理解其语义正确性尚未独立验证</label>
        <button className="primary-button" disabled={busy || !reviewed} onClick={onRegister}>审核通过并登记数据集</button>
      </div>}
    </>}
  </section>;
}
