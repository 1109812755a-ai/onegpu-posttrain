import { useEffect, useState } from "react";
import type { EvaluationMatrix, EvaluationMatrixIndexItem } from "./types";
import { EvaluationMatrixResults } from "./EvaluationMatrixResults";

async function getJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`本地 API 返回 HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export function MatrixLoadingState() {
  return <section className="panel comparison-state" role="status" aria-live="polite">正在读取本地评测矩阵…</section>;
}

export function MatrixErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <div className="comparison-alert" role="alert"><span>{message}</span><button onClick={onRetry}>重试</button></div>;
}

export function MatrixEmptyState({ onBackToData }: { onBackToData: () => void }) {
  return <section className="panel comparison-state">
    <strong>尚无评测矩阵</strong><p>完成并登记统一评测后，矩阵会出现在这里。此页面不会自动创建或启动评测。</p>
    <button className="button secondary" onClick={onBackToData}>返回数据页</button>
  </section>;
}

export function EvaluationMatrixPage({ matrixId, onMatrixChange, onBackToData }: {
  matrixId: string;
  onMatrixChange: (id: string) => void;
  onBackToData: () => void;
}) {
  const [items, setItems] = useState<EvaluationMatrixIndexItem[] | null>(null);
  const [matrix, setMatrix] = useState<EvaluationMatrix | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setItems(null);
    setMatrix(null);
    setError("");
    getJson<{ matrices: EvaluationMatrixIndexItem[] }>("/api/evaluation-matrices", controller.signal)
      .then((index) => {
        setItems(index.matrices);
        const preferred = index.matrices.find((item) => item.matrix_id === matrixId && item.status === "available")
          || index.matrices.find((item) => item.status === "available");
        if (!preferred) return;
        if (preferred.matrix_id !== matrixId) onMatrixChange(preferred.matrix_id);
        return getJson<EvaluationMatrix>(`/api/evaluation-matrices/${preferred.matrix_id}`, controller.signal).then(setMatrix);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "评测矩阵读取失败");
      });
    return () => controller.abort();
  }, [matrixId, refresh, onMatrixChange]);

  const available = items?.filter((item) => item.status === "available") || [];
  const selectedId = matrix?.matrix_id || matrixId;

  return (
    <div className="comparison-page-content">
      <section className="comparison-intro">
        <div>
          <div className="eyebrow">MODEL EVALUATION <span>／</span> 证据对照</div>
          <h1>阶段比较<span>。</span></h1>
          <p>查看同协议评测结果、样本覆盖和来源证据。这里只读已有产物，不会启动训练或评测。</p>
        </div>
        <button className="button secondary" onClick={() => setRefresh((value) => value + 1)} disabled={items === null}>
          重新读取
        </button>
      </section>

      {error && <MatrixErrorState message={error} onRetry={() => setRefresh((value) => value + 1)} />}
      {items === null && !error && <MatrixLoadingState />}
      {items && items.length === 0 && <MatrixEmptyState onBackToData={onBackToData} />}
      {items && items.some((item) => item.status === "invalid") && <div className="comparison-alert" role="status">
        <span>有 {items.filter((item) => item.status === "invalid").length} 份矩阵无法读取；未展示其内容，也不会把缺失数据当作 0。</span>
        <button onClick={() => setRefresh((value) => value + 1)}>重试读取</button>
      </div>}
      {available.length > 1 && <label className="comparison-selector">评测矩阵
        <select value={selectedId} onChange={(event) => onMatrixChange(event.target.value)} aria-label="选择评测矩阵">
          {available.map((item) => <option value={item.matrix_id} key={item.matrix_id}>{item.name} · {item.model_count ?? 0} 个模型</option>)}
        </select>
      </label>}
      {matrix && <EvaluationMatrixResults matrix={matrix} />}
    </div>
  );
}
