import { useState } from "react";

export function ModelDisplayNameEditor({ modelId, displayName, token, onSaved, compact = false }: {
  modelId: string; displayName?: string | null; token: string;
  onSaved: (displayName: string | null) => void; compact?: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(displayName || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const inputId = `model-display-name-${modelId}`;

  async function save() {
    const name = value.trim();
    if (!name) { setError("显示名不能为空"); return; }
    if (name.length > 80) { setError("显示名不能超过 80 个字符"); return; }
    if (/[\u0000-\u001f\u007f-\u009f]/.test(name)) { setError("显示名不能包含控制字符或换行"); return; }
    setBusy(true); setError(""); setStatus("");
    try {
      const response = await fetch(`/api/model-playground/models/${encodeURIComponent(modelId)}/display-name`, {
        method: "PUT", headers: { "Content-Type": "application/json", "X-Local-Launcher-Token": token },
        body: JSON.stringify({ display_name: name }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `本地服务返回 ${response.status}`);
      onSaved(payload.display_name); setValue(payload.display_name); setEditing(false); setStatus("显示名已保存");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "保存显示名失败"); }
    finally { setBusy(false); }
  }

  async function clear() {
    setBusy(true); setError(""); setStatus("");
    try {
      const response = await fetch(`/api/model-playground/models/${encodeURIComponent(modelId)}/display-name`, {
        method: "DELETE", headers: { "X-Local-Launcher-Token": token },
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || `本地服务返回 ${response.status}`);
      }
      onSaved(null); setValue(""); setEditing(false); setStatus("已清除显示名");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "清除显示名失败"); }
    finally { setBusy(false); }
  }

  return <div className={`model-display-name-editor${compact ? " compact" : ""}`}>
    {!editing ? <div className="model-display-name-actions">
      <button className="button secondary" type="button" onClick={() => { setValue(displayName || ""); setEditing(true); setError(""); }} disabled={busy}>
        {displayName ? "修改用户显示名" : "设置用户显示名"}
      </button>
      {displayName && <button className="text-button" type="button" onClick={clear} disabled={busy}>清除</button>}
    </div> : <div className="model-display-name-form">
      <label htmlFor={inputId}>用户显示名<input id={inputId} autoFocus maxLength={80} value={value}
        onChange={(event) => setValue(event.target.value)} disabled={busy} aria-invalid={Boolean(error)} /></label>
      <button className="button primary" type="button" onClick={save} disabled={busy}>{busy ? "保存中…" : "保存名称"}</button>
      <button className="button secondary" type="button" onClick={() => { setEditing(false); setError(""); }} disabled={busy}>取消</button>
    </div>}
    {error && <p className="notice danger" role="alert">{error}</p>}
    {status && <span role="status">{status}</span>}
  </div>;
}
