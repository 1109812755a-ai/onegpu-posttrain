async function readJson(fetchImpl, url, signal, label) {
  const response = await fetchImpl(url, { signal });
  if (!response.ok) throw new Error(`${label} (${response.status})`);
  return response.json();
}

export async function loadDataWorkbenchResources(fetchImpl = fetch, signal) {
  const [catalog, capabilities] = await Promise.allSettled([
    readJson(fetchImpl, "/api/data", signal, "数据目录读取失败"),
    readJson(fetchImpl, "/api/data/capabilities", signal, "来源能力读取失败"),
  ]);
  return { catalog, capabilities };
}
