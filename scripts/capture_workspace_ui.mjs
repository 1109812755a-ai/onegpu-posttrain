// Read-only browser capture of local workspace pages. Never clicks task actions.
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const edge = process.env.ONEGPU_EDGE_PATH || "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const origin = process.env.ONEGPU_UI_ORIGIN || "http://127.0.0.1:8769";
const outDir = resolve(process.argv[2] || "docs/reports");
const port = Number(process.env.ONEGPU_CDP_PORT || 19224);
const profile = mkdtempSync(join(tmpdir(), "onegpu-ux-capture-"));
const browser = spawn(edge, ["--headless=new", "--disable-gpu", "--no-first-run", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, "about:blank"], { windowsHide: true, stdio: "ignore" });

const pause = (ms) => new Promise((resolvePause) => setTimeout(resolvePause, ms));
let socket;
let sequence = 0;
const pending = new Map();
function command(method, params = {}) {
  const id = ++sequence;
  return new Promise((resolveCommand, rejectCommand) => {
    pending.set(id, { resolveCommand, rejectCommand });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

try {
  let target;
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      target = tabs.find((tab) => tab.type === "page");
      if (target?.webSocketDebuggerUrl) break;
    } catch { /* Edge has not started yet. */ }
    await pause(200);
  }
  if (!target?.webSocketDebuggerUrl) throw new Error("Edge CDP did not become ready");
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolveOpen, rejectOpen) => {
    socket.onopen = resolveOpen;
    socket.onerror = rejectOpen;
  });
  socket.onmessage = ({ data }) => {
    const response = JSON.parse(data);
    if (!response.id || !pending.has(response.id)) return;
    const { resolveCommand, rejectCommand } = pending.get(response.id);
    pending.delete(response.id);
    if (response.error) rejectCommand(new Error(response.error.message)); else resolveCommand(response.result);
  };
  await command("Page.enable");
  await command("Runtime.enable");
  mkdirSync(outDir, { recursive: true });
  for (const [name, page] of [["data", "workbench"], ["launcher", "launcher"], ["runs", "runs"], ["comparison", "comparison"]]) {
    await command("Page.navigate", { url: `${origin}/?page=${page}` });
    await pause(1400);
    for (const theme of ["light", "dark"]) {
      await command("Runtime.evaluate", { expression: `localStorage.setItem('onegpu-posttrain-theme-v1', ${JSON.stringify(theme)})` });
      await command("Page.reload", { ignoreCache: true });
      await pause(1600);
      for (const [viewport, width, height] of [["desktop", 1440, 900], ["mobile", 390, 844]]) {
        await command("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: viewport === "mobile" });
        await pause(250);
        const result = await command("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
        const path = join(outDir, `workspace-ux-${name}-${viewport}-${theme}.png`);
        writeFileSync(path, Buffer.from(result.data, "base64"));
        process.stdout.write(`${path}\n`);
      }
    }
  }
} finally {
  socket?.close();
  browser.kill();
  // The isolated browser profile is left in the OS temp directory; no workspace data is removed.
}
