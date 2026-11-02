import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const css = await readFile(new URL("../src/style.css", import.meta.url), "utf8");
const component = await readFile(new URL("../src/DatasetWorkbench.tsx", import.meta.url), "utf8");
const importApi = await readFile(new URL("../src/datasetImport.mjs", import.meta.url), "utf8");

test("source-format import actions remain usable on narrow screens", () => {
  assert.match(css, /@media\s*\(max-width:\s*560px\)\s*\{[^}]*\.import-actions\s*\{\s*flex-direction:\s*column/s);
  assert.match(css, /@media\s*\(max-width:\s*560px\)/);
  assert.match(css, /\.import-actions\s*>\s*button\s*\{\s*width:\s*100%/);
  assert.match(css, /@media\s*\(max-width:\s*680px\)/);
  assert.match(css, /\.import-steps\s*\{[^}]*grid-template-columns:\s*1fr/s);
  assert.match(component, /role="status"/);
  assert.match(component, /role="alert"/);
});

test("import progression keeps explicit preview, source declaration, and confirmation gates", () => {
  assert.match(component, /importPreview\?\.valid !== true/);
  assert.match(importApi, /source_format/);
  assert.match(component, /确认来源声明并登记数据集/);
  assert.match(component, /不会把许可证状态升级为已核验/);
});
