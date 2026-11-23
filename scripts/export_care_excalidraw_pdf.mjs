// Print the official Excalidraw SVG through Chrome so the paper receives
// a vector PDF with the embedded Excalifont glyphs.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("../build/excalidraw-cli/node_modules/playwright-core");
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const svgPath = resolve(root, "paper_asem", "figures",
  "care_architecture_excalidraw.svg");
const pdfPath = resolve(root, "paper_asem", "figures",
  "care_architecture_excalidraw.pdf");
const svg = readFileSync(svgPath, "utf8").replace(/^<\?xml[^>]*>\s*/,"");
const browser = await chromium.launch({
  headless: true,
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
});
try {
  const page = await browser.newPage({ viewport: { width: 1775, height: 985 } });
  await page.setContent(
    "<!doctype html><html><head><style>"+
    "@page{size:1775px 985px;margin:0}"+
    "html,body{width:1775px;height:985px;margin:0;background:#fffefa}"+
    "svg{display:block;width:1775px;height:985px}"+
    "</style></head><body>"+svg+"</body></html>",
    { waitUntil: "load" },
  );
  await page.evaluate(async () => { await document.fonts.ready; });
  await page.pdf({
    path: pdfPath,
    width: "1775px",
    height: "985px",
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    printBackground: true,
    preferCSSPageSize: true,
  });
  console.log("Wrote "+pdfPath);
} finally {
  await browser.close();
}
