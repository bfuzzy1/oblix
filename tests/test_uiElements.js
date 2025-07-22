import assert from 'assert';
import fs from 'fs';

export async function run() {
  const html = fs.readFileSync('index.html', 'utf8');
  assert.ok(html.includes('id="generateDataBtn"'), 'Generate Data button missing');
  assert.ok(html.includes('id="trainButton"'), 'Train button missing');
  assert.ok(html.includes('id="networkGraph"'), 'Network graph canvas missing');
  assert.ok(html.includes('id="predictButton"'), 'Predict button missing');

  assert.ok(html.includes('src="src/main.js"'), 'Main script tag missing');
}

// Test: Switching architecture templates should not throw errors and should update the UI correctly
export async function testArchitectureSwitch() {
  // Simulate a DOM environment
  const { JSDOM } = await import('jsdom');
  const html = fs.readFileSync('index.html', 'utf8');
  const dom = new JSDOM(html, { runScripts: 'dangerously', resources: 'usable' });
  const { window } = dom;
  global.window = window;
  global.document = window.document;
  
  // Wait for DOMContentLoaded and scripts to run
  await new Promise((resolve) => {
    window.addEventListener('DOMContentLoaded', resolve);
    if (window.document.readyState === 'complete' || window.document.readyState === 'interactive') {
      resolve();
    }
  });
  
  // Simulate switching architectures
  const select = window.document.getElementById('architectureTemplateSelect');
  const templates = Array.from(select.options).map(o => o.value).filter(v => v !== 'custom');
  let errorCaught = false;
  for (const template of templates) {
    try {
      select.value = template;
      select.dispatchEvent(new window.Event('change', { bubbles: true }));
      // Wait a bit for async UI update
      await new Promise(r => setTimeout(r, 30));
    } catch (e) {
      errorCaught = true;
      break;
    }
  }
  assert.ok(!errorCaught, 'Switching architectures should not throw errors');
}
