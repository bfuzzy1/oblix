import fs from 'fs';
import { JSDOM } from 'jsdom';

export async function run(assert) {
  const js = fs.readFileSync('src/ui/bindControls.js', 'utf8');
  assert.ok(js.includes("document.getElementById('learning-rate-slider')"));
  assert.ok(js.includes("learningRateSlider.addEventListener('input'"));
  assert.ok(js.includes('agent.learningRate = val;'));
  assert.ok(js.includes('learningRateValue.textContent = val.toFixed(2);'));
  assert.ok(js.includes('if (agent.learningRate === undefined) {'));
  assert.ok(js.includes("learningRateSlider.disabled = true;"));
  assert.ok(js.includes("learningRateValue.textContent = 'N/A';"));

  const dom = new JSDOM(`<input id="learning-rate-slider"><span id="learning-rate-value"></span>`);
  const { document, Event } = dom.window;
  const agent = { learningRate: 0.1 };
  const learningRateSlider = document.getElementById('learning-rate-slider');
  const learningRateValue = document.getElementById('learning-rate-value');

  learningRateSlider.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    agent.learningRate = val;
    learningRateValue.textContent = val.toFixed(2);
  });

  learningRateSlider.value = '0.5';
  learningRateSlider.dispatchEvent(new Event('input'));
  assert.strictEqual(agent.learningRate, 0.5);
  assert.strictEqual(learningRateValue.textContent, '0.50');
}
