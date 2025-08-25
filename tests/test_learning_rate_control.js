import fs from 'fs';
import { JSDOM } from 'jsdom';

export async function run(assert) {
  const html = fs.readFileSync('index.html', 'utf8');
  assert.ok(html.includes('id="learning-rate-slider"'));
  assert.ok(html.includes("learningRateSlider.addEventListener('input'"));
  assert.ok(html.includes('agent.learningRate'));
  assert.ok(html.includes('learningRateValue.textContent = val.toFixed(2);'));

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
