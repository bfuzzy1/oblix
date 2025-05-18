import assert from 'assert';
import fs from 'fs';

export async function run() {
  const html = fs.readFileSync('index.html', 'utf8');
  assert.ok(html.includes('id="generateDataBtn"'), 'Generate Data button missing');
  assert.ok(html.includes('id="trainButton"'), 'Train button missing');
  assert.ok(html.includes('id="networkGraph"'), 'Network graph canvas missing');
  assert.ok(html.includes('id="predictButton"'), 'Predict button missing');
  assert.ok(html.includes('id="playbackSlider"'), 'Playback slider missing');
  assert.ok(html.includes('src="src/main.js"'), 'Main script tag missing');
}
