import fs from 'fs';

export async function run(assert) {
  const html = fs.readFileSync('index.html', 'utf8');
  const js = fs.readFileSync('src/ui/index.js', 'utf8');
  assert.ok(html.includes('<option value="dyna">Dyna-Q</option>'));
  assert.ok(js.includes("import { DynaQAgent } from '../rl/dynaQAgent.js';"));
  assert.ok(js.includes("if (type === 'dyna') return new DynaQAgent(options);"));
  assert.ok(!html.includes('<option value="dqn">'));
  assert.ok(!html.includes('@tensorflow/tfjs'));
  assert.ok(!js.includes('@tensorflow/tfjs'));
}
