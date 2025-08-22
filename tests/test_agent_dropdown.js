import fs from 'fs';

export async function run(assert) {
  const html = fs.readFileSync('index.html', 'utf8');
  assert.ok(html.includes('<option value="dyna">Dyna-Q</option>'));
  assert.ok(html.includes("import { DynaQAgent } from './src/rl/dynaQAgent.js';"));
  assert.ok(html.includes("if (type === 'dyna') return new DynaQAgent(options);"));
  assert.ok(html.includes('<option value="dqn">DQN</option>'));
  assert.ok(html.includes("import { DQNAgent } from './src/rl/dqnAgent.js';"));
  assert.ok(html.includes("if (type === 'dqn') return new DQNAgent(options);"));
  assert.ok(html.includes('<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>'));
}
