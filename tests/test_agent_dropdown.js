import fs from 'fs';

export async function run(assert) {
  const html = fs.readFileSync('index.html', 'utf8');
  assert.ok(html.includes('<option value="dyna">Dyna-Q</option>'));
  assert.ok(html.includes("import { DynaQAgent } from './src/rl/dynaQAgent.js';"));
  assert.ok(html.includes("if (type === 'dyna') return new DynaQAgent(options);"));
}
