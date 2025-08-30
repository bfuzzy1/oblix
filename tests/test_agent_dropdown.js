import fs from 'fs';

export async function run(assert) {
  const html = fs.readFileSync('index.html', 'utf8');
  const js = fs.readFileSync('src/ui/index.js', 'utf8');
  assert.ok(html.includes('<option value="dyna">Dyna-Q</option>'));
  assert.ok(js.includes("import { DynaQAgent } from '../rl/dynaQAgent.js';"));
  assert.ok(js.includes("if (type === 'dyna') return new DynaQAgent(options);"));
  assert.ok(html.includes('<option value="qlambda">Q(&lambda;)</option>'));
  assert.ok(js.includes("import { QLambdaAgent } from '../rl/qLambdaAgent.js';"));
  assert.ok(js.includes("if (type === 'qlambda') return new QLambdaAgent(options);"));
  assert.ok(!html.includes('<option value="dqn">'));
  assert.ok(!html.includes('@tensorflow/tfjs'));
  assert.ok(!js.includes('@tensorflow/tfjs'));
}
