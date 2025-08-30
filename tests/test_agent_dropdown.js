import fs from 'fs';

export async function run(assert) {
  const html = fs.readFileSync('index.html', 'utf8');
  const js = fs.readFileSync('src/ui/index.js', 'utf8');
  assert.ok(html.includes('<option value="mc">Monte Carlo</option>'));
  assert.ok(js.includes("import { MonteCarloAgent } from '../rl/monteCarloAgent.js';"));
  assert.ok(js.includes("if (type === 'mc') return new MonteCarloAgent(options);"));
  assert.ok(html.includes('<option value="dyna">Dyna-Q</option>'));
  assert.ok(js.includes("import { DynaQAgent } from '../rl/dynaQAgent.js';"));
  assert.ok(js.includes("if (type === 'dyna') return new DynaQAgent(options);"));
  assert.ok(html.includes('<option value="qlambda">Q(&lambda;)</option>'));
  assert.ok(js.includes("import { QLambdaAgent } from '../rl/qLambdaAgent.js';"));
  assert.ok(js.includes("if (type === 'qlambda') return new QLambdaAgent(options);"));
  assert.ok(html.includes('<option value="ac">Actor-Critic</option>'));
  assert.ok(js.includes("import { ActorCriticAgent } from '../rl/actorCriticAgent.js';"));
  assert.ok(js.includes("if (type === 'ac') return new ActorCriticAgent(options);"));
  assert.ok(html.includes('<option value="double">Double Q-learning</option>'));
  assert.ok(js.includes("import { DoubleQAgent } from '../rl/doubleQAgent.js';"));
  assert.ok(js.includes("if (type === 'double') return new DoubleQAgent(options);"));
  assert.ok(!html.includes('<option value="dqn">'));
  assert.ok(!html.includes('@tensorflow/tfjs'));
  assert.ok(!js.includes('@tensorflow/tfjs'));
}
