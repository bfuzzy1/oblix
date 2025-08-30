import { ActorCriticAgent } from '../src/rl/actorCriticAgent.js';
import { RLAgent } from '../src/rl/agent.js';
import { GridWorldEnvironment } from '../src/rl/environment.js';

function seedRandom(seed) {
  Math.random = (() => {
    let x = seed;
    return () => {
      x = (x * 16807) % 2147483647;
      return (x - 1) / 2147483646;
    };
  })();
}

async function runEpisodes(agent, learn, episodes = 30, maxSteps = 50) {
  const env = new GridWorldEnvironment(5);
  let total = 0;
  for (let ep = 0; ep < episodes; ep++) {
    let state = env.reset();
    for (let st = 0; st < maxSteps; st++) {
      const action = await agent.act(state);
      const { state: nextState, reward, done } = env.step(action);
      total += reward;
      if (learn) await agent.learn(state, action, reward, nextState, done);
      state = nextState;
      if (done) break;
    }
  }
  return total;
}

export async function run(assert) {
  const state = new Float32Array([0, 0]);
  const agent = new ActorCriticAgent();
  const key = Array.from(state).join(',');
  agent.policyTable.set(key, new Float32Array([1, 2, 3, 4]));
  const probs = agent.policyProbs(state);
  const sum = probs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6);

  const origRandom = Math.random;
  seedRandom(1);
  const randomAgent = new RLAgent({ epsilon: 1, learningRate: 0 });
  const baselineReward = await runEpisodes(randomAgent, false);
  seedRandom(1);
  const acAgent = new ActorCriticAgent();
  const acReward = await runEpisodes(acAgent, true);
  Math.random = origRandom;
  assert.ok(acReward > baselineReward);
}
