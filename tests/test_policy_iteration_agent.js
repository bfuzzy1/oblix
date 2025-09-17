import assert from 'assert';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { PolicyIterationAgent } from '../src/rl/policyIterationAgent.js';

function simulateAgent(agent, environment) {
  let state = environment.reset();
  let done = false;
  let steps = 0;
  const maxSteps = environment.size * environment.size;
  while (!done && steps < maxSteps) {
    const action = agent.act(state);
    const { state: nextState, done: reachedGoal } = environment.step(action);
    state = nextState;
    done = reachedGoal;
    steps += 1;
  }
  return { done, steps, state };
}

export async function run() {
  const env = new GridWorldEnvironment(5);
  env.reset();
  const agent = new PolicyIterationAgent();
  agent.reset(env);
  const result = simulateAgent(agent, env);
  assert.ok(result.done, 'Policy iteration agent should reach the goal');
  assert.strictEqual(result.steps, (env.size - 1) * 2, 'Policy iteration agent should reach the goal using the optimal number of steps');
  const goal = env.getGoalPosition();
  assert.deepStrictEqual(Array.from(result.state), [goal.x, goal.y], 'Policy iteration agent should finish at the goal state');
}
