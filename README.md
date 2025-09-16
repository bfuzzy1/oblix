# oblix-rl

**oblix-rl** is a browser-based reinforcement learning playground written entirely in JavaScript. It lets you build simple agents and environments and watch them learn directly in the browser without external dependencies.

## Features

- **Grid World Environment:** Discrete environment for quick experimentation.
- **Tabular Q-Learning Agent:** Value estimates stored in a simple lookup table.
- **SARSA and Expected SARSA Agents:** Alternatives to standard Q-learning.
- **Dyna-Q Agent:** Combines Q-learning with planning from a learned model.
- **Optimistic Q-Learning Agent:** Uses high initial values to encourage exploration.
- **Epsilon-Greedy Exploration:** Exploration rate automatically decays during training.
- **Random Policy:** Baseline policy that selects actions uniformly at random.
- **Episode Trainer:** Run training loops with callbacks for visualization.
- **Pure ES6 Modules:** No build step, works with static file hosting.
- **Interactive Demo:** Open `index.html` to watch the agent learn in real time.
- **Adjustable Grid Size:** Change the environment dimensions directly from the UI.
- **Environment Persistence:** Save and load grid size and obstacles.

## Project Structure

- `src/` – core source files.
  - `rl/` – reinforcement learning agents, environments and trainers.
- `tests/` – unit tests executed with `node tests/run.js`.

## Running Tests

```
npm test
```

## Getting Started

The snippet below trains an agent in a 5x5 grid world:

```js
import { GridWorldEnvironment } from './src/rl/environment.js';
import { ExpectedSarsaAgent } from './src/rl/expectedSarsaAgent.js';
import { RLTrainer } from './src/rl/training.js';

const env = new GridWorldEnvironment(5, [], {
  stepPenalty: -0.01,
  obstaclePenalty: -0.1,
  goalReward: 1
});
const agent = new ExpectedSarsaAgent({ epsilon: 0.2 });
await RLTrainer.trainEpisodes(agent, env, 50, 50);
```

This repository focuses solely on reinforcement learning; previous model training utilities have been removed in favor of streamlined RL components.

## Dyna-Q Planning

`DynaQAgent` augments standard Q-learning with a simple model of the environment. Each real interaction stores the observed transition `(s, a) -> (s', r)`. After the real update, the agent samples a number of these stored transitions and applies additional Q-learning updates, effectively "planning" using its learned model. The number of planning iterations is configured with the `planningSteps` option.

```js
import { DynaQAgent } from './src/rl/dynaQAgent.js';
const agent = new DynaQAgent({ planningSteps: 10 });
```

## Saving and Loading Agents

You can persist a trained agent by converting it to a plain object and later recreating it:

```js
const saved = agent.toJSON();
const restored = RLAgent.fromJSON(saved);
```

## Frontend Demo

Open `index.html` in a browser to interact with the grid world. Use the Start, Pause and Reset buttons to control training and watch the agent improve.
Use the Grid Size input to resize the environment, tweak the step and obstacle penalties, and adjust the goal reward to explore different incentive structures. Toggle cells to set obstacles. The Save and Load buttons persist both the agent and the environment layout in your browser.
