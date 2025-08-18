# oblix-rl

**oblix-rl** is a browser-based reinforcement learning playground written entirely in JavaScript. It lets you build simple agents and environments and watch them learn directly in the browser without external dependencies.

## Features

- **Grid World Environment:** Discrete environment for quick experimentation.
- **Tabular Q-Learning Agent:** Value estimates stored in a simple lookup table.
- **Epsilon-Greedy Exploration:** Exploration rate automatically decays during training.
- **Episode Trainer:** Run training loops with callbacks for visualization.
- **Pure ES6 Modules:** No build step, works with static file hosting.
- **Interactive Demo:** Open `index.html` to watch the agent learn in real time.

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
import { RLAgent } from './src/rl/agent.js';
import { RLTrainer } from './src/rl/training.js';

const env = new GridWorldEnvironment(5);
const agent = new RLAgent({ epsilon: 0.2 });
await RLTrainer.trainEpisodes(agent, env, 50, 50);
```

This repository focuses solely on reinforcement learning; previous model training utilities have been removed in favor of streamlined RL components.

## Frontend Demo

Open `index.html` in a browser to interact with the grid world. Use the Start, Pause and Reset buttons to control training and watch the agent improve.
