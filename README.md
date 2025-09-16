# oblix-rl

**oblix-rl** is a browser-based reinforcement learning playground built entirely with modern JavaScript modules. It provides a
collection of tabular agents, a configurable grid world environment and lightweight utilities for experimenting with training
loops directly in the browser or from simple scripts.

## Table of contents
- [Overview](#overview)
- [Quick start](#quick-start)
  - [Install dependencies](#install-dependencies)
  - [Start the browser demo](#start-the-browser-demo)
  - [Run the test suite](#run-the-test-suite)
- [Features](#features)
  - [Agents](#agents)
  - [Exploration policies](#exploration-policies)
  - [Training utilities](#training-utilities)
  - [Environment & UI](#environment--ui)
  - [Persistence](#persistence)
- [Directory layout](#directory-layout)
- [Programmatic usage](#programmatic-usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project focuses on small, understandable reinforcement learning examples rather than production-scale performance. Every
piece of functionality lives in plain ES modules – no build step is required. Open the demo in a browser to watch agents explore
and learn, or import the modules in Node.js to run automated experiments.

## Quick start

### Install dependencies

```bash
npm install
```

Dependencies are intentionally minimal (`jsdom` is used for DOM-enabled tests).

### Start the browser demo

Serve the repository root with any static file server so the ES modules can load correctly:

```bash
# Pick one of the following commands
npx http-server .
# or
python3 -m http.server 8080
```

Then open `http://localhost:8080/index.html` (adjust the port if needed) to interact with the playground. The UI lets you tweak
environment rewards, adjust exploration parameters, watch live charts and save/load progress.

### Run the test suite

```bash
npm test
```

The test runner loads every file in `tests/` and reports pass/fail status.

## Features

### Agents

- **RLAgent:** Baseline tabular Q-learning agent with epsilon decay.
- **ExpectedSarsaAgent & SarsaAgent:** On-policy updates with configurable exploration.
- **DoubleQAgent:** Mitigates maximisation bias by maintaining two Q-tables.
- **DynaQAgent:** Blends real experience with model-based planning steps.
- **QLambdaAgent:** Eligibility traces for faster credit assignment.
- **MonteCarloAgent:** Batch value updates from complete episodes.
- **ActorCriticAgent:** Separate policy and value tables with softmax action selection.
- **OptimisticAgent:** Encourages exploration through optimistic initial values.

### Exploration policies

Choose from several action selection strategies provided by `src/rl/policies.js`:

- Epsilon-greedy and greedy exploitation
- Softmax/Boltzmann exploration
- Thompson sampling with Gaussian noise
- Upper Confidence Bound (UCB)
- Uniform random actions for baseline comparisons

### Training utilities

- **RLTrainer:** Start, pause and reset training loops with adjustable step intervals.
- **trainEpisodes helper:** Run synchronous training loops for scripted experiments.
- **ExperienceReplay:** Uniform or prioritised sampling with automatic importance weights.
- **MetricsTracker:** Episode counters, cumulative reward tracking and epsilon history.
- **Worker-ready design:** `src/rl/trainerWorker.js` mirrors the trainer API for responsive UIs.

### Environment & UI

- **GridWorldEnvironment:** Adjustable size, obstacle editing and reward configuration.
- **UI bindings:** `src/ui/` wires controls for grid editing, live charts and parameter tweaks.
- **Pure ES modules:** Load directly in modern browsers without bundling.

### Persistence

- Save and restore agent state via `agent.toJSON()` / `RLAgent.fromJSON()`.
- `src/rl/storage.js` stores agents and environments in `localStorage` for quick restarts.

## Directory layout

```
├── index.html          # Browser playground entry point
├── src/
│   ├── rl/             # Agents, environments, trainers and utilities
│   └── ui/             # UI bindings, live chart and agent factory
├── tests/              # Node-based regression tests (npm test)
├── package.json        # npm scripts and dependencies
└── README.md           # Project documentation
```

## Programmatic usage

All modules are ES modules, so they can be imported directly from scripts. The snippet below trains a Dyna-Q agent for several
episodes and logs the serialised agent:

```js
import { GridWorldEnvironment } from './src/rl/environment.js';
import { DynaQAgent } from './src/rl/dynaQAgent.js';
import { RLTrainer } from './src/rl/training.js';

const env = new GridWorldEnvironment(5, [], {
  stepPenalty: -0.01,
  obstaclePenalty: -0.1,
  goalReward: 1
});

const agent = new DynaQAgent({
  epsilon: 0.2,
  epsilonDecay: 0.995,
  minEpsilon: 0.05,
  planningSteps: 10
});

await RLTrainer.trainEpisodes(agent, env, 50, 75);
console.log(agent.toJSON());
```

To mirror the browser experience, create an `RLTrainer` instance and subscribe to its updates:

```js
import { RLTrainer } from './src/rl/training.js';
import { ExperienceReplay } from './src/rl/experienceReplay.js';

const trainer = new RLTrainer(agent, env, {
  intervalMs: 50,
  replaySamples: 32,
  replayStrategy: 'priority',
  replayBuffer: new ExperienceReplay(2000, 0.6, 0.4, 0.001),
  onStep: (state, reward, done, metrics) => {
    console.log(`reward: ${reward.toFixed(2)}`, metrics);
  }
});

trainer.start();
setTimeout(() => trainer.pause(), 5000);
```

## Contributing

Pull requests are welcome! Please follow the project coding style, keep documentation up to date and run `npm test` before
submitting changes.

## License

[MIT](LICENSE)
