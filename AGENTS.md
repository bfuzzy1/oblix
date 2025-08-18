# AGENTS.md – Development Guidelines for oblix-rl

## Project Overview

**oblix-rl** is a small reinforcement learning playground written in pure JavaScript. It contains a tabular Q-learning agent, a grid world environment and utilities for running training episodes.

### Source Layout
- `src/rl/` – agents, environments and trainers
- `tests/` – unit tests executed with `node tests/run.js`

## Development Standards
- Indentation: 2 spaces
- Semicolons: required
- Naming: camelCase for variables/functions, PascalCase for classes
- Modules: ES6 import/export syntax

## Testing Requirements
- Add tests for each new feature
- Run `node tests/run.js` and ensure it passes before committing

## Pull Request Process
Provide **Context**, **Description** and **Changes** sections in every pull request.
