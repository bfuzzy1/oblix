# AGENTS.md – Development Guidelines for oblix-rl

## Project Overview

**oblix-rl** is a small reinforcement learning playground written in pure JavaScript. It contains a tabular Q-learning agent, a grid world environment and utilities for running training episodes.

## Repository Structure
- `src/rl/` – agents, environments and trainers
- `tests/` – unit tests executed with `npm test`

## Development Guidelines

### Code Style
- Indentation: 2 spaces
- Semicolons: required
- Naming: camelCase for variables/functions, PascalCase for classes
- Modules: ES6 import/export syntax

### Documentation
- Update README and inline comments when behavior changes
- Keep docs concise and in English

### Tooling
- Use `rg` (ripgrep) for code searches; avoid `grep -R` for performance

## Testing
- Add tests for each new feature or bug fix
- Run `npm test` and ensure it passes before committing (skip for documentation-only changes)

## Commit Guidelines
- Use clear, imperative commit messages; Conventional Commit style preferred
- Avoid external URLs in commits
- Separate unrelated changes into distinct commits

## Pull Request Process
- Provide **Context**, **Description** and **Changes** sections in every pull request
- End each pull request message with `Passing to @codex for code review`
