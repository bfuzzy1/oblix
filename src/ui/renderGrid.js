import { saveEnvironment } from '../rl/storage.js';

let env;
let gridEl;
let size;
let onEnvironmentChange;

function getGoalPosition(environment, gridSize) {
  if (environment && typeof environment.getGoalPosition === 'function') {
    const goal = environment.getGoalPosition();
    if (goal && Number.isFinite(goal.x) && Number.isFinite(goal.y)) {
      return goal;
    }
  }
  const fallback = gridSize - 1;
  return { x: fallback, y: fallback };
}

function applyCellMetadata(cell, metadata) {
  if (!metadata || typeof metadata !== 'object') {
    cell.removeAttribute('data-reward');
    return;
  }
  if (Array.isArray(metadata.classes)) {
    for (const cls of metadata.classes) {
      if (typeof cls === 'string' && cls) {
        cell.classList.add(cls);
      }
    }
  }
  if (metadata.reward !== undefined) {
    const rewardValue = Number(metadata.reward);
    if (Number.isFinite(rewardValue)) {
      cell.dataset.reward = rewardValue.toFixed(2);
    } else {
      cell.removeAttribute('data-reward');
    }
  } else {
    cell.removeAttribute('data-reward');
  }
}

export function initRenderer(environment, element, gridSize, environmentChangeCallback = null) {
  env = environment;
  gridEl = element;
  size = gridSize;
  onEnvironmentChange = environmentChangeCallback;
  gridEl.style.setProperty('--size', size);
}

export function render(state) {
  if (!gridEl || !env) return;
  const position = state ?? (typeof env.getState === 'function' ? env.getState() : null);
  if (!position) return;
  const goal = getGoalPosition(env, size);
  gridEl.innerHTML = '';
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      if (env.isObstacle(x, y)) cell.classList.add('obstacle');
      if (x === position[0] && y === position[1]) cell.classList.add('agent');
      if (x === goal.x && y === goal.y) cell.classList.add('goal');
      const metadata = typeof env.describeCell === 'function' ? env.describeCell(x, y) : null;
      applyCellMetadata(cell, metadata);
      cell.addEventListener('click', () => {
        if (x === env.agentPos.x && y === env.agentPos.y) return;
        if (x === goal.x && y === goal.y) return;
        env.toggleObstacle(x, y);
        saveEnvironment(env);
        if (typeof onEnvironmentChange === 'function') {
          onEnvironmentChange(env);
        }
        render(env.getState());
      });
      gridEl.appendChild(cell);
    }
  }
}
