import { saveEnvironment } from '../rl/storage.js';

let env;
let gridEl;
let size;
let onEnvironmentChange;

const DEFAULT_AGENT_COLOR = 'agent-marker-primary';

function toGridPosition(state) {
  if (!state) return null;
  if (ArrayBuffer.isView(state) || Array.isArray(state)) {
    const [sx, sy] = state;
    return {
      x: Math.trunc(Number.isFinite(sx) ? sx : 0),
      y: Math.trunc(Number.isFinite(sy) ? sy : 0)
    };
  }
  if (typeof state === 'object') {
    if (Number.isFinite(state.x) && Number.isFinite(state.y)) {
      return { x: Math.trunc(state.x), y: Math.trunc(state.y) };
    }
    if (Array.isArray(state.state) || ArrayBuffer.isView(state.state)) {
      return toGridPosition(state.state);
    }
  }
  return null;
}

function normalizeAgentStates(input) {
  if (!input) return [];
  const list = Array.isArray(input) ? input : [input];
  const result = [];
  for (let i = 0; i < list.length; i += 1) {
    const entry = list[i];
    if (!entry) continue;
    if (entry.position && typeof entry.position === 'object') {
      const pos = toGridPosition(entry.position);
      if (!pos) continue;
      result.push({
        x: pos.x,
        y: pos.y,
        colorClass: entry.colorClass || DEFAULT_AGENT_COLOR,
        label: entry.label || null
      });
      continue;
    }
    const pos = toGridPosition(entry);
    if (!pos) continue;
    const colorClass = typeof entry === 'object' && entry.colorClass
      ? entry.colorClass
      : DEFAULT_AGENT_COLOR;
    const label = typeof entry === 'object' && entry.label ? entry.label : null;
    result.push({ x: pos.x, y: pos.y, colorClass, label });
  }
  return result;
}

function renderAgentMarkers(cell, agents) {
  if (!agents.length) {
    return;
  }
  if (agents.length === 1) {
    cell.classList.add('agent');
  } else {
    cell.classList.remove('agent');
  }
  const markerContainer = document.createElement('div');
  markerContainer.className = 'agent-markers';
  const manyAgents = agents.length > 3;
  for (const agentData of agents) {
    const marker = document.createElement('span');
    const colorClass = agentData.colorClass || DEFAULT_AGENT_COLOR;
    marker.className = `agent-marker ${manyAgents ? 'agent-marker-many' : colorClass}`;
    if (agentData.label) {
      marker.title = agentData.label;
    }
    markerContainer.appendChild(marker);
  }
  cell.appendChild(markerContainer);
}

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
  const agentStates = normalizeAgentStates(state ?? (typeof env.getState === 'function' ? env.getState() : null));
  if (!agentStates.length) return;
  const goal = getGoalPosition(env, size);
  gridEl.innerHTML = '';
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      if (env.isObstacle(x, y)) cell.classList.add('obstacle');
      const agentsHere = agentStates.filter(agentData => agentData.x === x && agentData.y === y);
      const hasAgentAtCell = agentsHere.length > 0;
      if (hasAgentAtCell) {
        renderAgentMarkers(cell, agentsHere);
      }
      if (x === goal.x && y === goal.y) cell.classList.add('goal');
      const metadata = typeof env.describeCell === 'function' ? env.describeCell(x, y) : null;
      applyCellMetadata(cell, metadata);
      cell.addEventListener('click', () => {
        if (hasAgentAtCell) return;
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
