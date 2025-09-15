import { saveEnvironment } from '../rl/storage.js';

const ACTIONS = [
  { key: 'up', symbol: '↑' },
  { key: 'down', symbol: '↓' },
  { key: 'left', symbol: '←' },
  { key: 'right', symbol: '→' }
];

const ZERO_Q = new Float32Array(4);

let env;
let gridEl;
let size;
let currentAgent = null;

function keyWithinGrid(key, gridSize) {
  if (!key) return false;
  const [xStr, yStr] = key.split(',');
  const x = parseInt(xStr, 10);
  const y = parseInt(yStr, 10);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  if (gridSize === undefined) return true;
  return x >= 0 && y >= 0 && x < gridSize && y < gridSize;
}

function getAgentTables(agent) {
  if (!agent) return [];
  const tables = [];
  if (agent.qTable && typeof agent.qTable.get === 'function' && agent.qTable.size > 0) {
    tables.push(agent.qTable);
    return tables;
  }
  if (agent.qTableA && typeof agent.qTableA.get === 'function') tables.push(agent.qTableA);
  if (agent.qTableB && typeof agent.qTableB.get === 'function') tables.push(agent.qTableB);
  if (tables.length === 0 && agent.qTable && typeof agent.qTable.get === 'function') {
    tables.push(agent.qTable);
  }
  return tables;
}

function summarizeQValues(agent, gridSize) {
  const tables = getAgentTables(agent);
  if (tables.length === 0) {
    return { range: 0, min: 0, max: 0 };
  }
  let min = Infinity;
  let max = -Infinity;
  for (const table of tables) {
    for (const [key, values] of table.entries()) {
      if (!keyWithinGrid(key, gridSize)) continue;
      for (const value of values) {
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
  }
  if (min === Infinity) {
    return { range: 0, min: 0, max: 0 };
  }
  const range = Math.max(Math.abs(min), Math.abs(max));
  return { range, min, max };
}

function getCellQValues(agent, x, y) {
  const tables = getAgentTables(agent);
  const key = `${x},${y}`;
  if (tables.length === 0) return ZERO_Q;
  if (tables.length === 1) {
    return tables[0].get(key) ?? ZERO_Q;
  }
  let combined = null;
  let count = 0;
  for (const table of tables) {
    const values = table.get(key);
    if (!values) continue;
    if (!combined) {
      combined = new Float32Array(values);
    } else {
      for (let i = 0; i < values.length; i++) {
        combined[i] += values[i];
      }
    }
    count++;
  }
  if (!combined) {
    return ZERO_Q;
  }
  if (count > 1) {
    for (let i = 0; i < combined.length; i++) {
      combined[i] /= count;
    }
  }
  return combined;
}

function colorForValue(value, range) {
  if (!Number.isFinite(value) || !Number.isFinite(range) || range <= 0) {
    return 'rgba(255, 255, 255, 0.08)';
  }
  const normalized = Math.max(-1, Math.min(1, value / range));
  const intensity = Math.abs(normalized);
  const hue = normalized >= 0 ? 140 : 0;
  const saturation = Math.round(30 + intensity * 60);
  const lightness = Math.round(30 + (1 - intensity) * 20);
  const alpha = Math.min(0.3 + intensity * 0.5, 0.8);
  return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha.toFixed(2)})`;
}

function createQOverlay(qVals, range) {
  const layer = document.createElement('div');
  layer.className = 'q-layer';
  let bestIndex = 0;
  for (let i = 1; i < qVals.length; i++) {
    if (qVals[i] > qVals[bestIndex]) bestIndex = i;
  }
  ACTIONS.forEach((action, index) => {
    const value = qVals[index] ?? 0;
    const el = document.createElement('div');
    el.className = `q-value q-${action.key}`;
    el.dataset.action = action.key;
    el.dataset.value = Number.isFinite(value) ? value.toString() : '0';
    el.textContent = `${action.symbol} ${Number.isFinite(value) ? value.toFixed(2) : '0.00'}`;
    el.style.background = colorForValue(value, range);
    if (index === bestIndex) {
      el.classList.add('best');
    }
    layer.appendChild(el);
  });
  return layer;
}

export function initRenderer(environment, element, gridSize) {
  env = environment;
  gridEl = element;
  size = gridSize;
  currentAgent = null;
  gridEl.style.setProperty('--size', size);
}

export function render(state, agent) {
  if (!gridEl || !env) return;
  if (agent !== undefined) {
    currentAgent = agent;
  }
  const activeAgent = currentAgent;
  const { range } = activeAgent ? summarizeQValues(activeAgent, size) : { range: 0 };
  gridEl.innerHTML = '';
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      if (env.isObstacle(x, y)) cell.classList.add('obstacle');
      if (x === state[0] && y === state[1]) cell.classList.add('agent');
      if (x === size - 1 && y === size - 1) cell.classList.add('goal');
      cell.addEventListener('click', () => {
        if (x === env.agentPos.x && y === env.agentPos.y) return;
        if (x === size - 1 && y === size - 1) return;
        env.toggleObstacle(x, y);
        saveEnvironment(env);
        render(env.getState(), currentAgent);
      });
      if (activeAgent) {
        const qVals = getCellQValues(activeAgent, x, y);
        cell.appendChild(createQOverlay(qVals, range));
      }
      gridEl.appendChild(cell);
    }
  }
}
