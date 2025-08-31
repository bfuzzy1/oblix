import { saveEnvironment } from '../rl/storage.js';

let env;
let gridEl;
let size;

export function initRenderer(environment, element, gridSize) {
  env = environment;
  gridEl = element;
  size = gridSize;
  gridEl.style.setProperty('--size', size);
}

export function render(state) {
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
        render(env.getState());
      });
      gridEl.appendChild(cell);
    }
  }
}
