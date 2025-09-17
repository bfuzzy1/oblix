import { GridWorldEnvironment } from './environment.js';

function clamp(value, min, max) {
  const intValue = Math.trunc(value);
  if (!Number.isFinite(intValue)) return min;
  if (intValue < min) return min;
  if (intValue > max) return max;
  return intValue;
}

function createDefaultPattern(size) {
  const max = size - 1;
  const mid = clamp(Math.floor(max / 2), 0, max);
  const pattern = [
    { x: max, y: max },
    { x: max, y: 0 },
    { x: 0, y: max },
    { x: mid, y: mid }
  ];
  const unique = [];
  const seen = new Set();
  for (const pos of pattern) {
    const key = `${pos.x},${pos.y}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(pos);
  }
  return unique;
}

function sanitizePattern(pattern = [], size) {
  if (!Array.isArray(pattern) || pattern.length === 0) {
    return createDefaultPattern(size);
  }
  const max = size - 1;
  const unique = [];
  const seen = new Set();
  for (const pos of pattern) {
    if (!pos) continue;
    const x = clamp(pos.x, 0, max);
    const y = clamp(pos.y, 0, max);
    const key = `${x},${y}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push({ x, y });
  }
  return unique.length > 0 ? unique : createDefaultPattern(size);
}

export class MovingGoalEnvironment extends GridWorldEnvironment {
  constructor(size = 5, obstacles = [], rewardConfig = {}, options = {}) {
    super(size, obstacles, rewardConfig);
    this.scenarioId = 'moving-goal';
    this.goalPattern = [];
    this.goalIndex = 0;
    this.goalPos = super.getGoalPosition();
    this.moveFrequency = 6;
    this.stepCounter = 0;
    this.applyScenarioOptions(options);
  }

  applyScenarioOptions(options = {}) {
    const pattern = sanitizePattern(options.goalPattern, this.size);
    this.goalPattern = pattern;
    const frequency = Number(options.moveFrequency);
    if (Number.isFinite(frequency) && frequency > 0) {
      this.moveFrequency = Math.trunc(frequency);
    }
    const maxIndex = this.goalPattern.length - 1;
    const providedIndex = Number(options.goalIndex);
    if (Number.isFinite(providedIndex) && maxIndex >= 0) {
      const normalized = ((Math.trunc(providedIndex) % this.goalPattern.length) + this.goalPattern.length) % this.goalPattern.length;
      this.goalIndex = normalized;
    } else {
      this.goalIndex = 0;
    }
    this.goalPos = this.goalPattern[this.goalIndex] ?? super.getGoalPosition();
  }

  getGoalPosition() {
    return this.goalPos ?? super.getGoalPosition();
  }

  advanceGoal() {
    if (!this.goalPattern.length) {
      return;
    }
    this.goalIndex = (this.goalIndex + 1) % this.goalPattern.length;
    this.goalPos = this.goalPattern[this.goalIndex];
  }

  reset() {
    this.stepCounter = 0;
    if (!Array.isArray(this.goalPattern) || this.goalPattern.length === 0) {
      this.goalPattern = sanitizePattern([], this.size);
      this.goalIndex = 0;
    }
    this.goalPos = this.goalPattern[this.goalIndex] ?? super.getGoalPosition();
    return super.reset();
  }

  step(action) {
    const result = super.step(action);
    this.stepCounter += 1;
    if (result.done) {
      this.advanceGoal();
      this.stepCounter = 0;
    } else if (this.moveFrequency > 0 && this.stepCounter % this.moveFrequency === 0) {
      this.advanceGoal();
    }
    return result;
  }

  describeCell(x, y) {
    const info = { ...super.describeCell(x, y) };
    if (!Array.isArray(info.classes)) {
      info.classes = [];
    }
    const goal = this.getGoalPosition();
    if (x === goal.x && y === goal.y) {
      info.classes = [...info.classes, 'moving-goal'];
    } else if (this.goalPattern.some((pos, index) => index !== this.goalIndex && pos.x === x && pos.y === y)) {
      info.classes = [...info.classes, 'future-goal'];
    }
    return info;
  }

  getScenarioConfig() {
    return {
      goalPattern: this.goalPattern.map(pos => ({ ...pos })),
      moveFrequency: this.moveFrequency,
      goalIndex: this.goalIndex
    };
  }

  getScenarioMetadata() {
    return {
      id: this.scenarioId,
      goal: this.getGoalPosition(),
      futureGoals: this.goalPattern
        .map((pos, index) => ({ ...pos, index }))
        .filter(entry => entry.index !== this.goalIndex)
    };
  }
}

export default MovingGoalEnvironment;
