import { GridWorldEnvironment } from './environment.js';

function clamp(value, min, max) {
  const intValue = Math.trunc(value);
  if (!Number.isFinite(intValue)) return min;
  if (intValue < min) return min;
  if (intValue > max) return max;
  return intValue;
}

function sanitizeRewards(cells = [], size) {
  if (!Array.isArray(cells) || cells.length === 0) {
    return [];
  }
  const max = size - 1;
  const sanitized = new Map();
  for (const cell of cells) {
    if (!cell) continue;
    const rewardValue = Number(cell.reward ?? cell.value);
    if (!Number.isFinite(rewardValue)) continue;
    const x = clamp(cell.x, 0, max);
    const y = clamp(cell.y, 0, max);
    const key = `${x},${y}`;
    sanitized.set(key, { x, y, reward: rewardValue });
  }
  return Array.from(sanitized.values());
}

export class RewardGridEnvironment extends GridWorldEnvironment {
  constructor(size = 5, obstacles = [], rewardConfig = {}, rewardCells = []) {
    super(size, obstacles, rewardConfig);
    this.scenarioId = 'reward-grid';
    this.rewardMap = new Map();
    this.rewardCells = [];
    this.setRewardCells(rewardCells);
  }

  setRewardCells(cells = []) {
    const sanitized = sanitizeRewards(cells, this.size);
    this.rewardCells = sanitized.map(entry => ({ ...entry }));
    this.rewardMap = new Map(sanitized.map(entry => [`${entry.x},${entry.y}`, entry.reward]));
  }

  calculateReward(newX, newY, done) {
    const baseReward = super.calculateReward(newX, newY, done);
    const key = `${newX},${newY}`;
    const bonus = this.rewardMap.get(key) ?? 0;
    return baseReward + bonus;
  }

  describeCell(x, y) {
    const info = { ...super.describeCell(x, y) };
    if (!Array.isArray(info.classes)) {
      info.classes = [];
    }
    const key = `${x},${y}`;
    if (this.rewardMap.has(key)) {
      const reward = this.rewardMap.get(key);
      info.classes = [
        ...info.classes,
        reward >= 0 ? 'reward-bonus' : 'reward-penalty'
      ];
      info.reward = reward;
    }
    return info;
  }

  getScenarioConfig() {
    return {
      rewardCells: this.rewardCells.map(entry => ({ ...entry }))
    };
  }

  getScenarioMetadata() {
    return {
      id: this.scenarioId,
      rewardCells: this.rewardCells.map(entry => ({ ...entry }))
    };
  }
}

export default RewardGridEnvironment;
