import { GridWorldEnvironment } from './environment.js';

function clampCoordinate(value, max) {
  const intValue = Math.trunc(value);
  if (!Number.isFinite(intValue)) return 0;
  if (intValue < 0) return 0;
  if (intValue > max) return max;
  return intValue;
}

function clampOffset(value, size) {
  const intValue = Math.trunc(value);
  if (!Number.isFinite(intValue)) return 0;
  const limit = size - 1;
  if (intValue < -limit) return -limit;
  if (intValue > limit) return limit;
  return intValue;
}

function normalizeWindColumns(columns = [], size = 5) {
  const normalized = [];
  for (const entry of columns) {
    if (!entry) continue;
    const x = clampCoordinate(entry.x, size - 1);
    if (!Number.isInteger(x)) continue;
    const offsets = Array.isArray(entry.offsets) ? entry.offsets : [];
    const sanitizedOffsets = offsets
      .map(offset => {
        const weight = Number(offset?.weight ?? offset?.probability ?? 1);
        if (!Number.isFinite(weight) || weight <= 0) {
          return null;
        }
        const dx = clampOffset(offset?.dx ?? 0, size);
        const dy = clampOffset(offset?.dy ?? 0, size);
        return { dx, dy, weight };
      })
      .filter(Boolean);
    if (sanitizedOffsets.length === 0) {
      sanitizedOffsets.push({ dx: 0, dy: -1, weight: 1 });
    }
    const totalWeight = sanitizedOffsets.reduce((acc, cur) => acc + cur.weight, 0);
    let cumulative = 0;
    const distribution = sanitizedOffsets.map(offset => {
      cumulative += offset.weight / totalWeight;
      return { ...offset, threshold: cumulative };
    });
    normalized.push({ x, distribution, offsets: sanitizedOffsets.map(o => ({ dx: o.dx, dy: o.dy, weight: o.weight })) });
  }
  return normalized;
}

export class WindyGridEnvironment extends GridWorldEnvironment {
  constructor(size = 5, obstacles = [], rewardConfig = {}, windColumns = []) {
    super(size, obstacles, rewardConfig);
    this.scenarioId = 'windy';
    this.windMap = new Map();
    this.rawWindConfig = [];
    this.setWind(windColumns);
  }

  setWind(columns = []) {
    const normalized = normalizeWindColumns(columns, this.size);
    this.windMap = new Map(normalized.map(entry => [entry.x, entry.distribution]));
    this.rawWindConfig = normalized.map(entry => ({ x: entry.x, offsets: entry.offsets }));
  }

  applyWind(x, y) {
    const distribution = this.windMap.get(x);
    if (!distribution || distribution.length === 0) {
      return { x, y };
    }
    const rand = Math.random();
    for (const entry of distribution) {
      if (rand <= entry.threshold) {
        const nextX = clampCoordinate(x + entry.dx, this.size - 1);
        const nextY = clampCoordinate(y + entry.dy, this.size - 1);
        return { x: nextX, y: nextY };
      }
    }
    return { x, y };
  }

  step(action) {
    let newX = this.agentPos.x;
    let newY = this.agentPos.y;
    switch (action) {
      case 0:
        if (newY > 0) newY -= 1;
        break;
      case 1:
        if (newY < this.size - 1) newY += 1;
        break;
      case 2:
        if (newX > 0) newX -= 1;
        break;
      case 3:
        if (newX < this.size - 1) newX += 1;
        break;
    }
    const drifted = this.applyWind(newX, newY);
    newX = drifted.x;
    newY = drifted.y;
    if (this.isObstacle(newX, newY)) {
      return { state: this.getState(), reward: this.obstaclePenalty, done: false };
    }
    this.agentPos = { x: newX, y: newY };
    const goal = this.getGoalPosition();
    const done = this.agentPos.x === goal.x && this.agentPos.y === goal.y;
    const reward = this.calculateReward(this.agentPos.x, this.agentPos.y, done);
    return { state: this.getState(), reward, done };
  }

  describeCell(x, y) {
    const info = { ...super.describeCell(x, y) };
    if (!Array.isArray(info.classes)) {
      info.classes = [];
    }
    if (this.windMap.has(x)) {
      info.classes = [...info.classes, 'wind-zone'];
    }
    return info;
  }

  getScenarioConfig() {
    return { windColumns: this.rawWindConfig.map(entry => ({ x: entry.x, offsets: entry.offsets.map(o => ({ ...o })) })) };
  }

  getScenarioMetadata() {
    return {
      id: this.scenarioId,
      windColumns: this.rawWindConfig.map(entry => ({ x: entry.x, offsets: entry.offsets.map(o => ({ ...o })) }))
    };
  }
}

export default WindyGridEnvironment;
