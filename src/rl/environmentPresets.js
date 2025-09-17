import { GridWorldEnvironment, DEFAULT_REWARD_CONFIG } from './environment.js';
import { WindyGridEnvironment } from './windyGridEnvironment.js';
import { MovingGoalEnvironment } from './movingGoalEnvironment.js';
import { RewardGridEnvironment } from './rewardGridEnvironment.js';

export const DEFAULT_SCENARIO_ID = 'classic';

function cloneObstacles(obstacles = []) {
  return obstacles.map(obstacle => ({ x: obstacle.x, y: obstacle.y }));
}

function deepClone(value) {
  if (value === undefined || value === null) {
    return value;
  }
  return JSON.parse(JSON.stringify(value));
}

function mergeRewards(defaults = {}, overrides) {
  const base = { ...DEFAULT_REWARD_CONFIG, ...defaults };
  if (!overrides || typeof overrides !== 'object') {
    return base;
  }
  const merged = { ...base };
  for (const key of Object.keys(base)) {
    if (overrides[key] !== undefined) {
      const value = Number(overrides[key]);
      if (Number.isFinite(value)) {
        merged[key] = value;
      }
    }
  }
  return merged;
}

function createWindColumns(size) {
  if (size <= 2) {
    return [];
  }
  const middleStart = Math.max(1, Math.floor(size / 2) - 1);
  const middleEnd = Math.min(size - 2, middleStart + 1);
  const columns = [];
  for (let x = middleStart; x <= middleEnd; x += 1) {
    columns.push({
      x,
      offsets: [
        { dx: 0, dy: -1, weight: 0.7 },
        { dx: 1, dy: 0, weight: 0.2 },
        { dx: 0, dy: 0, weight: 0.1 }
      ]
    });
  }
  if (size > 3) {
    const gustColumn = Math.min(size - 2, middleEnd + 1);
    columns.push({
      x: gustColumn,
      offsets: [
        { dx: 0, dy: -1, weight: 0.5 },
        { dx: -1, dy: -1, weight: 0.2 },
        { dx: 0, dy: 0, weight: 0.3 }
      ]
    });
  }
  return columns;
}

function createMovingGoalPattern(size) {
  const max = size - 1;
  const mid = Math.max(0, Math.floor(max / 2));
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

function createRewardCells(size) {
  const max = size - 1;
  const mid = Math.max(0, Math.floor(max / 2));
  const nearMid = Math.max(0, mid - 1);
  return [
    { x: mid, y: mid, reward: 0.4 },
    { x: mid, y: nearMid, reward: 0.25 },
    { x: nearMid, y: mid, reward: -0.3 },
    { x: max, y: nearMid, reward: 0.6 }
  ];
}

const scenarioDefinitions = [
  {
    id: 'classic',
    label: 'Classic Grid',
    description: 'Deterministic grid world with a static goal.',
    defaultSize: 5,
    defaultRewards: {},
    defaultObstacles: () => [],
    create: ({ size, obstacles, rewards }) => new GridWorldEnvironment(size, cloneObstacles(obstacles), rewards)
  },
  {
    id: 'windy',
    label: 'Windy Pass',
    description: 'Stochastic wind columns push the agent off course.',
    defaultSize: 7,
    defaultRewards: { stepPenalty: -0.04, obstaclePenalty: -0.2 },
    defaultScenarioConfig: size => ({ windColumns: createWindColumns(size) }),
    create: ({ size, obstacles, rewards, scenarioConfig }) => {
      const config = scenarioConfig?.windColumns ? scenarioConfig : { windColumns: createWindColumns(size) };
      return new WindyGridEnvironment(size, cloneObstacles(obstacles), rewards, config.windColumns);
    }
  },
  {
    id: 'moving-goal',
    label: 'Moving Target',
    description: 'Goal positions cycle during an episode.',
    defaultSize: 6,
    defaultRewards: { stepPenalty: -0.02, goalReward: 1.2 },
    defaultScenarioConfig: size => ({ goalPattern: createMovingGoalPattern(size), moveFrequency: 8 }),
    create: ({ size, obstacles, rewards, scenarioConfig }) => new MovingGoalEnvironment(
      size,
      cloneObstacles(obstacles),
      rewards,
      scenarioConfig ?? { goalPattern: createMovingGoalPattern(size), moveFrequency: 8 }
    )
  },
  {
    id: 'reward-grid',
    label: 'Treasure Fields',
    description: 'Sparse rewards and penalties scattered across the map.',
    defaultSize: 6,
    defaultRewards: { obstaclePenalty: -0.15, goalReward: 1.5 },
    defaultScenarioConfig: size => ({ rewardCells: createRewardCells(size) }),
    create: ({ size, obstacles, rewards, scenarioConfig }) => new RewardGridEnvironment(
      size,
      cloneObstacles(obstacles),
      rewards,
      scenarioConfig?.rewardCells ?? createRewardCells(size)
    )
  }
];

const scenarioMap = new Map(scenarioDefinitions.map(scenario => [scenario.id, scenario]));

function resolveScenarioOptions(scenario, options = {}) {
  const rawSize = Number.isFinite(options.size) ? Math.trunc(options.size) : scenario.defaultSize ?? 5;
  const size = Math.max(2, rawSize);
  const rewards = mergeRewards(scenario.defaultRewards, options.rewards);
  const obstacles = Array.isArray(options.obstacles)
    ? cloneObstacles(options.obstacles)
    : (typeof scenario.defaultObstacles === 'function'
      ? cloneObstacles(scenario.defaultObstacles(size))
      : cloneObstacles(scenario.defaultObstacles ?? []));
  const scenarioConfig = options.scenarioConfig
    ? deepClone(options.scenarioConfig)
    : (typeof scenario.defaultScenarioConfig === 'function'
      ? deepClone(scenario.defaultScenarioConfig(size))
      : deepClone(scenario.defaultScenarioConfig));
  return { size, rewards, obstacles, scenarioConfig };
}

export function getScenarioDefinitions() {
  return scenarioDefinitions.map(({ id, label, description }) => ({ id, label, description }));
}

export function getScenarioById(id) {
  return scenarioMap.get(id) ?? scenarioMap.get(DEFAULT_SCENARIO_ID);
}

export function createEnvironmentFromScenario(id, options = {}) {
  const scenario = getScenarioById(id);
  const resolved = resolveScenarioOptions(scenario, options);
  const env = scenario.create(resolved);
  env.scenarioId = scenario.id;
  return env;
}
