import { createEnvironmentFromScenario } from '../src/rl/environmentPresets.js';
import { saveEnvironment, loadEnvironment } from '../src/rl/storage.js';

function createStorage() {
  const store = new Map();
  return {
    setItem(key, value) {
      store.set(String(key), String(value));
    },
    getItem(key) {
      return store.has(String(key)) ? store.get(String(key)) : null;
    },
    removeItem(key) {
      store.delete(String(key));
    },
    clear() {
      store.clear();
    }
  };
}

export async function run(assert) {
  const windyEnv = createEnvironmentFromScenario('windy');
  const windyStorage = createStorage();
  saveEnvironment(windyEnv, windyStorage);
  const loadedWindy = loadEnvironment(windyStorage);
  assert.strictEqual(loadedWindy.scenarioId, 'windy');
  assert.ok(Array.isArray(loadedWindy.scenarioConfig.windColumns));
  assert.ok(loadedWindy.scenarioConfig.windColumns.length > 0);
  const windyRehydrated = createEnvironmentFromScenario(loadedWindy.scenarioId, {
    size: loadedWindy.size,
    obstacles: loadedWindy.obstacles,
    rewards: loadedWindy.rewards,
    scenarioConfig: loadedWindy.scenarioConfig
  });
  const firstWindColumn = loadedWindy.scenarioConfig.windColumns[0];
  const windyMetadata = windyRehydrated.describeCell(firstWindColumn.x, 0);
  assert.ok(Array.isArray(windyMetadata.classes) && windyMetadata.classes.includes('wind-zone'));

  const movingEnv = createEnvironmentFromScenario('moving-goal');
  const movingStorage = createStorage();
  saveEnvironment(movingEnv, movingStorage);
  const loadedMoving = loadEnvironment(movingStorage);
  assert.strictEqual(loadedMoving.scenarioId, 'moving-goal');
  assert.ok(Array.isArray(loadedMoving.scenarioConfig.goalPattern));
  assert.ok(loadedMoving.scenarioConfig.goalPattern.length > 0);
  const movingRehydrated = createEnvironmentFromScenario(loadedMoving.scenarioId, {
    size: loadedMoving.size,
    obstacles: loadedMoving.obstacles,
    rewards: loadedMoving.rewards,
    scenarioConfig: loadedMoving.scenarioConfig
  });
  const goal = movingRehydrated.getGoalPosition();
  assert.ok(loadedMoving.scenarioConfig.goalPattern.some(pos => pos.x === goal.x && pos.y === goal.y));

  const rewardEnv = createEnvironmentFromScenario('reward-grid');
  const rewardStorage = createStorage();
  saveEnvironment(rewardEnv, rewardStorage);
  const loadedReward = loadEnvironment(rewardStorage);
  assert.strictEqual(loadedReward.scenarioId, 'reward-grid');
  assert.ok(Array.isArray(loadedReward.scenarioConfig.rewardCells));
  assert.ok(loadedReward.scenarioConfig.rewardCells.length > 0);
  const rewardRehydrated = createEnvironmentFromScenario(loadedReward.scenarioId, {
    size: loadedReward.size,
    obstacles: loadedReward.obstacles,
    rewards: loadedReward.rewards,
    scenarioConfig: loadedReward.scenarioConfig
  });
  const rewardCell = loadedReward.scenarioConfig.rewardCells[0];
  const rewardMetadata = rewardRehydrated.describeCell(rewardCell.x, rewardCell.y);
  assert.ok(Array.isArray(rewardMetadata.classes));
  const expectedClass = rewardCell.reward >= 0 ? 'reward-bonus' : 'reward-penalty';
  assert.ok(rewardMetadata.classes.includes(expectedClass));
  assert.strictEqual(rewardMetadata.reward, rewardCell.reward);
}
