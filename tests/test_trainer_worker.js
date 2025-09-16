import { Worker } from 'node:worker_threads';

function waitFor(worker, predicate, timeout = 1000) {
  return new Promise((resolve, reject) => {
    const removeListener = worker.off?.bind(worker) || worker.removeListener.bind(worker);
    const timer = setTimeout(() => {
      removeListener('message', onMessage);
      reject(new Error('Timed out waiting for worker message'));
    }, timeout);
    const onMessage = message => {
      try {
        if (predicate(message)) {
          clearTimeout(timer);
          removeListener('message', onMessage);
          resolve(message);
        }
      } catch (err) {
        clearTimeout(timer);
        removeListener('message', onMessage);
        reject(err);
      }
    };
    worker.on('message', onMessage);
  });
}

export async function run(assert) {
  const worker = new Worker(new URL('../src/rl/trainerWorker.js', import.meta.url), {
    type: 'module'
  });
  try {
    worker.postMessage({
      type: 'config',
      payload: {
        agent: {
          type: 'rl',
          params: { epsilon: 0.5, learningRate: 0.1 },
          revision: 0
        },
        env: {
          size: 2,
          obstacles: []
        },
        trainer: {
          intervalMs: 5
        }
      }
    });

    const initial = await waitFor(
      worker,
      msg => msg.type === 'progress' && msg.payload.metrics.steps === 0
    );
    assert.strictEqual(initial.payload.metrics.episode, 1);
    assert.strictEqual(initial.payload.metrics.cumulativeReward, 0);
    assert.strictEqual(initial.payload.metrics.epsilon.toFixed(2), '0.50');

    worker.postMessage({ type: 'start' });
    const progressed = await waitFor(
      worker,
      msg => msg.type === 'progress' && msg.payload.metrics.steps > 0
    );
    assert.ok(progressed.payload.metrics.steps > 0);

    worker.postMessage({
      type: 'config',
      payload: {
        agent: {
          type: 'rl',
          params: { epsilon: 0.5, learningRate: 0.1 },
          revision: 0
        },
        env: {
          size: 2,
          obstacles: [{ x: 0, y: 1 }]
        },
        trainer: {
          intervalMs: 5
        }
      }
    });
    const requestId = 1234;
    worker.postMessage({
      type: 'agent:getState',
      payload: { requestId }
    });
    await waitFor(
      worker,
      msg => msg.type === 'agent:state' && msg.payload?.requestId === requestId
    );
    const continued = await waitFor(
      worker,
      msg => msg.type === 'progress'
        && msg.payload.metrics.steps > 0
        && msg.payload.metrics.episode >= progressed.payload.metrics.episode
    );
    assert.ok(continued.payload.metrics.steps > 0);

    worker.postMessage({ type: 'pause' });
    worker.postMessage({ type: 'resetTrainerState' });
    const reset = await waitFor(
      worker,
      msg => msg.type === 'progress' && msg.payload.metrics.steps === 0
    );
    assert.ok(reset.payload.metrics.episode >= 1);

    worker.postMessage({ type: 'agent:update', payload: { epsilon: 0.25 } });
    const updated = await waitFor(
      worker,
      msg => msg.type === 'progress' && msg.payload.metrics.epsilon <= 0.26
    );
    assert.strictEqual(updated.payload.metrics.epsilon.toFixed(2), '0.25');
  } finally {
    await worker.terminate();
  }
}
