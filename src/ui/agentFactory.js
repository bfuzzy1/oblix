import { RLAgent } from '../rl/agent.js';
import { POLICIES } from '../rl/policies.js';
import { SarsaAgent } from '../rl/sarsaAgent.js';
import { ExpectedSarsaAgent } from '../rl/expectedSarsaAgent.js';
import { DynaQAgent } from '../rl/dynaQAgent.js';
import { QLambdaAgent } from '../rl/qLambdaAgent.js';
import { MonteCarloAgent } from '../rl/monteCarloAgent.js';
import { ActorCriticAgent } from '../rl/actorCriticAgent.js';
import { DoubleQAgent } from '../rl/doubleQAgent.js';
import { OptimisticAgent } from '../rl/optimisticAgent.js';

const agentFactory = {
  rl: RLAgent,
  sarsa: SarsaAgent,
  expected: ExpectedSarsaAgent,
  dyna: DynaQAgent,
  qlambda: QLambdaAgent,
  mc: MonteCarloAgent,
  ac: ActorCriticAgent,
  double: DoubleQAgent,
  optimistic: OptimisticAgent
};

export function createAgent(type, options = {}) {
  const defaults = {
    epsilon: 1,
    epsilonDecay: 0.995,
    minEpsilon: 0.05,
    policy: POLICIES.EPSILON_GREEDY,
    lambda: 0
  };
  const AgentClass = agentFactory[type] || RLAgent;
  const resolvedType = agentFactory[type] ? type : 'rl';
  const agent = new AgentClass({ ...defaults, ...options });
  Object.defineProperty(agent, '__factoryType', {
    value: resolvedType,
    writable: true,
    configurable: true,
    enumerable: false
  });
  return agent;
}

export { agentFactory };
