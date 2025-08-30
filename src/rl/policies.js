export const POLICIES = {
  EPSILON_GREEDY: 'epsilon-greedy',
  GREEDY: 'greedy',
  SOFTMAX: 'softmax',
  THOMPSON: 'thompson',
  UCB: 'ucb',
  RANDOM: 'random'
};

export function selectAction(policy, agent, state, qVals, update = true) {
  switch (policy) {
    case POLICIES.GREEDY:
      return agent.bestAction(qVals);
    case POLICIES.SOFTMAX:
      return agent._softmax(qVals);
    case POLICIES.THOMPSON:
      return agent._thompson(state, qVals, update);
    case POLICIES.UCB:
      return agent._ucb(state, qVals, update);
    case POLICIES.RANDOM:
      return agent._random();
    case POLICIES.EPSILON_GREEDY:
    default:
      return agent._epsilonGreedy(qVals);
  }
}
