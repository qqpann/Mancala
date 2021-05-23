import random
from typing import Union

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class MaxAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, id: int, deterministic: bool = False, seed=42):
        self.id = id
        self._seed = seed
        self.deterministic = deterministic

    @staticmethod
    def _score_of_action(act, state: BaseState) -> float:
        turn = state.current_player
        state.proceed_action(act)
        reward = state.rewards[turn]
        return reward

    def policy(self, state: BaseState) -> Union[None, int]:
        """
        Make a move.

        Params
        ---
        state: mancala state object

        Returns
        ---
        action: int
        """
        legal_actions = state.legal_actions(state.current_player)
        if legal_actions is None:
            return None
        action_rewards = [
            MaxAgent._score_of_action(a, state.clone()) for a in legal_actions
        ]
        max_reward = max(action_rewards)
        max_actions = [
            a for a, r in zip(legal_actions, action_rewards) if r == max_reward
        ]
        if self.deterministic:
            random.seed(self._seed)
        return random.choice(max_actions)
