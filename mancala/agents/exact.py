import random
from typing import Union

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class ExactAgent(BaseAgent):
    """Agent to use continue-on-point rule"""

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

    @staticmethod
    def _turn_kept_by_action(state: BaseState, act: int) -> bool:
        turn = state.current_player
        state.proceed_action(act)
        kept = state.current_player == turn
        return kept

    def policy(self, state: BaseState) -> Union[int, None]:
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
        action_turn_kepts = [
            ExactAgent._turn_kept_by_action(state.clone(), a) for a in legal_actions
        ]
        exact_actions = [a for a, kept in zip(legal_actions, action_turn_kepts) if kept]
        if self.deterministic:
            random.seed(self._seed)
        if len(exact_actions) > 0:
            return random.choice(exact_actions)
        else:
            # action_rewards = [
            #     ExactAgent._score_of_action(a, state.clone()) for a in legal_actions
            # ]
            return random.choice(legal_actions)
