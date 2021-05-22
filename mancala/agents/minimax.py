from typing import List, Union
import random
import sys

import numpy as np

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


def minimax(state: BaseState, depth: int, maximizing_player_id: int) -> float:
    """
    MiniMax function
    """
    # Ref: https://en.wikipedia.org/wiki/Minimax
    if depth == 0 or state.is_terminal():
        return state.rewards_float(maximizing_player_id)

    if state.turn == maximizing_player_id:
        value = -float("inf")
        for act in state.legal_actions(state.turn):
            child = state.clone()
            child.proceed_action(act)
            value = max(value, minimax(child, depth - 1, child.turn))
        return value
    else:
        value = float("inf")
        for act in state.legal_actions(state.turn):
            child = state.clone()
            child.proceed_action(act)
            value = min(value, minimax(child, depth - 1, child.turn))
        return value


class MiniMaxAgent(BaseAgent):
    """
    Agent based on mini-max algorithm
    """

    def __init__(self, id: int):
        self.deterministic = False
        self._seed = 42
        self._depth = 4
        self.id = id

    def policy(self, state: BaseState) -> int:
        legal_actions = state.legal_actions(state.current_player)
        action_rewards = [
            minimax(state.clone().proceed_action(a), 4, self.id) for a in legal_actions
        ]
        print(legal_actions)
        print(action_rewards)
        max_reward = max(action_rewards)
        max_actions = [
            a for a, r in zip(legal_actions, action_rewards) if r == max_reward
        ]
        if self.deterministic:
            random.seed(self._seed)
        return random.choice(max_actions)