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
    return alphabeta(state, depth, maximizing_player_id, -float("inf"), float("inf"))


def alphabeta(
    state: BaseState, depth: int, maximizing_player_id: int, alpha: float, beta: float
) -> float:
    """
    MiniMax with alpha-beta pruning
    """
    # Ref: https://en.wikipedia.org/wiki/Alpha–beta_pruning
    if depth == 0 or state.is_terminal():
        return state.rewards_float(maximizing_player_id)

    legal_actions = state.legal_actions(state.current_player)
    if legal_actions is None:
        # return state.rewards_float(1 - state.turn)
        return alphabeta(
            state.proceed_action(None), depth - 1, maximizing_player_id, alpha, beta
        )
    if state.turn == maximizing_player_id:
        for act in legal_actions:
            child = state.clone()
            child.proceed_action(act)
            alpha = max(
                alpha, alphabeta(child, depth - 1, maximizing_player_id, alpha, beta)
            )
            if alpha >= beta:
                break
        return alpha
    else:
        for act in legal_actions:
            child = state.clone()
            child.proceed_action(act)
            beta = min(
                beta, alphabeta(child, depth - 1, maximizing_player_id, alpha, beta)
            )
            if alpha >= beta:
                break
        return beta


class MiniMaxAgent(BaseAgent):
    """
    Agent based on mini-max algorithm
    """

    def __init__(self, id: int, depth: int = 2):
        self.deterministic = False
        self._seed = 42
        self._depth = depth
        self.id = id

    def policy(self, state: BaseState) -> Union[int, None]:
        legal_actions = state.legal_actions(state.current_player)
        if legal_actions is None:
            return None
        action_rewards = [
            minimax(state.clone().proceed_action(a), self._depth, state.current_player)
            for a in legal_actions
        ]
        # print(legal_actions)
        # print(action_rewards)
        max_reward = max(action_rewards)
        max_actions = [
            a for a, r in zip(legal_actions, action_rewards) if r == max_reward
        ]
        if self.deterministic:
            random.seed(self._seed)
        return random.choice(max_actions)