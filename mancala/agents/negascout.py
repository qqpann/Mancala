from typing import List, Union
import random

import numpy as np

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


def negamax(state: BaseState, depth: int, maximizing_player_id: int) -> float:
    color = 1 if maximizing_player_id == state.turn else -1
    if depth == 0 or state.is_terminal():
        return color * state.rewards_float(state.turn)
    value = -float("inf")
    legal_actions = state.legal_actions(state.turn)
    if legal_actions is None:
        return state.rewards_float(1 - state.turn)
    for act in legal_actions:
        child = state.clone().proceed_action(act)
        value = max(value, negamax(child, depth - 1, maximizing_player_id))
    return -value


def negascout(state: BaseState, depth: int, maximizing_player_id: int):
    return pvs(state, depth, maximizing_player_id, -float("inf"), float("inf"))


def pvs(
    state: BaseState, depth: int, maximizing_player_id: int, alpha: float, beta: float
) -> float:
    """
    Principal variation search (PVS), also known as NegaScout
    alpha:  minimum score that the maximizing player is assured of
    beta: maximum score that the minimizing player is assured of
    """
    color = 1 if maximizing_player_id == state.turn else -1
    # Ref: https://en.wikipedia.org/wiki/Principal_variation_search
    if depth == 0 or state.is_terminal():
        return state.rewards_float(maximizing_player_id)

    legal_actions = state.legal_actions(state.turn)
    if legal_actions is None:
        return state.rewards_float(1 - state.turn)
    sorted_actions = legal_actions.copy()
    # The search order should be small to large idx, since closer to point pocket is more important
    for act in legal_actions:
        if state._can_continue_on_point(act):
            sorted_actions.insert(0, sorted_actions.pop(sorted_actions.index(act)))

    for i, act in enumerate(sorted_actions):
        child = state.clone().proceed_action(act)
        if i == 0:
            score = -pvs(child, depth - 1, maximizing_player_id, -beta, -alpha)
        else:
            score = -pvs(child, depth - 1, maximizing_player_id, -alpha - 0.01, -alpha)

            if alpha <= score <= beta:
                score = -pvs(child, depth - 1, maximizing_player_id, -beta, -score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return alpha


class NegaScoutAgent(BaseAgent):
    """
    Agent based on mini-max algorithm
    """

    def __init__(self, id: int, depth: int = 2):
        self.deterministic = False
        self._seed = 42
        self._depth = depth
        self.id = id

    def policy(self, state: BaseState) -> Union[int, None]:
        assert self.id == state.current_player
        legal_actions = state.legal_actions(state.current_player)
        if legal_actions is None:
            return None
        action_rewards = [
            negascout(state.clone().proceed_action(a), self._depth, self.id)
            for a in legal_actions
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