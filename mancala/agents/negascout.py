from typing import List, Union
import random

import numpy as np

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


def pvs(
    state: BaseState, depth: int, maximizing_player_id: int, alpha: float, beta: float
) -> float:
    """
    Principal variation search (PVS), also known as NegaScout
    alpha:  minimum score that the maximizing player is assured of
    beta: maximum score that the minimizing player is assured of
    """
    # Ref: https://en.wikipedia.org/wiki/Principal_variation_search
    if depth == 0 or state.is_terminal():
        return state.rewards_float(maximizing_player_id)

    actions = state.legal_actions(state.turn)
    sorted_actions = actions.copy()
    # The search order should be small to large idx, since closer to point pocket is more important
    for act in actions:
        if state._can_continue_on_point(act):
            sorted_actions.insert(0, sorted_actions.pop(sorted_actions.index(act)))

    for i, act in enumerate(sorted_actions):
        child = state.clone().proceed_action(act)
        if i == 0:
            if child.turn != state.turn:
                score = -pvs(child, depth - 1, maximizing_player_id, -beta, -alpha)
            else:
                score = pvs(child, depth - 1, maximizing_player_id, alpha, beta)
        else:
            if child.turn != state.turn:
                score = -pvs(
                    child, depth - 1, maximizing_player_id, -alpha - 0.01, -alpha
                )
            else:
                score = pvs(child, depth - 1, maximizing_player_id, alpha, beta)

            if alpha <= score <= beta:
                if child.turn != state.turn:
                    score = -pvs(child, depth - 1, maximizing_player_id, -beta, -score)
                else:
                    # score = pvs(child, depth - 1, maximizing_player_id, alpha, beta)
                    pass
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return alpha


class NegaScoutAgent(BaseAgent):
    """
    Agent based on mini-max algorithm
    """

    def __init__(self, id: int, depth: int = 4):
        self.deterministic = False
        self._seed = 42
        self._depth = depth
        self.id = id

    def policy(self, state: BaseState) -> int:
        assert self.id == state.current_player
        legal_actions = state.legal_actions(self.id)
        action_rewards = [
            pvs(
                state.clone().proceed_action(a),
                self._depth,
                self.id,
                -float("inf"),
                float("inf"),
            )
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