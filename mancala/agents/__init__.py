import random
from typing import Dict, List, Union

import numpy as np
from gym.utils import seeding

from mancala.agents.a3c.agent import A3CAgent
from mancala.agents.base import BaseAgent
from mancala.agents.exact import ExactAgent
from mancala.agents.human import HumanAgent
from mancala.agents.max import MaxAgent
from mancala.agents.minimax import MiniMaxAgent
from mancala.agents.negascout import NegaScoutAgent
from mancala.agents.random import RandomAgent
from mancala.state.base import BaseState

ALL_AI_AGENTS = ["random", "exact", "max", "minimax", "negascout", "a3c", "mixed"]
ARENA_AI_AGENTS = ["random", "max", "negascout", "a3c"]


def init_agent(agent_type: str, id: int, depth: int = 2) -> BaseAgent:
    assert agent_type in (ALL_AI_AGENTS + ["human"])
    if agent_type == "human":
        return HumanAgent(id)
    elif agent_type == "random":
        return RandomAgent(id)
    elif agent_type == "exact":
        return ExactAgent(id)
    elif agent_type == "max":
        return MaxAgent(id)
    elif agent_type == "minimax":
        return MiniMaxAgent(id, depth)
    elif agent_type == "negascout":
        return NegaScoutAgent(id, depth)
    elif agent_type == "a3c":
        return A3CAgent(id)
    elif agent_type == "mixed":
        return MixedAgent(id)
    else:
        raise ValueError


def init_random_agent(
    id: int, choices: List[str], weights: List[float], depth: int = 2
):
    name = np.random.choice(choices, 1, weights)
    return init_agent(name, id, depth)


WEIGHTED_AGENTS = {"random": 1 / 20, "max": 9 / 20, "minimax": 10 / 20}
MIXED_AGENTS = [k for k, _ in WEIGHTED_AGENTS.items()]
MIXED_WEIGHTS = [v for _, v in WEIGHTED_AGENTS.items()]


class MixedAgent(BaseAgent):
    """
    Mixed types of agents
    """

    def __init__(
        self,
        id: int,
        agent_names: List[str] = MIXED_AGENTS,
        weights: List[float] = MIXED_WEIGHTS,
    ):
        assert len(agent_names) == len(weights)
        self._seed = 42
        self.np_random, seed = seeding.np_random(self._seed)
        self.agents = [init_agent(a, id) for a in agent_names]
        self.weights = weights
        self.set_id(id)

    def _agent(self) -> BaseAgent:
        return self.np_random.choice(self.agents, 1, p=self.weights)[0]

    def set_id(self, id):
        super().set_id(id)
        for agent in self.agents:
            agent.set_id(id)

    def policy(self, state: BaseState) -> Union[int, None]:
        assert self.id == state.current_player
        legal_actions = state.legal_actions(state.current_player)
        if legal_actions is None:
            return None
        return self._agent().policy(state)