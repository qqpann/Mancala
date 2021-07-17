import random
from typing import Dict, List, Union
from gym.utils import seeding

import numpy as np

from mancala.agents import init_agent
from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState

WEIGHTED_AGENTS = {"random": 1 / 20, "exact": 9 / 20, "minimax": 10 / 20}


class MixedAgent(BaseAgent):
    """
    Mixed types of agents
    """

    def __init__(
        self,
        id: int,
        depth: int = 2,
        weighted_agents: Dict[str, float] = WEIGHTED_AGENTS,
    ):
        self.deterministic = False
        self._seed = 42
        self._depth = depth
        self.id = id
        agents = []
        weights = []
        for agent, weight in weighted_agents.items():
            agents.append(init_agent(agent, id))
            weights.append(weight)
        self.agents = agents
        self.weights = weights
        self.np_random, seed = seeding.np_random(self._seed)

    def _agent(self) -> BaseAgent:
        return self.np_random.choice(self.agents, 1, p=self.weights)[0]

    def policy(self, state: BaseState) -> Union[int, None]:
        assert self.id == state.current_player
        legal_actions = state.legal_actions(state.current_player)
        if legal_actions is None:
            return None
        return self._agent().policy(state)