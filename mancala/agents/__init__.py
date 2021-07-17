from typing import List
import numpy as np

from mancala.agents.a3c.agent import A3CAgent
from mancala.agents.base import BaseAgent
from mancala.agents.exact import ExactAgent
from mancala.agents.human import HumanAgent
from mancala.agents.max import MaxAgent
from mancala.agents.minimax import MiniMaxAgent
from mancala.agents.negascout import NegaScoutAgent
from mancala.agents.random import RandomAgent

ALL_AI_AGENTS = ["random", "exact", "max", "minimax", "negascout", "a3c"]
ARENA_AI_AGENTS = ["random", "max", "negascout", "a3c"]


def init_agent(agent_type: str, id: int) -> BaseAgent:
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
        return MiniMaxAgent(id)
    elif agent_type == "negascout":
        return NegaScoutAgent(id)
    elif agent_type == "a3c":
        return A3CAgent(id)
    else:
        raise ValueError


def init_random_agent(id: int, choices: List[str], weights: List[float]):
    name = np.random.choice(choices, 1, weights)
    return init_agent(name, id)
