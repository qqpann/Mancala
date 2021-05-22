from mancala.agents.negascout import NegaScoutAgent
from mancala.agents.minimax import MiniMaxAgent
from mancala.agents.base import BaseAgent
from mancala.agents.exact import ExactAgent
from mancala.agents.human import HumanAgent
from mancala.agents.max import MaxAgent
from mancala.agents.random import RandomAgent

ALL_AI_AGENTS = ["random", "exact", "max", "minimax", "negascout"]


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
    else:
        raise ValueError