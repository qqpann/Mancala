import time
from collections import defaultdict
from pprint import pprint
from typing import DefaultDict, List, Tuple

from pandas import DataFrame
from tqdm import tqdm

from mancala.agents import ALL_AI_AGENTS, init_agent
from mancala.agents.base import BaseAgent
from mancala.game import CLIGame
from mancala.mancala import MancalaEnv


def play_one_game(agent0: BaseAgent, agent1: BaseAgent) -> int:
    env = MancalaEnv(agent0, agent1)
    game = CLIGame(env, silent=True)
    winner = game.play_silent()
    return winner


def play_games(agent0: BaseAgent, agent1: BaseAgent, fights: int):
    p1wins = 0
    for _ in tqdm(range(fights), desc=f"p0 {agent0} vs p1 {agent1}", leave=False):
        p1wins += play_one_game(agent0, agent1)
    return p1wins / fights * 100


def play_arena(agents: List[str] = ALL_AI_AGENTS, fights: int = 100):
    wins: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    times: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for agent_name0 in agents:
        for agent_name1 in agents:
            timestamp = time.time()
            agent0 = init_agent(agent_name0, 0)
            agent1 = init_agent(agent_name1, 1)
            p1win_rate = play_games(agent0, agent1, fights)
            wins["p0_" + agent_name0]["p1_" + agent_name1] = p1win_rate
            times["p0_" + agent_name0]["p1_" + agent_name1] += time.time() - timestamp

    return wins, times


if __name__ == "__main__":
    wins, _ = play_arena()
    print(DataFrame(wins))
