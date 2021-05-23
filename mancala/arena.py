from collections import defaultdict
import time
from pprint import pprint
from typing import DefaultDict, List
from tqdm import tqdm

from pandas import DataFrame
from mancala.agents import ALL_AI_AGENTS
from mancala.game import CLIGame
from mancala.mancala import MancalaEnv


def play_arena(agents: List[str] = ALL_AI_AGENTS, fights: int = 100):
    wins: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    times: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for agent_name0 in agents:
        for agent_name1 in agents:
            p1wins = 0
            timestamp = time.time()
            for _ in tqdm(
                range(fights), desc=f"p0 {agent_name0} vs p1 {agent_name1}", leave=False
            ):
                env = MancalaEnv([agent_name0, agent_name1])
                game = CLIGame(env, silent=True)
                winner = game.play_silent()
                p1wins += winner
            wins["p0_" + agent_name0]["p1_" + agent_name1] = p1wins / fights * 100
            times["p0_" + agent_name0]["p1_" + agent_name1] += time.time() - timestamp

    return wins, times


if __name__ == "__main__":
    wins, _ = play_arena()
    print(DataFrame(wins))
