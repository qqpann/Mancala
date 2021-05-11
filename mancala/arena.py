from collections import defaultdict
from pprint import pprint
from typing import DefaultDict

import pandas as pd

from mancala.game import CLIGame
from mancala.mancala import ALL_AI_AGENTS, MancalaEnv


def main(fights: int = 200):
    wins: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for agent_name0 in ALL_AI_AGENTS:
        for agent_name1 in ALL_AI_AGENTS:
            p1wins = 0
            for _ in range(fights):
                env = MancalaEnv([agent_name0, agent_name1])
                game = CLIGame(env, silent=True)
                winner = game.play_silent()
                p1wins += winner
            wins["p0_" + agent_name0]["p1_" + agent_name1] = p1wins / fights * 100

    return wins


if __name__ == "__main__":
    wins = main()
    print(pd.DataFrame(wins))
