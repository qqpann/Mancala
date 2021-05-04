from mancala.agents.random_agent import RandomAgent
from mancala.mancala import MancalaEnv


def play_cli():
    mancala = MancalaEnv(RandomAgent())
    mancala.play_cli()


if __name__ == "__main__":
    play_cli()
