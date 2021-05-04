from mancala.agents.random_agent import RandomAgent
from mancala.mancala import MancalaEnv


def cli():
    mancala = MancalaEnv(RandomAgent())
    mancala.play_cli()


if __name__ == "__main__":
    cli()
