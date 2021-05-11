import argparse

from mancala.agents import HumanAgent, RandomAgent
from mancala.game import CLIGame
from mancala.mancala import MancalaEnv

parser = argparse.ArgumentParser(description="Mancala playable on cli")

parser.add_argument(
    "--player0", type=str, default="human", help="Player that makes the first move"
)
parser.add_argument(
    "--player1", type=str, default="random", help="Player that makes move the next turn"
)


def play_cli():
    args = parser.parse_args()

    env = MancalaEnv([args.player0, args.player1])
    game = CLIGame(env)
    game.play_cli()


if __name__ == "__main__":
    play_cli()
