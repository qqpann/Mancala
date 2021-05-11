import argparse

import pandas

from mancala.agents import ALL_AI_AGENTS, HumanAgent, RandomAgent
from mancala.arena import play_arena
from mancala.game import CLIGame
from mancala.mancala import MancalaEnv

parser = argparse.ArgumentParser(description="Mancala playable on cli")
subparsers = parser.add_subparsers(dest="command")

play_parser = subparsers.add_parser("play")
play_parser.add_argument(
    "--player0",
    type=str,
    default="human",
    help="Player that makes the first move",
    choices=ALL_AI_AGENTS + ["human"],
)
play_parser.add_argument(
    "--player1",
    type=str,
    default="random",
    help="Player that makes move the next turn",
    choices=ALL_AI_AGENTS + ["human"],
)

arena_parser = subparsers.add_parser("arena")
arena_parser.add_argument(
    "--num_games",
    type=int,
    default=100,
    help="How many times each pairs of agents should play together",
)


def cli():
    args = parser.parse_args()
    if args.command == "play":
        env = MancalaEnv([args.player0, args.player1])
        game = CLIGame(env)
        game.play_cli()
    elif args.command == "arena":
        wins = play_arena(args.num_games)
        print(pandas.DataFrame(wins))
