import argparse

from pandas import DataFrame

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
arena_parser.add_argument(
    "--only",
    type=str,
    default="",
    help=f"Explicitly select agents to compare from {ALL_AI_AGENTS}",
)


def cli():
    args = parser.parse_args()
    if args.command == "play":
        env = MancalaEnv([args.player0, args.player1])
        game = CLIGame(env)
        game.play_cli()
    elif args.command == "arena":
        agents = ALL_AI_AGENTS
        if args.only:
            agents = args.only.split(",")
        wins, times = play_arena(agents, args.num_games)
        print("Wins (percent for p1 to win):")
        print(DataFrame(wins))
        print()
        print("Time:")
        print(DataFrame(times))
    elif args.command == "train":
        pass
