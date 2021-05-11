# https://github.com/openai/gym/blob/master/docs/creating-environments.md
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import gym
import numpy as np
from gym import Env, error, spaces, utils
from gym.utils import seeding

from mancala.agents import BaseAgent, init_agent
from mancala.state.base import BaseState


@dataclass
class Rule:
    multi_lap: bool = True
    capture_opposite: bool = True
    continue_on_point: bool = True
    pockets: int = 6
    initial_stones: int = 4
    stones_half: int = 6 * 4


turn_names = ["player0", "player1"]


class MancalaState(BaseState):
    """
    Mancala State
    ---
    The board state and its utils
    """

    def __init__(
        self,
        board: Union[np.ndarray, None] = None,
        turn: int = 0,  # player: 0, ai: 1
    ):
        self.rule = Rule()
        if board is not None:
            assert board.shape == ((self.rule.pockets + 1) * 2,)
            self.board = board
        else:
            self.board = MancalaState.init_board(self.rule)
        self.turn = turn

        self.hand = 0
        self.action_choices = [str(i) for i in range(1, self.rule.pockets + 1)]

    @staticmethod
    def init_board(rule: Rule) -> np.ndarray:
        board = np.zeros(((rule.pockets + 1) * 2,), dtype=np.int32)
        # Player 1 side
        for i in range(0, rule.pockets):
            board[i] = rule.initial_stones
        # Player 2 side
        for i in range(rule.pockets + 1, rule.pockets * 2 + 1):
            board[i] = rule.initial_stones
        return board

    def legal_actions(self, turn: int) -> List[int]:
        if turn == 0:
            all_actions = list(self._player0_field_range)
        else:
            all_actions = list(self._player1_field_range)
        return self.filter_available_actions(all_actions)

    @property
    def current_player(self) -> int:
        return self.turn

    def __repr__(self):
        return f"<MancalaState: [{self.board}, {self.turn}]>"

    def clone(self) -> MancalaState:
        return MancalaState(board=self.board.copy(), turn=self.turn)

    def get_reward(self, turn: Union[int, None] = None) -> int:
        point = self.board[self._active_player_point_index]
        return point

    @property
    def rewards(self):
        return [
            self.board[self._player0_point_index],
            self.board[self._player1_point_index],
        ]

    def take_pocket(self, idx: int) -> None:
        """
        Params
        idx: index of the pocket to manipulate
        """
        self.hand += self.board[idx]
        self.board[idx] = 0

    def fill_pocket(self, idx: int, num: int = 1) -> None:
        """
        Params
        idx: index of the pocket to manipulate
        num: number of stones to fill in
        """
        assert self.hand > 0 and num <= self.hand
        # print(f"[DEBUG] Fill {num} into idx:{idx} pocket")
        self.board[idx] += num
        self.hand -= num

    def next_idx(self, idx: int) -> int:
        """
        Params
        idx :int: index to check

        Returns
        :int: the next index
        """
        next_idx = (idx + 1) % ((self.rule.pockets + 1) * 2)
        return next_idx

    def opposite_idx(self, idx: int) -> int:
        """
        Params
        idx :int: index to check

        Returns
        :int: the opposide field index
        """
        assert idx <= self.rule.pockets * 2
        return self.rule.pockets * 2 - idx

    @property
    def _player0_field_range(self):
        return range(0, self.rule.pockets)

    @property
    def _player1_field_range(self):
        return range(self.rule.pockets + 1, self.rule.pockets * 2 + 1)

    @property
    def _player0_point_index(self) -> int:
        return self.rule.pockets

    @property
    def _player1_point_index(self) -> int:
        return self.rule.pockets * 2 + 1

    @property
    def _active_player_point_index(self) -> int:
        return (
            self._player0_point_index if self.turn == 0 else self._player1_point_index
        )

    def is_current_sided_pointpocket(self, idx: int) -> bool:
        if self.turn == 0:
            return idx == self.rule.pockets
        else:
            return idx == self.rule.pockets * 2 + 1

    def is_current_sided_fieldpocket(self, idx: int) -> bool:
        if self.turn == 0:
            return 0 <= idx < self.rule.pockets
        else:
            return self.rule.pockets + 1 <= idx < self.rule.pockets * 2 + 1

    @property
    def sided_all_actions(self) -> List[int]:
        if self.turn == 0:
            return list(self._player0_field_range)
        else:
            return list(self._player1_field_range)

    def filter_available_actions(self, actions: List[int]) -> List[int]:
        return [i for i in actions if self.board[i] > 0]

    @property
    def sided_available_actions(self) -> List[int]:
        return self.filter_available_actions(self.sided_all_actions)

    @property
    def _winner(self) -> Union[int, None]:
        winner: Union[int, None] = None
        p0_actions = self.filter_available_actions(list(self._player0_field_range))
        p1_actions = self.filter_available_actions(list(self._player1_field_range))
        p0_points = self.board[self._player0_point_index]
        p1_points = self.board[self._player1_point_index]
        if len(p0_actions) == 0:
            p1_points += sum([self.board[i] for i in p1_actions])
        if len(p1_actions) == 0:
            p0_points += sum([self.board[i] for i in p0_actions])

        if p0_points > self.rule.stones_half:
            winner = 0
        elif p1_points > self.rule.stones_half:
            winner = 1
        elif len(p0_actions) == 0 or len(p1_actions) == 0:
            winner = 1 * (p1_points > p0_points)
        return winner

    @property
    def _done(self) -> bool:
        return self._winner is not None

    def proceed_action(self, idx: int) -> None:
        self.take_pocket(idx)
        continue_turn = False
        for _ in range(self.hand):
            idx = self.next_idx(idx)
            if (
                self.hand == 1
                and self.rule.continue_on_point
                and self.is_current_sided_pointpocket(idx)
            ):
                continue_turn = True
            if (
                self.hand == 1
                and self.rule.capture_opposite
                and self.is_current_sided_fieldpocket(idx)
                and self.board[idx] == 0
                and self.board[self.opposite_idx(idx)] > 0
            ):
                self.take_pocket(self.opposite_idx(idx))
                self.fill_pocket(self._active_player_point_index, self.hand)
                break
            self.fill_pocket(idx)
        if not (continue_turn and self.rule.multi_lap):
            self.flip_turn()

    def flip_turn(self):
        self.turn = 1 if self.turn == 0 else 0


class MancalaEnv(Env):
    metadata = {"render.modes": ["human"]}

    # Core Env functions
    # ------------------
    def __init__(self, agent_types: List[str]):
        super().__init__()
        self.rule = Rule()
        self.state = MancalaState()
        self.possible_agents = ["player0", "player1"]
        self.agents = self.init_agents(agent_types)

        # self.agents_dict = MancalaEnv.init_agents(agent_modes, agent_names=self.agents)
        # WIP
        # In respect to OpenSpiel API
        # self.agents = ["player0", "player1"]
        self.action_spaces = {
            i: spaces.Discrete(self.rule.pockets) for i in self.agents
        }
        self.observation_space = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=2 * self.rule.stones_half,
                        shape=(2, self.rule.pockets),
                        dtype=np.float16,
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=2 * self.rule.stones_half,
                        shape=(2 * self.rule.pockets,),
                        dtype=np.float16,
                    ),
                }
            )
            for i in self.agents
        }
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {
            i: {"legal_moves": list(range(0, self.rule.pockets))} for i in self.agents
        }
        # agent_selection

    def init_agents(self, agent_types):
        assert len(agent_types) == len(self.possible_agents)
        return [init_agent(atype, i) for i, atype in enumerate(agent_types)]

    @property
    def current_agent(self) -> BaseAgent:
        return self.agents[self.state.current_player]

    # @staticmethod
    # def init_agents(
    #     agent_modes: List[str], agent_names: List[str]
    # ) -> Dict[str, BaseAgent]:
    #     pass

    def reset(self) -> MancalaState:
        """
        Env core function
        """
        self.state = MancalaState()
        return self.state

    def step(self, action: int) -> Tuple[MancalaState, int, bool]:
        """
        Env core function
        """
        cloned_state = self.state.clone()
        cloned_state.proceed_action(action)
        reward = cloned_state.get_reward()
        done = cloned_state._done
        return (cloned_state, reward, done)

    def render(self, mode: str = "human") -> None:
        """
        Env core function
        ---
        Currently only supports CLI render mode.
        """
        print("\n" + "====" * (self.rule.pockets + 1))
        # AI side
        print(f"[{self.state.board[self.state._player1_point_index]:>2}]", end=" ")
        for i in self.state._player1_field_range[::-1]:
            print(f"{self.state.board[i]:>2}", end=" ")
        print("\n" + "----" * (self.rule.pockets + 1))
        # Player side
        print(" " * 4, end=" ")
        for i in self.state._player0_field_range:
            print(f"{self.state.board[i]:>2}", end=" ")
        print(f"[{self.state.board[self.state._player0_point_index]:>2}]", end=" ")
        print("\n" + "====" * (self.rule.pockets + 1))

    def close(self) -> None:
        """
        Env core function
        """
        pass

    # Common Env functions
    # --------------------
    @property
    def actions(self) -> List[int]:
        return list(range(self.rule.pockets))

    def transist(
        self, state: MancalaState, action: int
    ) -> Tuple[MancalaState, int, bool]:
        """
        Returns
        next_state:
        reward:
        done:
        """
        pass

    @staticmethod
    def transistion_func(state: BaseState, action: int) -> None:
        """
        Params
        state:
        action:

        Returns
        transition_probs: List[flaot]
        """
        pass
