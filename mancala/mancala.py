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
from mancala.rule import Rule


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
        self.must_skip = False

        self.hand = 0

    def __repr__(self):
        return f"<MancalaState: [{self.board}, {self.turn}]>"

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

    def legal_actions(self, turn: int) -> Union[None, List[int]]:
        if self.must_skip:
            return None
        if turn == 0:
            all_actions = list(self._player0_field_range)
        else:
            all_actions = list(self._player1_field_range)
        return self.filter_available_actions(all_actions)

    @property
    def current_player(self) -> int:
        return self.turn

    def clone(self) -> MancalaState:
        return MancalaState(board=self.board.copy(), turn=self.turn)

    def get_reward(self, turn: Union[int, None] = None) -> int:
        point = self.board[self._active_player_point_index]
        return point

    @property
    def rewards(self) -> List[float]:
        r0 = self.board[self._player0_point_index]
        r1 = self.board[self._player1_point_index]
        return [r0, r1]

    def rewards_float(self, receiver_player_id) -> float:
        if self._done and self._winner == receiver_player_id:
            return 1
        elif self._done:
            return -1
        else:
            if receiver_player_id == 0:
                return 0.01 * (self.rewards[0] - self.rewards[1])
            else:
                return 0.01 * (self.rewards[1] - self.rewards[0])

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
        assert 0 <= idx <= self.rule.pockets * 2
        return self.rule.pockets * 2 - idx

    def _can_continue_on_point(self, idx) -> bool:
        return self._active_player_point_index - idx == self.board[idx]

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

    def is_opponent_sided_pointpocket(self, idx: int) -> bool:
        if self.turn != 0:
            return idx == self.rule.pockets
        else:
            return idx == self.rule.pockets * 2 + 1

    def filter_available_actions(self, actions: List[int]) -> List[int]:
        return [i for i in actions if self.board[i] > 0]

    @property
    def _winner(self) -> Union[int, None]:
        winner: Union[int, None] = None
        p0_all_actions = self.filter_available_actions(list(self._player0_field_range))
        p1_all_actions = self.filter_available_actions(list(self._player1_field_range))
        p0_points = self.board[self._player0_point_index]
        p1_points = self.board[self._player1_point_index]
        if len(p0_all_actions) == 0:
            p1_points += sum([self.board[i] for i in p1_all_actions])
        if len(p1_all_actions) == 0:
            p0_points += sum([self.board[i] for i in p0_all_actions])

        if p0_points > self.rule.stones_half:
            winner = 0
        elif p1_points > self.rule.stones_half:
            winner = 1
        elif len(p0_all_actions) == 0 or len(p1_all_actions) == 0:
            winner = 1 * (p1_points > p0_points)
        return winner

    @property
    def _done(self) -> bool:
        return self._winner is not None

    def is_terminal(self) -> bool:
        return self._done

    def proceed_action(self, act: Union[int, None]) -> MancalaState:
        if act is None:
            self.flip_turn(False)
            return self
        self.take_pocket(act)
        skip_opponent = False
        for _ in range(self.hand):
            act = self.next_idx(act)
            # Do not fill into opponent's point pocket
            if self.is_opponent_sided_pointpocket(act):
                act = self.next_idx(act)
            # Skip rule
            if (
                self.hand == 1
                and self.rule.continue_on_point
                and self.is_current_sided_pointpocket(act)
            ):
                skip_opponent = True
            # Capture rule
            if (
                self.hand == 1
                and self.rule.capture_opposite
                and self.is_current_sided_fieldpocket(act)
                and self.board[act] == 0
                and self.board[self.opposite_idx(act)] > 0
            ):
                self.take_pocket(self.opposite_idx(act))
                self.fill_pocket(self._active_player_point_index, self.hand)
                break
            self.fill_pocket(act)
        self.flip_turn(skip_opponent and self.rule.multi_lap)
        return self

    def flip_turn(self, skip_opponent: bool) -> None:
        self.turn = 1 if self.turn == 0 else 0
        self.must_skip = skip_opponent


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

    def step(self, action: Union[int, None]) -> Tuple[MancalaState, int, bool]:
        """
        Env core function
        """
        clone = self.state.clone()
        clone.proceed_action(action)
        reward = clone.get_reward()
        done = clone._done
        return (clone, reward, done)

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
