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
from mancala.rule import Rule
from mancala.state.base import BaseState

turn_names = ["player0", "player1"]

WINNER_DRAW = -1
WINNER_P0 = 0
WINNER_P1 = 1
WINNER_NOT_OVER: None = None

# REWARD_ONE_POINT = 0.01
REWARD_IMMEDIATE = 0.01
REWARD_ILLEGAL_PENALTY = 0.05
REWARD_MULTIPLIER = 1


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
        res = self.filter_available_actions(all_actions)
        if len(res) == 0:
            return None
        return res

    @property
    def current_player(self) -> int:
        return self.turn

    def clone(self) -> MancalaState:
        return MancalaState(board=self.board.copy(), turn=self.turn)

    @property
    def _winner(self) -> Union[int, None]:
        """
        winner
        0: player0
        1: player1
        -1: draw
        None: game is not over
        """
        game_over = False
        winner: Union[int, None] = WINNER_NOT_OVER
        p0_all_actions = self.filter_available_actions(list(self._player0_field_range))
        p1_all_actions = self.filter_available_actions(list(self._player1_field_range))
        p0_points = self.board[self._player0_point_index]
        p1_points = self.board[self._player1_point_index]
        if len(p0_all_actions) == 0:
            game_over = True
            p1_points += sum([self.board[i] for i in p1_all_actions])
        if len(p1_all_actions) == 0:
            game_over = True
            p0_points += sum([self.board[i] for i in p0_all_actions])
        if p0_points > self.rule.stones_half or p1_points > self.rule.stones_half:
            game_over = True

        if game_over:
            if p1_points > p0_points:
                winner = WINNER_P1
            elif p0_points > p1_points:
                winner = WINNER_P0
            else:
                winner = WINNER_DRAW
        return winner

    @property
    def _done(self) -> bool:
        return self._winner is not WINNER_NOT_OVER

    @property
    def scores(self) -> List[int]:
        r0 = self.board[self._player0_point_index]
        r1 = self.board[self._player1_point_index]
        return [r0, r1]

    def get_reward(self, receiver_player_id: int) -> float:
        if not self._done:
            if receiver_player_id == 0:
                return REWARD_IMMEDIATE * (self.scores[0] - self.scores[1])
            else:
                return REWARD_IMMEDIATE * (self.scores[1] - self.scores[0])
        else:
            if self._winner == receiver_player_id:
                return 1
            elif self._winner == WINNER_DRAW and receiver_player_id == 1:
                return 1
            elif self._winner == WINNER_DRAW and receiver_player_id == 0:
                return -1
            else:
                return -1

    @property
    def rewards(self) -> List[float]:
        return [
            self.get_reward(0) * REWARD_MULTIPLIER,
            self.get_reward(1) * REWARD_MULTIPLIER,
        ]

    def take_pocket(self, idx: int) -> None:
        """
        Params
        idx: index of the pocket to manipulate
        """
        assert self.board[idx] > 0, f"Empty pocket {idx}; turn{self.turn}; {self.board}"
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
    def _player_field_ranges(self) -> List[range]:
        return [self._player0_field_range, self._player1_field_range]

    @property
    def _active_player_field_range(self) -> range:
        return self._player_field_ranges[self.turn]

    @property
    def _player0_point_index(self) -> int:
        return self.rule.pockets

    @property
    def _player1_point_index(self) -> int:
        return self.rule.pockets * 2 + 1

    @property
    def _player_point_indexes(self) -> List[int]:
        return [self._player0_point_index, self._player1_point_index]

    @property
    def _active_player_point_index(self) -> int:
        return self._player_point_indexes[self.turn]

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

    def proceed_action(self, act: Union[int, None]) -> MancalaState:
        if act is None:
            self.flip_turn(skip_opponent=False)
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
        self.flip_turn(skip_opponent)
        return self

    def flip_turn(self, skip_opponent: bool) -> None:
        self.turn = 1 - self.turn
        self.must_skip = skip_opponent

    @property
    def perspective_boards(self) -> List[np.ndarray]:
        reversed_board = np.concatenate(
            (
                self.board[self.rule.pockets + 1 : self.rule.pockets * 2 + 2],
                self.board[0 : self.rule.pockets + 1],
            ),
            axis=0,
        )
        return [self.board, reversed_board]


class MancalaEnv(Env):
    metadata = {"render.modes": ["human"]}

    # Core Env functions
    # ------------------
    def __init__(self, player0: BaseAgent, player1: BaseAgent):
        super().__init__()
        self.rule = Rule()
        self.state = MancalaState()
        self.possible_agents = [str(player0), str(player1)]
        self.agents = [player0, player1]

        # self.agents_dict = MancalaEnv.init_agents(agent_modes, agent_names=self.agents)
        # WIP
        # In respect to OpenSpiel API
        # self.agents = ["player0", "player1"]
        self.action_space = spaces.Discrete(self.rule.pockets)
        self.observation_space = spaces.Dict(
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
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {
            i: {"legal_moves": list(range(0, self.rule.pockets))} for i in self.agents
        }
        # agent_selection

    @property
    def current_agent(self) -> BaseAgent:
        return self.agents[self.state.current_player]

    def flip_p0p1(self) -> None:
        new_p1, new_p0 = self.agents
        new_p0.set_id(0)
        new_p1.set_id(1)
        self.agents = [new_p0, new_p1]

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

    def step(
        self,
        action: Union[int, None],
        inplace: bool = False,
        until_next_turn: bool = False,
        illegal_penalty: bool = False,
    ) -> Tuple[MancalaState, float, bool]:
        """
        Env core function
        """
        # assert self.action_space.contains(action)
        clone = self.state.clone()
        current_turn = clone.turn
        old_reward = clone.rewards[current_turn]
        legal_actions = clone.legal_actions(current_turn)
        if (
            illegal_penalty
            and action is not None
            and legal_actions is not None
            and action not in legal_actions
        ):
            return (
                clone,
                min(
                    -1 - REWARD_ILLEGAL_PENALTY * REWARD_MULTIPLIER,
                    -REWARD_ILLEGAL_PENALTY * REWARD_MULTIPLIER,
                    clone.rewards[current_turn]
                    - REWARD_ILLEGAL_PENALTY * REWARD_MULTIPLIER,
                ),
                True,
            )
        clone.proceed_action(action)
        while (
            until_next_turn
            and not clone._done
            and (clone.turn != current_turn or clone.must_skip)
        ):
            clone.proceed_action(self.agents[clone.turn].policy(clone))
        reward = clone.rewards[current_turn] - old_reward - REWARD_IMMEDIATE * 0.2
        done = clone._done
        if inplace:
            self.state = clone
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
        super().close()

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
