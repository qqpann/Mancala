from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from gym.utils import seeding
from torch.autograd import Variable

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState
from mancala.agents.a3c.model import ActorCritic


class AgentA3C(BaseAgent):
    """Agent which leverages Actor Critic Learning"""

    def __init__(
        self, id: int, depth: int, state, action_space, model_path: str, dtype
    ):
        self.deterministic = False
        self._seed = 42
        self._depth = depth
        self.id = id

        self.np_random, _ = seeding.np_random(self._seed)
        self._dtype = dtype

        self._model = ActorCritic(state.board.shape[0], action_space).type(dtype)
        self._model.load_state_dict(torch.load(model_path))

    def policy(self, state: BaseState) -> Union[int, None]:
        """Return move which ends in score hole"""
        assert not state.is_terminal()
        assert self.id == state.current_player
        clone = state.clone()
        move_options = state.legal_actions(state.current_player)
        assert move_options is not None

        board = torch.from_numpy(clone.board).type(self._dtype)
        cx = Variable(torch.zeros(1, 400).type(self._dtype), volatile=True)
        hx = Variable(torch.zeros(1, 400).type(self._dtype), volatile=True)

        _, logit, (hx, cx) = self._model(
            (Variable(board.unsqueeze(0), volatile=True), (hx, cx))
        )
        prob = F.softmax(logit)
        scores = [
            (action, score)
            for action, score in enumerate(prob[0].data.tolist())
            if action in move_options
        ]

        valid_actions = [action for action, _ in scores]
        valid_scores = np.array([score for _, score in scores])

        final_move = self.np_random.choice(
            valid_actions, 1, p=valid_scores / valid_scores.sum()
        )[0]
        return final_move
