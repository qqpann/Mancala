from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from gym.utils import seeding
from mancala.agents.a3c.model import ActorCritic
from mancala.agents.base import BaseAgent
from mancala.rule import Rule
from mancala.state.base import BaseState
from torch.autograd import Variable


class A3CAgent(BaseAgent):
    """Agent which leverages Actor Critic Learning"""

    def __init__(
        self,
        id: int,
        model_path: Union[str, None] = "",
        model: Union[ActorCritic, None] = None,
    ):
        self.deterministic = False
        self._seed = 42
        self.id = id

        self.np_random, _ = seeding.np_random(self._seed)
        if torch.cuda.is_available():
            self._dtype = torch.cuda.FloatTensor
        else:
            self._dtype = torch.FloatTensor

        rule = Rule()

        def init_board(rule: Rule) -> np.ndarray:
            board = np.zeros(((rule.pockets + 1) * 2,), dtype=np.int32)
            # Player 1 side
            for i in range(0, rule.pockets):
                board[i] = rule.initial_stones
            # Player 2 side
            for i in range(rule.pockets + 1, rule.pockets * 2 + 1):
                board[i] = rule.initial_stones
            return board

        board = init_board(rule)
        action_space = spaces.Discrete(6)
        if model is None:
            self._model: ActorCritic = ActorCritic(board.shape[0], action_space).type(
                self._dtype
            )
            if model_path is None:
                pass
            elif model_path != "":
                self._model.load_state_dict(torch.load(model_path))
            else:
                outputs_dir = Path("outputs")
                best = [str(p) for p in outputs_dir.glob("*_best_*")]
                best = sorted(best, key=lambda x: x.split("_")[-3])
                # best = sorted(best, key=lambda x: x.split("_")[-1])
                try:
                    self._model.load_state_dict(torch.load(str(best[-1])))
                except Exception as e:
                    print(best)
                    raise e
        else:
            self._model = model

    def policy(self, state: BaseState) -> Union[int, None]:
        """Return move which ends in score hole"""
        assert not state._done
        assert self.id == state.current_player
        legal_actions = state.legal_actions(state.current_player)
        if legal_actions is None:
            return None
        turn_offset = state.turn * (state.rule.pockets + 1)

        # board = torch.from_numpy(clone.board).type(self._dtype)
        board = torch.from_numpy(state.perspective_boards[state.turn]).type(self._dtype)
        cx = Variable(torch.zeros(1, 400).type(self._dtype))
        hx = Variable(torch.zeros(1, 400).type(self._dtype))

        with torch.no_grad():
            _, logit, (hx, cx) = self._model((Variable(board.unsqueeze(0)), (hx, cx)))
        prob = F.softmax(logit, dim=1)
        # action = prob.multinomial(num_samples=1).data
        # final_move = action.cpu().numpy()[0][0] + turn_offset
        scores = [
            (action + turn_offset, score)
            for action, score in enumerate(prob[0].data.tolist())
            if action + turn_offset in legal_actions
        ]

        valid_actions = [action for action, _ in scores]
        valid_scores = np.array([score for _, score in scores])

        final_move = self.np_random.choice(
            valid_actions, 1, p=valid_scores / valid_scores.sum()
        )[0]
        # final_move = valid_actions[int(np.argmax(valid_scores))]
        return final_move
