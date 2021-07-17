import datetime
from mancala.agents import init_agent, init_random_agent
from mancala.agents.mixed import MixedAgent
import time
from collections import deque
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.utils import seeding
from torch.autograd import Variable

from mancala.agents.a3c.agent import A3CAgent
from mancala.arena import play_games
from mancala.mancala import MancalaEnv

# from tensorboard_logger import configure, log_value


EVALUATION_EPISODES = 100
PERFORMANCE_GAMES = 200


def test(rank, args, shared_model, dtype):
    test_ctr = 0
    torch.manual_seed(args.seed + rank)

    # set up logger
    timestring = (
        str(date.today())
        + "_"
        + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    )
    run_name = args.save_name + "_" + timestring
    # configure("logs/run_" + run_name, flush_secs=5)

    training_agent_id = 0
    agent0 = A3CAgent(0, model=shared_model)
    # agent1 = MixedAgent(1)
    agent1 = init_random_agent(1, ["max", "negascout"], [0.1, 0.9])
    env = MancalaEnv(agent0, agent1)
    env.seed(args.seed + rank)
    np_random, _ = seeding.np_random(args.seed + rank)
    state = env.reset()

    # model = ActorCritic(state.board.shape[0], env.action_space).type(dtype)
    model = agent0._model

    model.eval()

    state_vec = torch.from_numpy(state.board).type(dtype)
    reward_sum = 0
    max_reward = -99999999
    max_winrate = 0
    rewards_recent = deque([], 100)
    done = True

    start_time = time.time()
    last_test = time.time()

    episode_length = 0
    while True:
        episode_length += 1
        if done and training_agent_id != 0:
            env.flip_p0p1()
            training_agent_id = 1 - training_agent_id
        if done:
            env.agent1 = init_random_agent(1, ["max", "negascout"], [0.1, 0.9])
        if done and np.random.random() > 0.5:
            env.flip_p0p1()
            training_agent_id = 1 - training_agent_id
            if env.current_agent.id != training_agent_id:
                env.step(
                    env.current_agent.policy(env.state),
                    inplace=True,
                    until_next_turn=True,
                )
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 400).type(dtype))
            hx = Variable(torch.zeros(1, 400).type(dtype))
        else:
            cx = Variable(cx.data.type(dtype))
            hx = Variable(hx.data.type(dtype))

        with torch.no_grad():
            value, logit, (hx, cx) = model((Variable(state_vec.unsqueeze(0)), (hx, cx)))
        prob = F.softmax(logit, dim=1)
        # log_prob = F.log_softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()

        # scores = [(action, score) for action, score in enumerate(prob[0].data.tolist())]
        legal_actions = state.legal_actions(state.current_player)
        turn_offset = env.state.turn * (env.rule.pockets + 1)
        scores = [
            (action + turn_offset, score)
            for action, score in enumerate(prob[0].data.tolist())
            if action + turn_offset in legal_actions
        ]

        valid_actions = [action for action, _ in scores]
        valid_scores = np.array([score for _, score in scores])

        final_move = np_random.choice(
            valid_actions, 1, p=valid_scores / valid_scores.sum()
        )[0]

        # assert not env.state.must_skip, env.render()
        # act = action.cpu().numpy()[0][0] + turn_offset

        state, reward, done = env.step(
            final_move, inplace=True, until_next_turn=True, illegal_penalty=True
        )
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            rewards_recent.append(reward_sum)
            rewards_recent_avg = sum(rewards_recent) / len(rewards_recent)
            print(
                "{} | {} | Episode Reward {: >4}, Length {: >2} | Avg Reward {:0.2f}".format(
                    datetime.datetime.now().isoformat(),
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    round(reward_sum, 2),
                    episode_length,
                    round(rewards_recent_avg, 2),
                )
            )

            # if not stuck or args.evaluate:
            # log_value("Reward", reward_sum, test_ctr)
            # log_value("Reward Average", rewards_recent_avg, test_ctr)
            # log_value("Episode length", episode_length, test_ctr)

            if (
                reward_sum >= max_reward
                or time.time() - last_test > 60 * 8
                or (
                    len(rewards_recent) > 12
                    and time.time() - last_test > 60 * 2
                    and sum(list(rewards_recent)[-5:])
                    > sum(list(rewards_recent)[-10:-5])
                )
            ):

                # if the reward is better or every 15 minutes
                last_test = time.time()
                max_reward = reward_sum if reward_sum > max_reward else max_reward

                path_output = args.save_name + "_max"
                torch.save(shared_model.state_dict(), path_output)
                path_now = "{}_{}".format(
                    args.save_name, datetime.datetime.now().isoformat()
                )
                torch.save(shared_model.state_dict(), path_now)

                agent0 = A3CAgent(0, model=shared_model)
                agent1 = A3CAgent(1, model=shared_model)
                win_rate_v_random, _ = play_games(
                    agent0, init_agent("random", 1), PERFORMANCE_GAMES
                )
                win_rate_v_exact, _ = play_games(
                    agent0, init_agent("exact", 1), PERFORMANCE_GAMES
                )
                _, win_rate_exact_v = play_games(
                    init_agent("exact", 0), agent1, PERFORMANCE_GAMES
                )
                win_rate_v_minmax, _ = play_games(
                    agent0, init_agent("minimax", 1), PERFORMANCE_GAMES
                )
                _, win_rate_minmax_v = play_games(
                    init_agent("minimax", 1), agent1, PERFORMANCE_GAMES
                )

                msg = "{t} | Random: {r0:.1f}% | Exact: {e0:.1f}%/{e1:.1f}% | MinMax: {m0:.1f}%/{m1:.1f}%".format(
                    t=datetime.datetime.now().strftime("%c"),
                    r0=win_rate_v_random,
                    e0=win_rate_v_exact,
                    # e1=0,
                    e1=win_rate_exact_v,
                    m0=win_rate_v_minmax,
                    # m1=0,
                    m1=win_rate_minmax_v,
                )
                # msg = f"Win rate vs random: {win_rate_v_random}"
                print(msg)
                # log_value("WinRate_Random", win_rate_v_random, test_ctr)
                # log_value("WinRate_Exact", win_rate_v_exact, test_ctr)
                # log_value("WinRate_MinMax", win_rate_v_minmax, test_ctr)
                # log_value("WinRate_ExactP2", win_rate_exact_v, test_ctr)
                # log_value("WinRate_MinMaxP2", win_rate_minmax_v, test_ctr)
                avg_win_rate = (
                    win_rate_v_exact
                    + win_rate_v_minmax
                    # + win_rate_exact_v
                    # + win_rate_minmax_v
                ) / 4
                # avg_win_rate = win_rate_v_random
                if avg_win_rate > max_winrate:
                    print(
                        "Found superior model at {}".format(
                            datetime.datetime.now().isoformat()
                        )
                    )
                    torch.save(
                        shared_model.state_dict(),
                        "{}_{}_best_{}".format(
                            args.save_name,
                            datetime.datetime.now().isoformat(),
                            avg_win_rate,
                        ),
                    )
                    max_winrate = avg_win_rate

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            test_ctr += 1

            if test_ctr % 10 == 0 and not args.evaluate:
                torch.save(shared_model.state_dict(), args.save_name)
            if not args.evaluate:
                time.sleep(60)
            elif test_ctr == EVALUATION_EPISODES:
                # Ensure the environment is closed so we can complete the submission
                env.close()
                # gym.upload('monitor/' + run_name, api_key=api_key)

        state_vec = torch.from_numpy(state.board).type(dtype)