from mancala.agents import init_agent
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from mancala.agents.a3c.model import ActorCritic
from mancala.mancala import MancalaEnv


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, dtype):
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    agent0 = init_agent("a3c", 0)
    agent1 = init_agent("random", 1)
    env = MancalaEnv(agent0, agent1)
    env.seed(args.seed + rank)
    state = env.reset()

    # model = ActorCritic(state.board.shape[0], env.action_space).type(dtype)
    model = agent0._model

    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    values = []
    log_probs = []

    state = torch.from_numpy(state.board).type(dtype)
    done = True

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 400).type(dtype))
            hx = Variable(torch.zeros(1, 400).type(dtype))
        else:
            cx = Variable(cx.data.type(dtype))
            hx = Variable(hx.data.type(dtype))

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim=0)
            log_prob = F.log_softmax(logit, dim=0)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done = env.step(action.cpu().numpy()[0][0])
            done = done or episode_length >= args.max_episode_length

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state.board).type(dtype)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1).type(dtype)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1).type(dtype)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = (
                policy_loss - log_probs[i] * Variable(gae) - args.beta * entropies[i]
            )

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()