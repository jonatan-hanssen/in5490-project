import json
import gymnasium as gym


def make_env(env_name, seed):
    """
    Descr:
        Returns a "vectorized" environenment, meaning that it is wrapped in a gymnasium vector, allowing for
        some aditional functionality, such as parallel training of many environments.
    """

    def env_gen():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_gen


def read_params(params):
    file = open(params)
    params = json.load(file)
    # print(json.dumps(params, indent=4, separators=(":", ",")))
    return params


def save_params(params):
    file = open("hyperparams.json", "w")
    json.dump(params, file, indent=4, separators=(",", ":"))


def minibatch_generator(params, *args):
    b_obs = observations.view((-1, single_obs_shape))
    b_logs = logprobs.view(-1)
    b_acts = ations.view((-1) + envs.single_action_space.shape)
    b_advs = advantages.view(-1)
    b_vals = values.view(-1)
    b_rets = returns.view(-1)

    batch_idxs = np.random.choice(
        params["batch_size"], params["batch_size"], replace=False
    )
    for start in range(0, params["batch_size"], params["minibatch_size"]):
        end = start + params["minibatch_size"]
        minibatch_idxs = batch_idxs[start:end]

        yield (
            b_obs[minibatch_idxs],
            b_logs[minibatch_idxs],
            b_acts[minibatch_idxs],
            b_advs[minibatch_idxs],
            b_vals[minibatch_idxs],
            b_rets[minibatch_idxs],
        )
