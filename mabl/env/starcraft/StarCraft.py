from smac.env import StarCraft2Env


class StarCraft:

    def __init__(self, env_name):
        self.env = StarCraft2Env(map_name=env_name, continuing_episode=True, difficulty="7")
        env_info = self.env.get_env_info()

        self.n_obs = env_info["obs_shape"]
        self.n_state = env_info["state_shape"]
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]

    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}

    def step(self, action_dict):
        reward, done, info = self.env.step(action_dict)
        return self.to_dict(self.env.get_obs()), self.env.get_state(), {i: reward for i in range(self.n_agents)}, \
               {i: done for i in range(self.n_agents)}, info

    def reset(self):
        self.env.reset()
        return {i: obs for i, obs in enumerate(self.env.get_obs())}

    def get_state(self):
        return self.env.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_avail_agent_actions(self, handle):
        return self.env.get_avail_agent_actions(handle)
