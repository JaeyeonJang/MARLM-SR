from gym import spaces
import numpy as np
import multiagent.scenarios as scenarios
from ray.rllib.env.multi_agent_env import MultiAgentEnv

def preprocess(data, min, max):
    return (data - min) / (max - min)

class MultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        self.GP_len = config['GP_len']
        self.GP_num = config['GP_num']
        self.hidden_len = config['hidden_len']
        self.coordinator_len = config['coordinator_len']
        self.current_step = 0
        self.current_GP = 0
        self.epi = 0
        scenario = scenarios.load(config['scenario_name']).Scenario()
        world = scenario.make_world()
        self.world = world
        self.agents = self.world.agents
        # scenario callbacks
        self.reset_callback = scenario.reset_world
        # self.reward_callback = scenario.reward
        self.observation_callback = scenario.observation
        self.total_hidden_len = self.hidden_len * len(self.agents)
        self.time = 0
        # configure spaces
        self.action_space = []
        self.previous_reward = self.GP_num
        self.multiplier = config['multiplier']

        for agent in self.agents:
            if agent.level == 0:
                self.action_space.append(spaces.Discrete(5))
            else:
                self.action_space.append(spaces.Discrete(4))

        # observation space
        obs_upper = np.array([self.GP_len * self.GP_num] * 13 + [4] * 3 + [9] * 3 + [1] * self.hidden_len)  # for followers
        obs_lower = np.array([0] * 13 + [0] * 3 + [0] * 3 + [0] * self.hidden_len)   # for followers
        self.observation_space = spaces.Box(low=obs_lower, high= obs_upper, dtype=np.float32)


        obs_upper = np.array([self.GP_len * self.GP_num] * 13 + [4] * 3 + [10] * 3)  # for followers
        obs_lower = np.array([0] * 13 + [0] * 3 + [0] * 3)  # for followers
        self.leader_observation_space = spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32)
        self.leader_action_space = spaces.Box(0, 1, (self.total_hidden_len,), dtype = np.float32)

        self.set_graph()

        self.estimator_obs_upper = np.array([self.GP_len * self.GP_num] * 13 + [4] * 3 + [9] * 3) # for followers
        self.estimator_obs_lower = np.array([0] * 13 + [0] * 3 + [0] * 3)  # for followers

        self.coordinator_observation_space = spaces.Box(low=np.append(np.concatenate([self.estimator_obs_lower] * (self.coordinator_len+1)), [0] * self.total_hidden_len),
                                                      high= np.append(np.concatenate([self.estimator_obs_upper] * (self.coordinator_len+1)), [1] * self.total_hidden_len)
                                                        , dtype=np.float32)
        self.coordinator_action_space = spaces.Box(np.array([0]*len(self.begin_point), dtype=np.float32),
                                             np.array([1]*len(self.begin_point), dtype=np.float32))

        self.coordinator2_observation_space = spaces.Box(low=np.append(np.concatenate([self.estimator_obs_lower] * (self.coordinator_len+1)), [0] * self.total_hidden_len),
                                                      high= np.append(np.concatenate([self.estimator_obs_upper] * (self.coordinator_len+1)), [1] * self.total_hidden_len)
                                                        , dtype=np.float32)
        self.coordinator2_action_space = spaces.Box(low=np.array([0] * 2, dtype=np.float32), high= np.array([1] * 2, dtype=np.float32))



    def set_graph(self):
        self.agent_paths = [(0, 1), (0, 2), (1, 3), (2, 3)]
        self.final_idx = [3]
        self.init_idx = [0]
        idx = 0
        self.edges = dict()
        for path in self.agent_paths:
            if path[1] in self.edges:
                self.edges[path[1]].append(path[0])
            else:
                self.edges[path[1]] = [path[0]]

        self.begin_point = []
        self.end_point = []
        self.begin_point_dict = dict()

        for agent_id in np.arange(len(self.agents)-1,-1,-1):
            if agent_id in self.edges:
                for prev_id in self.edges[agent_id]:
                    self.begin_point.append(agent_id)
                    self.end_point.append(prev_id)
                    if agent_id not in self.begin_point_dict:
                        self.begin_point_dict[agent_id] = []
                    self.begin_point_dict[agent_id].append(idx)
                    idx += 1
                self.begin_point.append(agent_id)
                self.end_point.append(agent_id)
                self.begin_point_dict[agent_id].append(idx)
                idx += 1

    def step(self, action):
        if -1 in action:
            return self.outer_step(action[-1])
        elif -2 in action:
            return self.coordinator_step(action)
        else:
            return self.inner_step(action)

    def outer_step(self, action):
        self.goal = action
        obs = dict()
        reward = dict()
        share_obs = self._get_obs()
        for i, agent in enumerate(self.agents):
            obs[i] = np.append(share_obs,self.goal[self.hidden_len*i:self.hidden_len*(i+1)])
            reward[i] = self.inner_step_reward[i]
        self.coordinator_input = [share_obs]

        info = dict() #{'n': []}
        done = {"__all__": False}
        return obs, reward, done, info

    def get_estimator_input(self, action):
        one_hot_action = self._set_one_hot_action(action)
        obs = np.append(self._get_obs(),self.goal)
        for i in range(4):
            obs = np.append(obs, one_hot_action[i])
        inp = preprocess(np.array([obs]), self.predictor_lower, self.predictor_upper)
        return inp

    def inner_step(self, action):
        obs = dict()
        reward = dict()
        info = dict() #{'n': []}
        # set action for each agent
        action_out = []
        for i, agent in enumerate(self.agents):
            action_out.append(self._get_action(action[i], agent))
        # advance world state
        for i, agent in enumerate(self.agents):
            self.update_inventory(action_out[i], agent, i)
        # record observation for each agent
        self.current_step += 1
        share_obs = self._get_obs()

        if (self.current_step - 1) % (self.GP_len/self.coordinator_len) == 0 and self.current_step % self.GP_len > 1:
            self.coordinator_input.append(share_obs)

        if self.current_step % self.GP_len == 0:
            # self.current_step = 0
            self.current_GP += 1
            self.coordinator_input.append(share_obs)
            self.coordinator_input = np.append(np.concatenate(self.coordinator_input), self.goal)
            self.inner_step_reward = dict()
            self.final_rewards = dict()
            for i, agent in enumerate(self.agents):
                # reward[i] += sparse_reward[i]
                self.inner_step_reward[i] = 0
                if i in self.final_idx:
                    self.final_rewards[i] = 0

            reward[-2] = 0
            obs[-2] = self.coordinator_input
            reward[-3] = 0
            obs[-3] = self.coordinator_input
        else:
            for i, agent in enumerate(self.agents):
                reward[i] = 0
                obs[i] = np.append(share_obs, self.goal[self.hidden_len * i:self.hidden_len * (i + 1)])

        if self.current_step % 40 == 0:
            current_total_reward = 0
            sparse_reward = self._get_sparse_reward()
            for i, agent in enumerate(self.agents):
                # reward[i] += sparse_reward[i]
                self.total_reward += sparse_reward[i]
                current_total_reward += sparse_reward[i]
                if i in self.final_idx:
                    self.total_final_rewards[i] += sparse_reward[i]
            self.world.product_value.reset_value()
            self.world.demand_value.reset_value()
            self.world.inventory.level2[:] = 0

        done_n = {"__all__": False}
        return obs, reward, done_n, info

    def _set_one_hot_action(self,action):
        one_hot_action = dict()
        one_hot_action[0] = np.zeros(5)
        one_hot_action[0][action[0]] = 1
        one_hot_action[1] = np.zeros(4)
        one_hot_action[1][action[1]] = 1
        one_hot_action[2] = np.zeros(4)
        one_hot_action[2][action[2]] = 1
        one_hot_action[3] = np.zeros(4)
        one_hot_action[3][action[3]] = 1
        return one_hot_action


    def backpropagation(self, organizer_act):
        current_ids = self.final_idx
        current_rwd = [self.final_rewards[i] for i in self.final_idx]
        continue_flag = True

        while continue_flag:
            next_ids = []
            next_rwd = []
            for agent_idx, agent_id in enumerate(current_ids):
                if agent_id not in self.init_idx:
                    probs = organizer_act[np.array(self.begin_point_dict[agent_id])]
                    if sum(probs) == 0:
                        probs[:] = 1
                    probs = probs/probs.sum()

                    for idx_, idx in enumerate(self.begin_point_dict[agent_id]):
                        next_id = self.end_point[idx]
                        if next_id == agent_id:
                            self.inner_step_reward[agent_id] += current_rwd[agent_idx] * probs[idx_]
                        else:
                            next_ids.append(next_id)
                            next_rwd.append(current_rwd[agent_idx] * probs[idx_])
                else:
                    self.inner_step_reward[agent_id] += current_rwd[agent_idx]

            if len(next_ids) == 0:
                continue_flag = False
            else:
                current_ids = next_ids
                current_rwd = next_rwd

    def coordinator_step(self, action_set):
        given_reward = action_set[-3][0] * self.previous_reward/self.GP_num * (self.multiplier * action_set[-3][1])
        action = action_set[-2]

        obs = dict()
        reward = dict()
        for i, agent in enumerate(self.agents):
            if i in self.final_idx:
                self.final_rewards[i] = given_reward

        self.backpropagation(action)
        # total_reward = 0

        share_obs = self._get_obs()
        reward[-1] = 0
        obs[-1] = share_obs

        if self.current_GP >= self.GP_num:
            done = {"__all__": True}
            # self.error_writer.add_scalar('error', np.mean(self.errors), self.epi)
            # self.error_writer.add_scalar('error2', np.mean(self.errors2), self.epi)
            self.errors = []
            self.errors2 = []
            reward[-2] = self.total_reward
            obs[-2] = self.coordinator_input
            reward[-3] = self.total_reward
            obs[-3] = self.coordinator_input
            reward[-1] = self.total_reward
            self.previous_reward = abs(self.total_reward)
            for i, agent in enumerate(self.agents):
                reward[i] = self.inner_step_reward[i] + self.total_reward/len(self.agents)
                obs[i] = np.append(share_obs,self.goal[self.hidden_len * i:self.hidden_len * (i + 1)])
        else:
            done = {"__all__": False}

        info = {}
        return obs, reward, done, info


    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs = dict()
        share_obs = self._get_obs()
        # for i, agent in enumerate(self.agents):
        obs[-1] = share_obs
        self.current_step = 0
        self.current_GP = 0
        self.inner_reward = dict()
        self.inner_step_reward = dict()
        for i, agent in enumerate(self.agents):
            self.inner_reward[i] = 0
            self.inner_step_reward[i] = 0
        self.total_final_rewards = dict()
        for i in self.final_idx:
            self.total_final_rewards[i] = 0
        self.total_reward = 0
        self.epi += 1
        return obs


    # get observation for a particular agent
    def _get_obs(self):
        if self.observation_callback is None:
            return np.zeros(0)
        obs = self.observation_callback(self.world)
        return obs


    # get reward for a particular agent
    def _get_immediate_reward(self, action_out_list):
        reward = dict()
        reward[0] = 0
        reward[1] = 0
        reward[2] = 0
        reward[3] = 0 #there is no immediate reward for the last agent

        return reward

    def _get_sparse_reward(self):
        reward = dict()
        reward[0] = 0
        reward[1] = 0
        reward[2] = 0
        reward[3] = 0

        reward[3] -= self.world.inventory.level0_cost * sum(self.world.inventory.level0)
        reward[3] -= self.world.inventory.level1_cost * sum(self.world.inventory.level1[:3])
        reward[3] -= self.world.inventory.level1_cost * sum(self.world.inventory.level1[3:])
        reward[3] += sum(self.world.product_value.values * self.world.inventory.level2)
        reward[3] -= sum(np.maximum(self.world.inventory.level2 - self.world.demand_value.values, 0) * (self.world.product_value.values - 1))

        return reward

    # set env action for a particular agent
    def _get_action(self, action, agent):
        if agent.level == 0:
            action_out = np.zeros(len(agent.outputs))
            if action < len(action_out):
                action_out[action] = 1
            return action_out

        elif agent.level == 1:
            action_out = np.zeros(len(agent.outputs))
            if action < len(action_out):
                if sum(self.world.inventory.level0 >= agent.parts[action]) == len(self.world.inventory.level0):
                    self.world.inventory.level0 -= agent.parts[action]
                    action_out[action] = 1
            return action_out

        elif agent.level == 2:
            action_out = np.zeros(len(agent.outputs))
            if action < len(action_out):
                if sum(self.world.inventory.level1 >= agent.parts[action]) == len(self.world.inventory.level1):
                    self.world.inventory.level1 -= agent.parts[action]
                    action_out[action] = 1
            return action_out

    def update_inventory(self, action_out, agent, idx):
        if agent.level == 0:
            # print(self.world.inventory.level0, action_out)
            self.world.inventory.level0 += action_out
        elif agent.level == 1:
            if idx == 1:
                self.world.inventory.level1[:3] += action_out
            elif idx == 2:
                self.world.inventory.level1[3:] += action_out
        elif agent.level == 2:
            self.world.inventory.level2 += action_out
            # self.goal -= action_out
            # self.goal = np.maximum(self.goal, 0)

