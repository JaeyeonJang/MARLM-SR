from gym import spaces
import numpy as np
import multiagent.scenarios as scenarios
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from copy import copy

def preprocess(data, min, max):
    return (data - min) / (max - min)

class MultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        self.GP_len = config['GP_len']
        self.GP_num = config['GP_num']
        self.coordinator_len = config['coordinator_len']
        self.current_step = 0
        self.current_GP = 0
        scenario = scenarios.load(config['scenario_name']).Scenario()
        world = scenario.make_world(self.GP_len)
        self.world = world
        self.agents = self.world.agents
        # scenario callbacks
        self.reset_callback = scenario.reset_world
        self.change_GP = scenario.change_GP
        # self.reward_callback = scenario.reward
        self.observation_callback = scenario.observation
        self.global_observation_callback = scenario.global_observation
        action_len = 2 * len(self.world.final_idx) #for leader
        self.time = 0
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.final_idx = self.world.final_idx
        self.agent_paths = self.world.agent_paths
        self.hidden_len = config['hidden_len']
        self.multiplier = config['multiplier']


        self.total_hidden_len = len(self.agents) * self.hidden_len
        self.previous_reward = self.GP_num
        for idx, agent in enumerate(self.agents):
            if idx in [0, 1]:
                self.action_space.append(spaces.Discrete(3))
                obs_upper, obs_lower = scenario.observation_bound(world, idx, self.GP_num, self.GP_len)
                obs_upper = np.append(obs_upper,[1]*self.hidden_len)
                obs_lower = np.append(obs_lower,[0]*self.hidden_len)
                self.observation_space.append(spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32))
            else:
                self.action_space.append(spaces.Discrete(5))
                obs_upper, obs_lower = scenario.observation_bound(world, idx, self.GP_num, self.GP_len)
                obs_upper = np.append(obs_upper,[1]*self.hidden_len)
                obs_lower = np.append(obs_lower,[0]*self.hidden_len)
                self.observation_space.append(spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32))

        self.estimator_obs_upper, self.estimator_obs_lower = scenario.global_observation_bound(self.world, self.GP_num, self.GP_len)
        obs_upper, obs_lower = scenario.global_observation_bound(self.world, self.GP_num, self.GP_len)

        self.set_graph()

        self.leader_action_space = spaces.Box(low=np.array([0] * self.total_hidden_len), high=np.array([1] * self.total_hidden_len), dtype=np.float32)
        obs_upper, obs_lower = scenario.global_observation_bound(self.world, self.GP_num, self.GP_len)
        self.leader_observation_space = spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32)

        self.coordinator_observation_space = spaces.Box(low=np.append(np.concatenate([self.estimator_obs_lower] * (self.coordinator_len+1)), [0] * self.total_hidden_len),
                                                      high= np.append(np.concatenate([self.estimator_obs_upper] * (self.coordinator_len+1)), [1] * self.total_hidden_len),
                                                        dtype=np.float32)
        self.coordinator_action_space = spaces.Box(np.array([0]*len(self.begin_point), dtype=np.float32),
                                             np.array([1]*len(self.begin_point), dtype=np.float32))
        obs_upper = np.array([1]*2)  # for observer
        obs_lower = np.array([0]*2)  # for observer
        self.observer_observation_space = spaces.Box(low=obs_lower, high= obs_upper, dtype=np.float32)
        self.observer_action_space = spaces.Discrete(2)

        self.coordinator2_observation_space = spaces.Box(low=np.append(np.concatenate([self.estimator_obs_lower] * (self.coordinator_len+1)), [0] * self.total_hidden_len),
                                                      high= np.append(np.concatenate([self.estimator_obs_upper] * (self.coordinator_len+1)), [1] * self.total_hidden_len),
                                                        dtype=np.float32)
        self.coordinator2_action_space = spaces.Box(low=np.array([0]*4, dtype=np.float32), high= np.array([1]*4, dtype=np.float32))

    def set_graph(self):
        self.init_idx = [0, 1]
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

    def outer_step(self,outer_action):
        self.hidden_goal = outer_action


        obs = dict()
        info = dict() #{'n': []}
        reward = dict()
        # demand_vec = self.get_demand_vec()
        for i, agent in enumerate(self.agents):
            obs[i] = np.append(self.observation_callback(self.world, i), self.hidden_goal[self.hidden_len * i : self.hidden_len * (i + 1) ])
            reward[i] = 0
            if i in self.inner_step_reward:
                reward[i] += self.inner_step_reward[i]

        self.coordinator_input = [self.global_observation_callback(self.world)]

        done_n = {"__all__": False}
        return obs, reward, done_n, info


    def inner_step(self, action):
        self.immediate_reward = dict()
        obs = dict()
        reward = dict()
        info = dict() #{'n': []}
        # set action for each agent
        action_out = np.zeros((len(self.world.agents) + len(self.world.destinations)) * 2)
        for i, agent in enumerate(self.agents):
            action_out += self._get_action(action[i], agent)

        self.update_inventory(action_out)
        # record observation for each agent
        self.current_step += 1

        if (self.current_step - 1) % (self.GP_len/self.coordinator_len) == 0 and self.current_step > 1:
            self.coordinator_input.append(self.global_observation_callback(self.world))

        if self.current_step < self.GP_len:
            for i, agent in enumerate(self.agents):
                reward[i] = 0
                obs[i] = np.append(self.observation_callback(self.world, i),
                                   self.hidden_goal[self.hidden_len * i: self.hidden_len * (i + 1)])

        elif self.current_step == self.GP_len:
            total_sparse_reward = 0
            self.current_step = 0
            self.current_GP += 1

            # inner_reward = self._get_leader_reward()
            inner_reward_sum = 0
            self.final_rewards = dict()
            self.inner_step_reward = dict()
            self.goal_reward = 0
            for i, agent in enumerate(self.agents):
                self.inner_step_reward[i] = 0


            if self.current_GP >= self.GP_num:
                sparse_reward = self._get_sparse_reward()
                for i, agent in enumerate(self.agents):
                    self.total_reward += sparse_reward[i]

            self.coordinator_input.append(self.global_observation_callback(self.world))
            self.coordinator_input = np.append(np.concatenate(self.coordinator_input), self.hidden_goal)
            # print(self.coordinator_input.shape)

            reward[-2] = 0
            obs[-2] = self.coordinator_input
            reward[-3] = 0
            obs[-3] = self.coordinator_input
            # self.world = self.change_GP(self.world)
        done_n = {"__all__": False}
        return obs, reward, done_n, info

    def coordinator_step(self, action_set):
        given_reward = action_set[-3][:-1] * self.previous_reward/self.GP_num/len(self.agents) * (self.multiplier * action_set[-3][-1])
        action = action_set[-2]
        obs = dict()
        reward = dict()

        for i, agent in enumerate(self.agents):
            if i in self.final_idx:
                self.final_rewards[i] = given_reward[i-2]

        self.backpropagation(action)

        reward[-1] = 0
        obs[-1] = self.global_observation_callback(self.world)

        if self.current_GP >= self.GP_num:
            done = {"__all__": True}
            self.errors = []
            self.errors2 = []
            reward[-2] = self.total_reward
            obs[-2] = self.coordinator_input
            reward[-3] = self.total_reward
            obs[-3] = self.coordinator_input
            reward[-1] = self.total_reward
            # demand_vec = self.get_demand_vec()
            self.previous_reward = abs(self.total_reward)
            for i, agent in enumerate(self.agents):
                reward[i] = self.inner_step_reward[i] + self.total_reward/len(self.agents)
                obs[i] = np.append(self.observation_callback(self.world, i),
                                   self.hidden_goal[self.hidden_len * i: self.hidden_len * (i + 1)])
        else:
            done = {"__all__": False}

        info = {}
        return obs, reward, done, info


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

    def bonus_backpropagation(self, organizer_act):
        current_ids = self.final_idx
        current_rwd = [self.bonus_final_rewards[i] for i in self.final_idx]
        continue_flag = True

        self.bonus_reward = dict()
        for i, agent in enumerate(self.agents):
            self.bonus_reward[i] = 0

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
                            self.bonus_reward[agent_id] += current_rwd[agent_idx] * probs[idx_]
                        else:
                            next_ids.append(next_id)
                            next_rwd.append(current_rwd[agent_idx] * probs[idx_])
                else:
                    self.bonus_reward[agent_id] += current_rwd[agent_idx]

            if len(next_ids) == 0:
                continue_flag = False
            else:
                current_ids = next_ids
                current_rwd = next_rwd


    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs = dict()
        obs[-1] = self.global_observation_callback(self.world)
        temp_demand = np.array([[10+np.random.randint(-5,0), 5+np.random.randint(-2,2)],[120+np.random.randint(-10,10), 80+np.random.randint(-10,10)],[40+np.random.randint(-5,5), 90+np.random.randint(-10,10)]])
        for idx in range(len(self.world.destinations)):
            # self.world.destinations[idx].demand = self.GP_num * self.world.destinations[idx].demand
            self.world.destinations[idx].demand = temp_demand[idx]
        # obs[-2] = np.random.random(2) #for observer
        self.current_step = 0
        self.current_GP = 0
        self.inner_reward = dict()
        self.inner_step_reward = dict()
        for i, agent in enumerate(self.agents):
            self.inner_reward[i] = 0
            self.inner_step_reward[i] = 0
        self.total_reward = 0

        self.old_inven = []
        for idx in range(len(self.world.destinations)):
            self.old_inven += [self.world.destinations[idx].inventory]
        self.old_inven = np.concatenate(self.old_inven)
        return obs


    # get observation for a particular agent
    def _get_obs(self, idx):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(self.world, idx)


    # get reward for a particular agent
    def _get_immediate_reward(self):
        reward = dict()
        for i in range(len(self.world.agents)):
            reward[i] = 0
        return reward

    def _get_sparse_reward(self):
        reward = dict()
        for i in range(len(self.world.agents)):
            reward[i] = 0
        path_cost = 0
        for path in self.world.paths:
            path_cost += path.count * path.cost
            # print(111, path.count, path.cost)
        inv_cost = 0
        for agent in self.world.agents:
            inv_cost += self.world.agent_inv_cost * sum(agent.inventory)

        # print(path_cost, inv_cost)
        for idx in self.world.final_idx:
            reward[idx] -= (path_cost+inv_cost)/len(self.world.final_idx)

        for destination in self.world.destinations:
            destination_reward = self.world.destination_rewards[destination.idx - 5]
            remaining = destination.inventory - destination.demand
            for i in range(2):
                if remaining[i] > 0:
                    destination_reward -= remaining[i] * self.world.destination_inv_cost
                else:
                    destination_reward -= np.abs(remaining[i]) * self.world.destination_under_cost
                    remaining[i] = 0
            destination_reward = max(0, destination_reward)
            destination.inventory = remaining

            total_count = 0
            for idx in destination.inbound_count:
                total_count += destination.inbound_count[idx]
            # print(destination.idx, total_count)
            if total_count > 0:
                for idx in destination.inbound_count:
                    reward[idx] += destination.inbound_count[idx]/total_count * destination_reward
                    destination.inbound_count[idx] = 0
            else:
                for idx in destination.inbound_count:
                    reward[idx] += destination_reward/len(destination.inbound_count)
        # print(reward, path_cost, inv_cost)
        return reward

    def _get_leader_reward(self):
        reward = dict()
        for i in range(len(self.world.agents)):
            reward[i] = 0
        for destination in self.world.destinations:
            idx = destination.idx - 5
            remaining = np.zeros(2)
            remaining[0] = self.A_demand[idx] - (destination.inventory[0] - destination.old_inventory[0])
            remaining[1] = self.B_demand[idx] - (destination.inventory[1] - destination.old_inventory[1])
            destination.old_inventory = copy(destination.inventory)
            # print(remaining)
            # remaining = destination.inventory - destination.demand
            # destination_reward = -sum(remaining**2)
            destination_reward = -sum(abs(remaining))

            total_count = 0
            for idx in destination.inbound_count2:
                total_count += destination.inbound_count2[idx]
            # print(destination.idx, total_count)
            if total_count > 0:
                for idx in destination.inbound_count2:
                    reward[idx] += destination.inbound_count2[idx]/total_count * destination_reward
                    destination.inbound_count2[idx] = 0
            else:
                for idx in destination.inbound_count:
                    reward[idx] += destination_reward/len(destination.inbound_count)
        # print(reward, path_cost, inv_cost)
        return reward

    # set env action for a particular agent
    def _get_action(self, action, agent):

        used_path = None
        n_agent = len(self.world.agents)
        n_total = len(self.world.agents) + len(self.world.destinations)
        action_out = np.zeros(n_total * 2)  # [A product] + [B product]
        if agent.idx == 0:
            if action < 2:
                action_out[agent.child_idx[action]] += 1
                used_path = agent.possible_path[action]

        elif agent.idx == 1:
            if action < 2:
                action_out[agent.child_idx[action]+n_total] += 1
                used_path = agent.possible_path[action]

        else:
            if action < 2:
                if agent.inventory[0] > 0:
                    action_out[agent.child_idx[action]] += 1
                    used_path = agent.possible_path[action]
                    agent.inventory[0] -= 1
                    if agent.child_idx[action] >= n_agent:
                        self.world.destinations[agent.child_idx[action]-n_agent].inbound_count[agent.idx] += 1
                        self.world.destinations[agent.child_idx[action]-n_agent].inbound_count2[agent.idx] += 1

            elif action < 4:
                if agent.inventory[1] > 0:
                    action_out[agent.child_idx[action-2] + n_total] += 1
                    used_path = agent.possible_path[action-2]
                    agent.inventory[1] -= 1
                    if agent.child_idx[action-2] >= n_agent:
                        self.world.destinations[agent.child_idx[action-2]-n_agent].inbound_count[agent.idx] += 1
                        self.world.destinations[agent.child_idx[action-2]-n_agent].inbound_count2[agent.idx] += 1

        if used_path != None:
            self.world.paths[self.world.connection_to_idx[used_path]].count += 1
        return action_out


    def update_inventory(self, action_sum):
        n_agent = len(self.world.agents)
        n_total = len(self.world.agents) + len(self.world.destinations)
        for i in range(len(action_sum)):
            agent_idx = i % n_total
            prod_idx = (i > n_total) * 1
            num_prod = action_sum[i]
            if agent_idx < n_agent:
                self.world.agents[agent_idx].inventory[prod_idx] += num_prod
            else:
                self.world.destinations[agent_idx - n_agent].inventory[prod_idx] += num_prod
