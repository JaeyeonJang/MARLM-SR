import gym
from gym import spaces
import numpy as np
import multiagent.scenarios as scenarios
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
# class MultiAgentEnv(gym.Env):

def preprocess(data, min, max):
    return (data - min) / (max - min)

class MultiAgentEnv(MultiAgentEnv):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, config=None, info_callback=None, done_callback=None, shared_viewer=True):
        self.episode_len = config['episode_len']
        self.GP_len = config['GP_len']
        self.coordinator_len = config['coordinator_len']
        self.current_step = 0
        scenario = scenarios.load(config['scenario_name']).Scenario()
        world = scenario.make_world(0)
        self.world = world
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(self.world.agents)
        # scenario callbacks
        self.reset_callback = scenario.reset_world
        # self.reward_callback = scenario.reward
        self.observation_callback = scenario.observation
        self.global_observation_callback = scenario.global_observation
        self.epi = 0
        self.previous_reward = 1
        self.current_shift_num = 0
        self.multiplier = config['multiplier']

        self.action_space = []
        for agent in world.agents:
            self.action_space.append(spaces.Discrete(5))

        self.hidden_len = config['hidden_len']
        self.total_hidden_len = len(self.agents) * self.hidden_len

        self.observation_space = dict()
        for i in range(14):
            if i in [0, 1]:
                obs_upper = np.array([self.world.range]*2 + [1] * 2 + [self.world.range] * 2 * 6 + [2] * 2 * 6 + [self.world.range] * 2 * 4 + [2] * 2 * 4 +
                                     [1] * 6 + [1] * self.hidden_len)  # for followers
                obs_lower = np.array([0]*2 + [-1] * 2 + [-self.world.range] * 2 * 6 + [-2] * 2 * 6 + [-self.world.range] * 2 * 4 + [-2] * 2 * 4 +
                                     [0] * 6 + [0] * self.hidden_len)  # for followers
                self.observation_space[i] = spaces.Box(low=obs_lower, high= obs_upper, dtype=np.float32)
            elif i in [2,3,4,5]:
                obs_upper = np.array([self.world.range]*2 + [1] * 2 + [self.world.range] * 2 * 6 + [2] * 2 * 6 + [self.world.range] * 2 * 4 + [2] * 2 * 4 +
                                     [1] * 6 + [1] * self.hidden_len)  # for followers
                obs_lower = np.array([0]*2 + [-1] * 2 + [-self.world.range] * 2 * 6 + [-2] * 2 * 6 + [-self.world.range] * 2 * 4 + [-2] * 2 * 4 +
                                     [0] * 6 + [0] * self.hidden_len)  # for followers
                self.observation_space[i] = spaces.Box(low=obs_lower, high= obs_upper, dtype=np.float32)
            else:
                obs_upper = np.array([self.world.range]*2 + [1] * 2 + [self.world.range] * 2 * 6 + [2] * 2 * 6 + [self.world.range] * 2 * 4 + [2] * 2 * 4 +
                                     [1] * 6 + [1] * self.hidden_len)  # for followers
                obs_lower = np.array([0]*2 + [-1] * 2 + [-self.world.range] * 2 * 6 + [-2] * 2 * 6 + [-self.world.range] * 2 * 4 + [-2] * 2 * 4 +
                                     [0] * 6 + [0] * self.hidden_len)  # for followers
                self.observation_space[i] = spaces.Box(low=obs_lower, high= obs_upper, dtype=np.float32)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

        self.die_time = np.ones(8) * -1
        self.rand_idx = np.arange(5)

        self.predictor_upper = np.array([self.world.range] * 2 * 14 + [1] * 2 * 14 + [self.world.range] * 4 * 2 + [1] * 4 * 2)
        self.predictor_lower = np.array([0] * 2 * 14 + [-1] * 2 * 14 + [0] * 4 * 2 + [-1] * 4 * 2)

        self.estimator_obs_upper, self.estimator_obs_lower = self.predictor_upper, self.predictor_lower
        self.set_graph()
        self.coordinator_observation_space = spaces.Box(low=np.append(np.concatenate([self.estimator_obs_lower] * 10), [0] * 14 + [0] * self.total_hidden_len),
                                                      high= np.append(np.concatenate([self.estimator_obs_upper] * 10), [1] * 14 + [1] * self.total_hidden_len), dtype=np.float32)

        self.coordinator_action_space = spaces.Box(np.array([0]*len(self.begin_point), dtype=np.float32),
                                             np.array([1]*len(self.begin_point), dtype=np.float32))


        obs_upper = np.array([self.world.range] * 2 * 14 + [1] * 2 * 14 + [self.world.range] * 4 * 2 + [1] * 4 * 2 + [1] * 14)
        obs_lower = np.array([0] * 2 * 14 + [-1] * 2 * 14 + [0] * 4 * 2 + [-1] * 4 * 2 + [0] * 14)
        self.leader_observation_space = spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32)
        self.leader_action_space = spaces.Box(low=np.array([0] * self.total_hidden_len), high=np.array([1] * self.total_hidden_len), dtype=np.float32)

        self.coordinator2_observation_space = spaces.Box(low=np.append(np.concatenate([self.estimator_obs_lower] * 10), [0] * 14 + [0] * self.total_hidden_len),
                                                      high= np.append(np.concatenate([self.estimator_obs_upper] * 10), [1] * 14 + [1] * self.total_hidden_len), dtype=np.float32)
        self.coordinator2_action_space = spaces.Box(low=np.array([0]*9, dtype=np.float32), high= np.array([1]*9, dtype=np.float32))

    def get_estimator_input(self):
        inp = preprocess(np.array([np.append(self.global_observation_callback(self.world),self.goal_set
                                             )]), self.predictor_lower, self.predictor_upper)
        return inp


    def set_graph(self):
        self.agent_paths = self.world.agent_paths
        self.final_idx = self.world.final_agents
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



    def change_env(self, action):
        for i, agent in enumerate(self.world.agents):
            if i not in self.world.die_idx:
                if action[i] == 0:
                    vel = np.array([1, 0])
                elif action[i] == 1:
                    vel = np.array([0, 1])
                elif action[i] == 2:
                    vel = np.array([-1, 0])
                elif action[i] == 3:
                    vel = np.array([0, -1])
                else:
                    vel = np.array([0, 0])
            else:
                vel = np.array([0, 0])
            temp_pos = agent.pos + vel
            if agent.boss != None:
                if np.abs(temp_pos[0]-self.world.agents[agent.boss].pos[0]) > self.world.agent_range or \
                        np.abs(temp_pos[1]-self.world.agents[agent.boss].pos[1]) > self.world.agent_range:
                    rel_pos = self.world.agents[agent.boss].pos - agent.pos
                    new_vel = np.zeros(2)
                    if rel_pos[0] != 0 and rel_pos[1] != 0:
                        idx = np.random.randint(2)
                        new_vel[idx] = np.sign(rel_pos[idx])
                    elif rel_pos[0] != 0:
                        new_vel[0] = np.sign(rel_pos[0])
                    elif rel_pos[1] != 0:
                        new_vel[1] = np.sign(rel_pos[1])
                    vel = new_vel
            temp_pos = agent.pos + vel
            if sum(temp_pos >= self.world.range) > 0 or sum(temp_pos < 0) > 0:
                vel = np.array([0, 0])

            agent.pos = agent.pos + vel
            agent.vel = vel

            # if i>=6:
            #     self.goal_set[2*(i-6):2*(i-5)] -= vel

    def predator_update(self):
        for i, predator in enumerate(self.world.predators):
            rel_pos = self.world.agents[predator.target_id].pos - predator.pos
            new_vel = np.zeros(2)
            if rel_pos[0] != 0 and rel_pos[1] != 0:
                idx = np.random.randint(2)
                new_vel[idx] = np.sign(rel_pos[idx])
            elif rel_pos[0] != 0:
                new_vel[0] = np.sign(rel_pos[0])
            elif rel_pos[1] != 0:
                new_vel[1] = np.sign(rel_pos[1])
            # temp_pos = predator.pos + new_vel
            # if sum(temp_pos >= self.world.range) > 0 and sum(temp_pos < 0) > 0:
            #     new_vel = np.array([0, 0])
            predator.vel = new_vel
            predator.pos = predator.pos + predator.vel

    def update_agent_states(self):
        new_die_idx = []
        for i, agent in enumerate(self.world.agents):
            if agent.alive and agent.level == 2:
                for j, predator in enumerate(self.world.predators):
                    if sum(agent.pos == predator.pos) == 2:
                        agent.alive = 0
                        new_die_idx.append(i)
                        self.world.die_idx.append(i)


        target_idx = []
        for i in np.arange(6, 6+8):
            if self.world.agents[i].alive:
                target_idx.append(i)
        if len(target_idx)>0:
            for i, predator in enumerate(self.world.predators):
                if predator.target_id in self.world.die_idx:
                    predator.target_id = np.random.choice(target_idx)
        return new_die_idx

    def step(self, action):
        if -1 in action:
            return self.outer_step(action[-1])
        elif -2 in action:
            return self.coordinator_step(action)
        else:
            return self.inner_step(action)

    def get_distance(self):
        distances = []
        for i in range(8):
            target = self.action[i]
            distance = np.sqrt(sum((np.array(self.world.predators[target].pos) - np.array(self.agents[i+6].pos)) ** 2))
            distances.append(distance)
        return np.array(distances)

    def inner_step(self, action):
        obs = dict()
        reward = dict()
        info = dict() #{'n': []}
        self.change_env(action)
        self.predator_update()
        new_die_idx = self.update_agent_states()
        if len(new_die_idx) > 0:
            for idx in new_die_idx:
                self.die_time[idx - 6] = self.current_step
        done = {"__all__": False}

        self.old_obs = self.global_observation_callback(self.world)
        self.coordinator_input.append(self.old_obs)

        self.current_step += 1
        if self.current_step < self.episode_len and len(self.world.die_idx) < 8:
            if (self.current_step + 1) % self.GP_len == 0:
                self.final_rewards = dict()
                self.inner_step_reward = dict()
                self.goal_reward = 0
                for i, agent in enumerate(self.agents):
                    self.inner_step_reward[i] = 0

                if len(self.coordinator_input) >= self.coordinator_len:
                    idx = np.append(
                        np.arange(0, len(self.coordinator_input), len(self.coordinator_input) / 9).astype(int), -1)
                    self.coordinator_input = np.array(self.coordinator_input)[idx]
                    self.coordinator_input = self.coordinator_input.flatten()
                else:
                    add = self.coordinator_len - len(self.coordinator_input)
                    self.coordinator_input = np.append([self.coordinator_input[0]] * add, self.coordinator_input)
                    self.coordinator_input = self.coordinator_input.flatten()

                entity_alive = []
                for entity in self.agents:  # world.entities:
                    entity_alive.append(entity.alive)
                self.coordinator_input = np.concatenate([self.coordinator_input, entity_alive, self.hidden_goal])
                # self.coordinator2_input = self.get_distance()

                self.coordinator_reward = 0
                reward[-2] = self.coordinator_reward
                obs[-2] = self.coordinator_input
                reward[-3] = self.coordinator_reward
                obs[-3] = self.coordinator_input



            else:
                for i, agent in enumerate(self.agents):
                    if i not in self.world.die_idx:
                        reward[i] = 0
                        # s, e = self.goal_idx(i)
                        obs[i] = np.append(self._get_obs(agent), self.hidden_goal[self.hidden_len * i : self.hidden_len * (i + 1)])

        else:
            self.final_rewards = dict()
            self.bonus_final_rewards = dict()
            self.inner_step_reward = dict()
            self.goal_reward = 0

            for i, agent in enumerate(self.agents):
                self.inner_step_reward[i] = 0

            # self.coordinator_reward = goal_total_reward
            for i, agent in enumerate(self.agents):
                if agent.level == 2:
                    if agent.alive:
                        agent_reward = self.episode_len
                        self.total_reward += agent_reward
                        # self.final_rewards[i] = agent_reward
                    else:
                        agent_reward = self.die_time[i - 6]
                        self.total_reward += agent_reward
                        # self.final_rewards[i] = agent_reward

            if len(self.coordinator_input) >= self.coordinator_len:
                idx = np.append(
                    np.arange(0, len(self.coordinator_input), len(self.coordinator_input) / 9).astype(int), -1)
                self.coordinator_input = np.array(self.coordinator_input)[idx]
                self.coordinator_input = self.coordinator_input.flatten()
            else:
                add = self.coordinator_len - len(self.coordinator_input)
                self.coordinator_input = np.append([self.coordinator_input[0]] * add, self.coordinator_input)
                self.coordinator_input = self.coordinator_input.flatten()

            entity_alive = []
            for entity in self.agents:  # world.entities:
                entity_alive.append(entity.alive)
            self.coordinator_input = np.concatenate([self.coordinator_input, entity_alive, self.hidden_goal])

            self.coordinator_reward = self.total_reward
            reward[-2] = self.coordinator_reward
            obs[-2] = self.coordinator_input
            reward[-3] = self.coordinator_reward
            obs[-3] = self.coordinator_input

        return obs, reward, done, info

    def coordinator_step(self, action_set):
        self.current_shift_num += 1
        given_reward = action_set[-3][:-1] * self.previous_reward/len(self.final_idx) * (self.multiplier * action_set[-3][-1])
        action = action_set[-2]
        obs = dict()
        reward = dict()

        j = 0
        for i, agent in enumerate(self.agents):
            if i in self.final_idx:
                self.final_rewards[i] = given_reward[j]
                j+=1

        self.backpropagation(action)


        reward[-1] = 0

        entity_alive = []
        for entity in self.agents:  # world.entities:
            entity_alive.append(entity.alive)
        obs[-1] = np.append(self.global_observation_callback(self.world), entity_alive)

        done = {"__all__": False}
        if self.current_step >= self.episode_len or len(self.world.die_idx) >= 8:
            self.current_step = 0
            self.current_round += 1
            reward[-1] = self.total_reward
            self.previous_reward = self.total_reward/self.current_shift_num
            self.current_shift_num = 0
            for i, agent in enumerate(self.agents):
                self.inner_step_reward[i] += self.total_reward/ len(self.agents)
            if self.current_round < 1:
                self.game_reset()
            else:
                self.coordinator_reward = 0
                reward[-2] = self.coordinator_reward
                obs[-2] = self.coordinator_input
                reward[-3] = self.coordinator_reward
                obs[-3] = self.coordinator_input

                for i, agent in enumerate(self.agents):
                    reward[i] = self.inner_step_reward[i]
                    # s, e = self.goal_idx(i)
                    obs[i] = np.append(self._get_obs(agent),
                                       self.hidden_goal[self.hidden_len * i: self.hidden_len * (i + 1)])
                done = {"__all__": True}
                self.errors = []
                self.errors2 = []
        info = {}
        return obs, reward, done, info

    def outer_step(self, action):
        self.hidden_goal = action
        reward = dict()
        obs = dict()
        for i, agent in enumerate(self.agents):
            reward[i] = self.inner_step_reward[i]
            self.inner_step_reward[i] = 0
            # s, e = self.goal_idx(i)
            obs[i] = np.append(self._get_obs(agent), self.hidden_goal[self.hidden_len * i: self.hidden_len * (i + 1)])
        self.old_obs = self.global_observation_callback(self.world)
        self.coordinator_input = [self.old_obs]

        info = dict() #{'n': []}
        done = {"__all__": False}
        return obs, reward, done, info

    def goal_idx(self, agent_id):
        if agent_id == 0:
            return 0 * 4, 4 * 4
        elif agent_id == 1:
            return 4 * 4, 8 * 4
        elif agent_id == 2:
            return 0 * 4, 2 * 4
        elif agent_id == 3:
            return 2 * 4, 4 * 4
        elif agent_id == 4:
            return 4 * 4, 6 * 4
        elif agent_id == 5:
            return 6 * 4, 8 * 4
        else:
            return (agent_id - 6) * 4, (agent_id - 5) * 4


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


    def game_reset(self):
        self.reset_callback(self.world, 0)
        self.agents = self.world.agents
        # # record observations for each agent
        # obs = dict()
        # for i, agent in enumerate(self.agents):
        #     obs[i] = self._get_obs(agent)
        self.old_obs = self.global_observation_callback(self.world)
        # self.coordinator_input = [self.old_obs]
        self.die_time = np.ones(8) * -1
        self.total_reward = 0
        # return obs


    def reset(self):
        # # reset world
        self.current_round = 0
        # np.random.shuffle(self.rand_idx)
        self.reset_callback(self.world, 0)
        self.agents = self.world.agents
        # # reset renderer
        self._reset_render()
        # # record observations for each agent
        obs = dict()
        entity_alive = []
        for entity in self.agents:  # world.entities:
            entity_alive.append(entity.alive)
        obs[-1] = np.append(self.global_observation_callback(self.world), entity_alive)
        self.current_step = 0
        self.die_time = np.ones(8) * -1
        # self.pick_score = 0
        self.goal_reward0 = 0
        self.goal_reward1 = 0
        self.inner_reward = dict()
        self.inner_step_reward = dict()
        for i, agent in enumerate(self.agents):
            self.inner_reward[i] = 0
            self.inner_step_reward[i] = 0
        self.errors = []
        self.errors2 = []
        self.total_reward = 0
        self.coordinator_reward = 0
        self.current_shift_num = 0
        return obs

    # # get info used for benchmarking
    # def _get_info(self, agent):
    #     if self.info_callback is None:
    #         return {}
    #     return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    # def _get_done(self, agent):
    #     if self.done_callback is None:
    #         return False
    #     return self.done_callback(agent, self.world)

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            # for agent in self.world.agents:
            #     comm = []
            #     for other in self.world.agents:
            #         if other is agent: continue
            #         if np.all(other.state.c == 0):
            #             word = '_'
            #         else:
            #             word = alphabet[np.argmax(other.state.c)]
            #         message += (other.name + ' to ' + agent.name + ': ' + word + '   ')

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.agents:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            for entity in self.world.predators:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)


            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = self.world.range
            buffer = 5
            # if self.shared_viewer:
            #     pos = np.zeros(self.world.dim_p)
            # else:
            #     pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(-buffer,buffer+cam_range,-buffer,buffer+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
