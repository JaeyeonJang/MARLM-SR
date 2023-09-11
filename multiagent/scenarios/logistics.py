import numpy as np
from multiagent.core_logistic import Destination, Agent, World, Path
from multiagent.scenario_logistics import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, shift_len):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(5)]
        for i in range(len(world.agents)):
            world.agents[i].idx = i
        world.agents[0].possible_path = [(0, 2), (0, 4)]
        world.agents[1].possible_path = [(1, 2), (1, 3)]
        world.agents[2].possible_path = [(2, 4), (2, 6)]
        world.agents[3].possible_path = [(3, 6), (3, 7)]
        world.agents[4].possible_path = [(4, 5), (4, 7)]

        world.destinations = [Destination(shift_len) for i in range(3)]
        for i in range(len(world.destinations)):
            world.destinations[i].idx = i + 5
        world.destinations[0].inbound_count = {4: 0}
        world.destinations[1].inbound_count = {2: 0, 3: 0}
        world.destinations[2].inbound_count = {3: 0, 4: 0}

        world.destinations[0].inbound_count2 = {4: 0}
        world.destinations[1].inbound_count2 = {2: 0, 3: 0}
        world.destinations[2].inbound_count2 = {3: 0, 4: 0}

        for i in range(len(world.agents)):
            for path in world.agents[i].possible_path:
                world.agents[i].child_idx.append(path[1])
                if path[1] < len(world.agents):
                    world.agents[path[1]].parent_idx.append(i)
                    world.agent_paths.append(path)
                else:
                    world.destinations[path[1]-len(world.agents)].parent_idx.append(i)

        connections = []
        for i in range(len(world.agents)):
            connections += world.agents[i].possible_path

        world.paths = [Path() for i in range(10)]
        for i in range(len(world.paths)):
            world.paths[i].connection = connections[i]
            world.connection_to_idx[connections[i]] = i
        return world

    def reset_world(self, world):
        for i in range(len(world.agents)):
            world.agents[i].inventory[:] = 0
        for i in range(len(world.destinations)):
            world.destinations[i].reset()
        for i in range(len(world.paths)):
            world.paths[i].set_cost()
            world.paths[i].count = 0
        return world

    def change_GP(self, world):
        for i in range(len(world.paths)):
            # world.paths[i].set_cost()
            world.paths[i].count = 0
        return world

    def observation(self, world, idx): #for agent idx
        my_inven = world.agents[idx].inventory
        parent_inven = []
        child_inven = []
        path_cost = []
        for p_idx in world.agents[idx].parent_idx:
            if p_idx not in [0, 1]:
                parent_inven+=[world.agents[p_idx].inventory]
        for c_idx in world.agents[idx].child_idx:
            if c_idx < len(world.agents):
                child_inven += [world.agents[c_idx].inventory]
            else:
                child_inven += [world.destinations[c_idx-len(world.agents)].inventory]
        for path in world.agents[idx].possible_path:
            path_idx = world.connection_to_idx[path]
            path_cost.append(world.paths[path_idx].cost)
        path_cost = np.array(path_cost)
        child_inven = np.concatenate(child_inven)
        if len(parent_inven) > 0:
            parent_inven = np.concatenate(parent_inven)
            return np.concatenate([my_inven] + [parent_inven] + [child_inven] + [path_cost])
        else:
            return np.concatenate([my_inven] + [child_inven] + [path_cost])

    def observation_bound(self, world, idx, shift_num, shift_len): #for agent idx
        my_inven = np.array([shift_len * shift_num, shift_len * shift_num])
        parent_inven = []
        child_inven = []
        path_cost = []
        for p_idx in world.agents[idx].parent_idx:
            if p_idx not in [0, 1]:
                parent_inven+= [np.array([shift_len * shift_num]*2)]
        for c_idx in world.agents[idx].child_idx:
            child_inven+=[np.array([shift_len * shift_num]*2)]
        for path in world.agents[idx].possible_path:
            path_cost.append(1)
        path_cost = np.array(path_cost)
        child_inven = np.concatenate(child_inven)
        if len(parent_inven) > 0:
            parent_inven = np.concatenate(parent_inven)
            upper_bound = np.concatenate([my_inven] + [parent_inven] + [child_inven] + [path_cost])
        else:
            upper_bound = np.concatenate([my_inven] + [child_inven] + [path_cost])
        lower_bound = np.zeros(len(upper_bound))
        return upper_bound, lower_bound

    def global_observation(self, world):
        inven = []
        path_cost = []
        demand = []
        for idx in range(len(world.agents)):
            if idx not in [0, 1]:
                inven+=[world.agents[idx].inventory]
        for idx in range(len(world.destinations)):
            inven += [world.destinations[idx].inventory]
        for idx in range(len(world.destinations)):
            demand += [world.destinations[idx].demand]
        for idx in range(len(world.paths)):
            path_cost.append(world.paths[idx].cost)
        path_cost = np.array(path_cost)
        inven = np.concatenate(inven)
        demand = np.concatenate(demand)
        return np.concatenate([inven] + [demand] + [path_cost])

    def global_observation_bound(self, world, shift_num, shift_len):
        inven = []
        path_cost = []
        demand = []
        for idx in range(len(world.agents)):
            if idx not in [0, 1]:
                inven+=[np.array([shift_len * shift_num]*2)]
        for idx in range(len(world.destinations)):
            inven += [np.array([shift_len * shift_num]*2)]
        for idx in range(len(world.destinations)):
            demand += [np.array([shift_len * shift_num]*2)]
        for idx in range(len(world.paths)):
            path_cost.append(1)
        path_cost = np.array(path_cost)
        inven = np.concatenate(inven)
        demand = np.concatenate(demand)
        upper_bound = np.concatenate([inven] + [demand] + [path_cost])
        lower_bound = np.zeros(len(upper_bound))
        return upper_bound, lower_bound