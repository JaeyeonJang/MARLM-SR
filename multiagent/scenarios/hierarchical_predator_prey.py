import numpy as np
from multiagent.core_hierarchical_predator_prey import World, Agent, Predator
from multiagent.scenario_hierarchical_predator_prey import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World()
        # set any world properties first
        num_agents = 2 + 4 + 8 # no leader
        num_predetors = 4

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.predators = [Predator() for i in range(num_predetors)]

        # make initial conditions
        self.reset_world(world, seed)
        return world

    def reset_world(self, world, seed):
        # random properties for agents
        levels = [0]*2 + [1]*4 + [2]*8
        bosses = [None, None, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        color_set = [np.array([0,1,0]), np.array([0,0.5,0.5]), np.array([0,0,1])]
        world.final_agents = []
        world.agent_paths = []

        agent_pos = world.agent_pos[seed]
        pred_pos = world.pred_pos[seed]

        for i, agent in enumerate(world.agents):
            agent.idx = i
            agent.level = levels[i]
            agent.boss = bosses[i]
            agent.color = color_set[agent.level]
            agent.alive = 1

            agent.pos = np.array(agent_pos[i])
            agent.default_pos = np.array(agent_pos[i])


            if agent.boss != None:
                world.agent_paths.append((agent.boss, i))
            if agent.level == 2:
                world.final_agents.append(i)

        targets = np.arange(6, 6+8)
        np.random.shuffle(targets)
        for i, predator in enumerate(world.predators):
            predator.target_id = targets[i]
            predator.color = np.array([1, 0, 0])
            predator.pos = np.array(pred_pos[i])

        world.die_idx = []



    def observation(self, agent, world):
        if agent.idx in world.group1_idx:
            idx = world.group1_idx
        else:
            idx = world.group2_idx

        entity_pos = []
        for entity in world.agents:  # world.entities:
            if entity.idx in idx and agent != entity:
                entity_pos.append(entity.pos - agent.pos)

        entity_vel = []
        for entity in world.agents:  # world.entities:
            if entity.idx in idx and agent != entity:
                entity_pos.append(entity.vel - agent.vel)

        entity_alive = []
        for entity in world.agents:  # world.entities:
            if entity.idx in idx and agent != entity:
                entity_alive.append(entity.alive)

        predator_pos = []
        for predator in world.predators:  # world.entities:
            predator_pos.append(predator.pos - agent.pos)

        predator_vel = []
        for predator in world.predators:  # world.entities:
            predator_vel.append(predator.vel - agent.vel)

        return np.concatenate([agent.pos]+[agent.vel]+entity_vel + entity_pos + predator_pos + predator_vel + [entity_alive])

    def global_observation(self, world):
        entity_pos = []
        for entity in world.agents:  # world.entities:
            entity_pos.append(entity.pos)

        entity_vel = []
        for entity in world.agents:  # world.entities:
            entity_pos.append(entity.vel)

        # entity_alive = []
        # for entity in world.agents:  # world.entities:
        #     entity_alive.append(entity.alive)

        predator_pos = []
        for predator in world.predators:  # world.entities:
            predator_pos.append(predator.pos)


        predator_vel = []
        for predator in world.predators:  # world.entities:
            predator_vel.append(predator.vel)

        return np.concatenate(entity_vel + entity_pos + predator_pos + predator_vel) # + [entity_alive]

    def global_observation(self, world):
        entity_pos = []
        for entity in world.agents:  # world.entities:
            entity_pos.append(entity.pos)

        entity_vel = []
        for entity in world.agents:  # world.entities:
            entity_pos.append(entity.vel)

        # entity_alive = []
        # for entity in world.agents:  # world.entities:
        #     entity_alive.append(entity.alive)

        predator_pos = []
        for predator in world.predators:  # world.entities:
            predator_pos.append(predator.pos)


        predator_vel = []
        for predator in world.predators:  # world.entities:
            predator_vel.append(predator.vel)

        return np.concatenate(entity_vel + entity_pos + predator_pos + predator_vel) # + [entity_alive]

    def real_global_observation(self, world):
        entity_pos = []
        for entity in world.agents:  # world.entities:
            entity_pos.append(entity.pos)

        entity_vel = []
        for entity in world.agents:  # world.entities:
            entity_pos.append(entity.vel)

        entity_alive = []
        for entity in world.agents:  # world.entities:
            entity_alive.append(entity.alive)

        predator_pos = []
        for predator in world.predators:  # world.entities:
            predator_pos.append(predator.pos)


        predator_vel = []
        for predator in world.predators:  # world.entities:
            predator_vel.append(predator.vel)

        return np.concatenate(entity_vel + entity_pos + predator_pos + predator_vel + [entity_alive])

    def default_global_observation(self, world, agent_idx):
        entity_pos = []
        for idx, entity in enumerate(world.agents):  # world.entities:
            if idx != agent_idx:
                entity_pos.append(entity.pos)
            else:
                entity_pos.append(entity.default_pos)

        entity_vel = []
        for idx, entity in enumerate(world.agents):  # world.entities:
            if idx != agent_idx:
                entity_vel.append(entity.pos)
            else:
                entity_vel.append(entity.default_vel)

        entity_alive = []
        for entity in world.agents:  # world.entities:
            entity_alive.append(entity.alive)

        predator_pos = []
        for predator in world.predators:  # world.entities:
            predator_pos.append(predator.pos)


        predator_vel = []
        for predator in world.predators:  # world.entities:
            predator_vel.append(predator.vel)

        return np.concatenate(entity_vel + entity_pos + predator_pos + predator_vel + [entity_alive])

