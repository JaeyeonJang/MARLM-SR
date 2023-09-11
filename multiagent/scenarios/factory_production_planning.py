import numpy as np
from multiagent.core_factory_production_planning import Inventory, Agent, World
from multiagent.scenario_factory_production_planning import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(4)]
        world.agents[0].level = 0
        world.agents[1].level = 1
        world.agents[2].level = 1
        world.agents[3].level = 2

        world.agents[0].outputs = world.inventory.level0_outputs
        world.agents[1].outputs = world.inventory.level1_outputs[:3]
        world.agents[2].outputs = world.inventory.level1_outputs[3:]
        world.agents[3].outputs = world.inventory.level2_outputs

        world.agents[1].parts = np.array([[2,0,0,0],[1,1,0,0],[0,2,0,0]])
        world.agents[2].parts = np.array([[0,0,2,0],[0,0,1,1],[0,0,0,2]])
        world.agents[3].parts = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])

        return world

    def reset_world(self, world):
        world.inventory.level0[:] = 0
        world.inventory.level1[:] = 0
        world.inventory.level2[:] = 0
        world.product_value.reset_value()
        world.demand_value.reset_value()
        return world

    def reset_value(self,world):
        world.product_value.reset_value()
        world.demand_value.reset_value()
        return world


    def observation(self, world):
        return np.concatenate([world.inventory.level0] + [world.inventory.level1]+[world.inventory.level2]+[world.product_value.values] + [world.demand_value.values])

