import numpy as np

class Inventory(object):
    def __init__(self):
        self.level0_outputs = ['a', 'b', 'c', 'd']
        self.level1_outputs = ['A', 'B', 'C', 'D', 'E', 'F']
        self.level2_outputs = ['AD', 'BE', 'CF']

        self.level0_output_dict = dict()
        self.level1_output_dict = dict()
        self.level2_output_dict = dict()

        for i in range(len(self.level0_outputs)):
            self.level0_output_dict[self.level0_outputs[i]] = i
            self.level0_output_dict[i] = self.level0_outputs[i]

        for i in range(len(self.level1_outputs)):
            self.level1_output_dict[self.level1_outputs[i]] = i
            self.level1_output_dict[i] = self.level1_outputs[i]

        for i in range(len(self.level2_outputs)):
            self.level2_output_dict[self.level2_outputs[i]] = i
            self.level2_output_dict[i] = self.level2_outputs[i]

        self.level0 = []
        self.level1 = []
        self.level2 = []

        for part in self.level0_outputs:
            self.level0.append(0.0)
        for part in self.level1_outputs:
            self.level1.append(0.0)
        for part in self.level2_outputs:
            self.level2.append(0.0)

        self.level0 = np.array(self.level0)
        self.level1 = np.array(self.level1)
        self.level2 = np.array(self.level2)

        self.level0_cost = 0.3
        self.level1_cost = 0.8

class Product_value(object):
    def __init__(self):
        self.values = np.array([2,3,4])
        self.reset_value()

    def reset_value(self):
        np.random.shuffle(self.values)

class Demand_value(object):
    def __init__(self):
        self.values = np.random.random(3)
        self.values = self.values/self.values.sum() * 10
        self.values = self.values.astype(int)


    def reset_value(self):
        self.values = np.random.random(3)
        self.values = self.values/self.values.sum() * 10
        self.values = self.values.astype(int)



class Agent(object):
    def __init__(self):
        self.level = None
        self.outputs = None
        self.parts = None

class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.inventory = Inventory()
        self.product_value = Product_value()
        self.demand_value = Demand_value()
        self.gamma = 0.99