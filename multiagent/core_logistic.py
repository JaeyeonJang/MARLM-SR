import numpy as np

class Destination(object):
    def __init__(self, shift_len):
        self.shift_len = shift_len
        self.idx = None
        self.parent_idx = []
        self.product_types = ['A', 'B']
        # np.random.seed(0)
        a_demand = np.random.randint(int(self.shift_len/3.0))
        self.inventory = np.array([0, 0])
        self.old_inventory = np.array([0, 0])
        self.demand = np.array([a_demand, int(self.shift_len/3.0) - a_demand])
        # self.demand = np.array([2, 1])
        self.inbound_count = None
        self.inbound_count2 = None

    def reset_inventory(self):
        self.inventory = np.array([0, 0])
        self.old_inventory = np.array([0, 0])

    def reset_demand(self):
        # np.random.seed(self.idx)
        a_demand = np.random.randint(int(self.shift_len/3.0))
        self.demand = np.array([a_demand, int(self.shift_len/3.0) - a_demand])
        # self.demand = np.array([2, 1])

    def reset(self):
        self.reset_inventory()
        self.reset_demand()

class Agent(object):
    def __init__(self):
        self.idx = None
        self.parent_idx = []
        self.child_idx = []
        self.possible_path = None
        self.inventory = np.array([0, 0]) #a, b

class Path(object): #need weight or cost?
    def __init__(self):
        self.connection = None
        self.count = 0
        self.set_cost()

    def set_cost(self):
        self.cost = np.random.random() * 0.3
        # self.cost = 0.2

class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.destinations = []
        self.paths = []
        self.connection_to_idx = dict()
        self.agent_inv_cost = 0.3
        self.destination_inv_cost = 3
        self.destination_under_cost = 8
        self.final_idx = [2, 3, 4]
        self.destination_rewards = [100, 300, 200]
        self.agent_paths = [] #paths connecting only agents
        self.gamma = 0.99