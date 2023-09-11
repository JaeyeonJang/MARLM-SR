import numpy as np

# physical/external base state of all entites
class Agent(object):
    def __init__(self):
        # physical position
        self.pos = None
        self.default_pos = None
        # properties:
        self.size = 0.50
        # for hierarchy
        self.level = None
        self.boss = None
        self.color = None
        self.idx = None
        self.vel = np.zeros(2)
        self.default_vel = np.zeros(2)
        self.alive = 1
        self.name = 'agent'

class Predator(Agent):
    def __init__(self):
        super(Predator, self).__init__()
        # score for landmark
        self.target_id = None
        self.name = 'predator'

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.predators = []
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        self.range = 30
        self.agent_range = 5
        self.agent_occupy_map = np.zeros((self.range, self.range))
        self.final_agents = []
        self.agent_paths = []
        self.group1_idx = [0, 2, 3, 6, 7, 8, 9]
        self.group2_idx = [1, 4, 5, 10, 11, 12, 13]
        self.die_idx = []


        for i in range(10000):
            np.random.seed(i)
            center = np.random.randint(3, 26, size=(5, 2, 2))
            pred_center = np.random.randint(0,29,size = (5, 4, 2))
            break_signal = False
            for t in range(5):
                if sum(np.abs(center[t, 0] - center[t, 1]) < 5) == 2:
                    break_signal = True
                    break
                for j in range(4):
                    for k in range(2):
                        if sum(np.abs(center[t,k] - pred_center[t,j]) < 5) == 2:
                            break_signal = True
                            break
                    if break_signal:
                        break
                if break_signal:
                    break
            if not break_signal:
                break


        self.agent_pos = [[center[k,0], center[k,1], center[k,0]+1, center[k,0]-1, center[k,1]+1, center[k,1]-1, center[k,0]+2, center[k,0]+3,
                           center[k,0]-2, center[k,0]-3, center[k,1]+2, center[k,1]+3, center[k,1]-2, center[k,1]-3] for k in range(5)]

        self.pred_pos = [[pred_center[k,0], pred_center[k,1], pred_center[k,2], pred_center[k,3]] for k in range(5)]

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.predators

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]
