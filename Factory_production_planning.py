import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import ray
from multiagent.env_factory_production_planning import MultiAgentEnv
from rllib_mod.ppo import PPOTrainer

BASE_PATH = './'
ray.init()
lr = 0.0001
num_workers = 4
rollout_fragment_length = 512
train_batch_size = 512
sgd_minibatch_size = 128
num_sgd_iter = 4
num_of_iter = 5000
checkpoint_iter = 1000
save_path = BASE_PATH+'saved_data/'
hidden_len = 3 #the length of goal vector
multiplier = 10 #the maximum value of team reward signal
GP_len = 40 #the length of goal period
GP_num = int(400/GP_len) #the number of goal periods

env_config = {
    'scenario_name': 'factory_production_planning.py',
    'GP_len': GP_len,
    'GP_num': GP_num,
    'coordinator_len': 5, #1+coordinator_len is the number of global states collected for the RGD
    'hidden_len': hidden_len,
    'multiplier': multiplier,
}

single_env = MultiAgentEnv(config=env_config)
ppo_trainer = PPOTrainer(
    env=MultiAgentEnv,
    config={
        "env_config": {
            'scenario_name': 'factory_production_planning.py',
            'GP_len': GP_len,
            'GP_num': GP_num,
            'coordinator_len': 5,
            'hidden_len': hidden_len,
            'multiplier': multiplier,
        },
        "lr": lr,
        "num_gpus": 0,
        "kl_coeff": 0.0,
        "num_workers": num_workers,
        "rollout_fragment_length": rollout_fragment_length,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "gamma": 0.99,
        "lambda": 0.95,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "clip_param": 0.05,
        "vf_clip_param": 10,
        "tfboard_path": BASE_PATH + 'factory_production_planning_GP%i'%GP_len,
        "multiagent": {
            "policies": {
                "coordinator2": (None, single_env.coordinator2_observation_space, single_env.coordinator2_action_space, {}),
                "coordinator": (None, single_env.coordinator_observation_space, single_env.coordinator_action_space, {}),
                "leader": (None, single_env.leader_observation_space, single_env.leader_action_space, {}),
                "agent1": (None, single_env.observation_space, single_env.action_space[0], {}),
                "agent2": (None, single_env.observation_space, single_env.action_space[1], {}),
                "agent3": (None, single_env.observation_space, single_env.action_space[2], {}),
                "agent4": (None, single_env.observation_space, single_env.action_space[3], {}),
            },
            "policy_mapping_fn": lambda
                x: "coordinator2" if x == -3 else "coordinator" if x == -2 else "leader" if x == -1 else "agent1" if x == 0 else "agent2" if x == 1 else "agent3" if x == 2 else "agent4",
            "policies_to_train": ["coordinator2", "coordinator", "leader", "agent1", "agent2", "agent3", "agent4"],
        },
        "monitor": True,
        "model": {
            "fcnet_hiddens": [128,128], #[128,128]
            "vf_share_layers": False,
        },
    },
)

real_result = []
for i in range(num_of_iter):
    print("== Iteration", i, "==")
    result = ppo_trainer.train()
    real_result.extend(result['hist_stats']['policy_leader_reward'])
    if (i + 1) % checkpoint_iter == 0:
        ppo_trainer.save(BASE_PATH + "save/factory_production_planning_checkpoint_" + str(lr) + '_' + str(i + 1))

