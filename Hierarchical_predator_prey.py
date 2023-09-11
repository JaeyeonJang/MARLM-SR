import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import ray
from multiagent.env_hierarchical_predator_prey import MultiAgentEnv
from rllib_mod.ppo import PPOTrainer

BASE_PATH = './'
ray.init()
lr = 0.0001
num_workers = 4
rollout_fragment_length = 1200
train_batch_size = 1024
sgd_minibatch_size = 256
num_sgd_iter = 4
num_of_iter = 5000
checkpoint_iter = 500
save_path = BASE_PATH + 'saved_data/'
hidden_len = 3 #the length of goal vector
multiplier = 0.1 #the maximum value of team reward signal
GP_len = 10 #the length of goal period

env_config = {
    'scenario_name': 'hierarchical_predator_prey.py',
    'episode_len': 200,
    'GP_len': GP_len,
    'hidden_len': hidden_len,
    'multiplier': multiplier,
    'coordinator_len': 10, #the number of global states collected for the RGD
}
single_env = MultiAgentEnv(config=env_config)

ppo_trainer = PPOTrainer(
    env=MultiAgentEnv,
    config={
        "env_config": {
            'scenario_name': 'hierarchical_predator_prey.py',
            'episode_len': 200,
            'GP_len': GP_len,
            'hidden_len': hidden_len,
            'multiplier': multiplier,
            'coordinator_len': 10,
        },
        "lr": lr,
        "num_gpus": 0,
        "kl_coeff": 0.0,
        "num_workers": num_workers,
        "rollout_fragment_length": rollout_fragment_length,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "gamma": 0.999,
        "lambda": 0.95,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "clip_param": 0.2,
        "vf_clip_param": 10,
        "tfboard_path": BASE_PATH + 'hierarchical_predator_prey_GP%i'%GP_len,
        "multiagent": {
            "policies": {
                "coordinator2": (
                None, single_env.coordinator2_observation_space, single_env.coordinator2_action_space, {}),
                "coordinator": (None, single_env.coordinator_observation_space, single_env.coordinator_action_space, {}),
                "leader": (None, single_env.leader_observation_space, single_env.leader_action_space, {}),
                "agent1": (None, single_env.observation_space[0], single_env.action_space[0], {}),
                "agent2": (None, single_env.observation_space[1], single_env.action_space[1], {}),
                "agent3": (None, single_env.observation_space[2], single_env.action_space[2], {}),
                "agent4": (None, single_env.observation_space[3], single_env.action_space[3], {}),
                "agent5": (None, single_env.observation_space[4], single_env.action_space[4], {}),
                "agent6": (None, single_env.observation_space[5], single_env.action_space[5], {}),
                "agent7": (None, single_env.observation_space[6], single_env.action_space[6], {}),
                "agent8": (None, single_env.observation_space[7], single_env.action_space[7], {}),
                "agent9": (None, single_env.observation_space[8], single_env.action_space[8], {}),
                "agent10": (None, single_env.observation_space[9], single_env.action_space[9], {}),
                "agent11": (None, single_env.observation_space[10], single_env.action_space[10], {}),
                "agent12": (None, single_env.observation_space[11], single_env.action_space[11], {}),
                "agent13": (None, single_env.observation_space[12], single_env.action_space[12], {}),
                "agent14": (None, single_env.observation_space[13], single_env.action_space[13], {}),

            },
            "policy_mapping_fn": lambda
                x: "coordinator2" if x == -3 else "coordinator" if x == -2 else "leader" if x == -1 else "agent1" if x == 0 else "agent2" if x == 1 else "agent3" if x == 2 else "agent4" if x == 3 else "agent5" if x == 4 else
            "agent6" if x == 5 else "agent7" if x == 6 else "agent8" if x == 7 else "agent9" if x == 8 else "agent10" if x == 9 else
            "agent11" if x == 10 else "agent12" if x == 11 else "agent13" if x == 12 else "agent14",
            "policies_to_train": ["coordinator2", "coordinator", "leader", "agent1", "agent2", "agent3", "agent4", "agent5", "agent6", "agent7", "agent8", "agent9", "agent10",
                                  "agent11", "agent12", "agent13", "agent14"],
        },
        "monitor": True,
        "model": {
            "fcnet_hiddens": [128, 128],
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
        ppo_trainer.save(BASE_PATH + "save/hierarchical_predator_prey_checkpoint_" + str(lr) + '_' + str(i + 1))


