from ppo import PPO
from utils import llama2_policy

consigliere = llama2_policy("use the key to open the door and then get to the goal", cos_sim_threshold=0, similarity_modifier=0.1)

print("Starting policy 1")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy1",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.agent.consigliere.reset_cache()
ppo.train()

print("Starting policy 2")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy2",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.agent.consigliere.reset_cache()
ppo.train()

print("Starting policy 3")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy3",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.agent.consigliere.reset_cache()
ppo.train()

print("Starting policy 4")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy4",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.agent.consigliere.reset_cache()
ppo.train()

print("Starting policy 5")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy5",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.agent.consigliere.reset_cache()
ppo.train()

print("Starting policy 6")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy6",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.train()

print("Starting policy 7")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy7",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.train()

print("Starting policy 8")
ppo = PPO(
        "hyperparams.json",
        result_file="data/doorkey_policy8",
        reward=False,
        policy=True,
        consigliere=consigliere,
    )
ppo.train()
