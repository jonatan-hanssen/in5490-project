from ppo import PPO
from utils import llama2_policy

consigliere = llama2_policy(goal, cos_sim_threshold=0, similarity_modifier=0.1)
