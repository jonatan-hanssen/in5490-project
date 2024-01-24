from ppo import PPO
from llama import Llama
import os

base_path = os.path.dirname(__file__)

generator = Llama.build(
    ckpt_dir=os.path.join(base_path, "../llama-2-7b-chat"),
    tokenizer_path=os.path.join(
        base_path, "../llama-2-7b-chat/tokenizer.model"
    ),
    max_seq_len=2048,
    max_batch_size=6,
)

# make the generator something stupid if you think it will never be called
# because of caching. Good if you want to run it with out a big gpu
# generator = "a straight up dog"


for i in range(1, 9):
    print("--------------------------------------")
    print(f"Starting policy {i}")
    ppo = PPO(
            "hyperparams.json",
            result_file=f"data/unlockpickup_policy{i}05-1",
            reward=False,
            policy=True,
            generator=generator,
            policy_sim_thres=0.5,
            policy_sim_mod=1,
        )
    ppo.train()
