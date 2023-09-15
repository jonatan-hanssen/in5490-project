# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama import Llama
import os

base_path = os.path.dirname(__file__)

generator = Llama.build(
    ckpt_dir=os.path.join(base_path, "../llama-2-7b-chat"),
    tokenizer_path=os.path.join(base_path, "../llama-2-7b-chat/tokenizer.model"),
    max_seq_len=512,
    max_batch_size=6,
)

dialog = [
    {
        "role": "system",
        "content": "You are a player playing a videogame. It is a top down turn based game, where each turn you can either move RIGHT, LEFT, FORWARD, PICK UP or DROP objects, or TOGGLE objects in front of you.",
    }
]


while True:
    inp = input("USER: ")

    dialog.append({"role": "user", "content": inp})

    results = generator.chat_completion(
        [dialog],  # type: ignore
        max_gen_len=15,
        temperature=0.6,
        top_p=0.9,
    )
    output = results[0]["generation"]["content"]
    print(output)

    dialog.append({"role": "assistant", "content": output})
