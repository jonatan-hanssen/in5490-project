# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama import Llama
import os

base_path = os.path.dirname(__file__)

generator = Llama.build(
    ckpt_dir=os.path.join(base_path, "../llama-2-7b-chat"),
    tokenizer_path=os.path.join(base_path, "../llama-2-7b-chat/tokenizer.model"),
    max_seq_len=2048,
    max_batch_size=6,
)

dialog = [
    {
        "role": "system",
        "content": "",
    }
]

dialog = []


while True:
    inp = input("USER: ")
    dialog.append({"role": "user", "content": inp})

    results = generator.chat_completion(
        [dialog],  # type: ignore
        max_gen_len=1024,
        temperature=0.6,
        top_p=0.9,
    )
    output = results[0]["generation"]["content"]

    print(len(dialog))

    dialog.append({"role": "assistant", "content": output})
    print(output)
