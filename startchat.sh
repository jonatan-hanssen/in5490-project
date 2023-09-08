torchrun --nproc_per_node 1 src/chatdemo.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path llama-2-7b-chat/tokenizer.model --max_seq_len 512 --max_batch_size 6
