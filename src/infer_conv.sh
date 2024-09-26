accelerate launch infer_conv.py --dataset redial --split train --tokenizer utils/dialogpt --model utils/dialogpt_model --text_tokenizer utils/roberta --text_encoder utils/roberta_model --n_prefix_conv 20 --prompt_encoder output_dir/conv/dialogpt5e4v1/final --per_device_eval_batch_size 32 --context_max_length 128 --resp_max_length 128 --prompt_max_length 128 --entity_max_length 32
accelerate launch infer_conv.py --dataset redial --split valid --tokenizer utils/dialogpt --model utils/dialogpt_model --text_tokenizer utils/roberta --text_encoder utils/roberta_model --n_prefix_conv 20 --prompt_encoder output_dir/conv/dialogpt5e4v1/final --per_device_eval_batch_size 32 --context_max_length 128 --resp_max_length 128 --prompt_max_length 128 --entity_max_length 32
accelerate launch infer_conv.py --dataset redial --split test --tokenizer utils/dialogpt --model utils/dialogpt_model --text_tokenizer utils/roberta --text_encoder utils/roberta_model --n_prefix_conv 20 --prompt_encoder output_dir/conv/dialogpt5e4v1/final --per_device_eval_batch_size 32 --context_max_length 128 --resp_max_length 128 --prompt_max_length 128 --entity_max_length 32

# accelerate launch infer_conv.py --dataset inspired --split train --tokenizer utils/dialogpt --model utils/dialogpt_model --text_tokenizer utils/roberta --text_encoder utils/roberta_model --n_prefix_conv 20 --prompt_encoder output_dir_inspired/conv/dialogpt1e4/best --per_device_eval_batch_size 32 --context_max_length 128 --resp_max_length 128 --prompt_max_length 128 --entity_max_length 32
# accelerate launch infer_conv.py --dataset inspired --split valid --tokenizer utils/dialogpt --model utils/dialogpt_model --text_tokenizer utils/roberta --text_encoder utils/roberta_model --n_prefix_conv 20 --prompt_encoder output_dir_inspired/conv/dialogpt1e4/best --per_device_eval_batch_size 32 --context_max_length 128 --resp_max_length 128 --prompt_max_length 128 --entity_max_length 32
# accelerate launch infer_conv.py --dataset inspired --split test --tokenizer utils/dialogpt --model utils/dialogpt_model --text_tokenizer utils/roberta --text_encoder utils/roberta_model --n_prefix_conv 20 --prompt_encoder output_dir_inspired/conv/dialogpt1e4/best --per_device_eval_batch_size 32 --context_max_length 128 --resp_max_length 128 --prompt_max_length 128 --entity_max_length 32