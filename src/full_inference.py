import argparse
import math
import os
import sys
import time
import json

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_conv import CRSConvDataCollator, CRSConvDataset
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_conv import ConvEvaluator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt


if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()

set_seed(42)

accelerator = Accelerator(device_placement=False, mixed_precision = 'fp16')
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained('utils/dialogpt')
tokenizer.add_special_tokens(gpt2_special_tokens_dict)
model = PromptGPT2forCRS.from_pretrained('utils/dialogpt_model')
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)

text_tokenizer = AutoTokenizer.from_pretrained('utils/roberta')
text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
text_encoder = AutoModel.from_pretrained('utils/roberta_model')
text_encoder.resize_token_embeddings(len(text_tokenizer))
text_encoder = text_encoder.to(device)

#####################################################################################
kg = DBpedia(dataset= 'sample_input', debug= False).get_entity_kg_info()
prompt_encoder = KGPrompt(
    model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
    n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases= 8,
    edge_index=kg['edge_index'], edge_type=kg['edge_type'],
    n_prefix_conv= 20
)

prompt_encoder.load('output_dir/prompt/dialogpt1e3/best')
prompt_encoder = prompt_encoder.to(device)
prompt_encoder = accelerator.prepare(prompt_encoder)


dataset = CRSConvDataset(
    dataset = 'redial', split = 'sample_input', tokenizer, debug= False,
    context_max_length = 128, resp_max_length=81, entity_max_length=32,
    prompt_tokenizer = 'utils/roberta', prompt_max_length= 128
)
data_collator_generator = CRSConvDataCollator(
    tokenizer = tokenizer, device= device, gen=True, use_amp = accelerator.use_fp16, debug= False,
    ignore_pad_token_for_loss= True,
    context_max_length= 128, resp_max_length=81,
    entity_max_length= 32, pad_entity_id=kg['pad_entity_id'],
    prompt_tokenizer= 'utils/roberta'
)
dataloader = DataLoader(
    dataset,
    batch_size= 32,
    num_workers = 4,
    collate_fn = data_collator_generator,
)
gen_dir = os.path.join('save', 'redial')
os.makedirs(gen_dir, exist_ok=True)
model_name = 'output_dir/conv/dialogpt5e4/final'.split('/')[-2]
gen_file_path = os.path.join(gen_dir, f'{model_name}_sample_input.jsonl')
evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)

for batch in tqdm(dataloader, disable= not accelerator.is_local_main_process):
    with torch.no_grad():
        token_embeds = text_encoder(**batch['prompt']).last_hidden_state
        prompt_embeds = prompt_encoder(
            entity_ids=batch['entity'],
            token_embeds=token_embeds,
            output_entity=False,
            use_conv_prefix=True
        )
        batch['context']['prompt_embeds'] = prompt_embeds

        gen_seqs = accelerator.unwrap_model(model).generate(
            **batch['context'],
            max_new_tokens= 50,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,  # Add this line
            eos_token_id=tokenizer.eos_token_id # add this also
        )
        gen_resp_ids = []
        for gen_seq, length in zip(gen_seqs, batch['context_len']):
            gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
            gen_resp_ids.append(gen_seq[length:])
        
        for resp_ids in gen_resp_ids:
            decoded_resp = tokenizer.decode(resp_ids, skip_special_tokens= False)
            print("-----------------------------------")
            print("Generated response:", decoded_resp)
            print("-----------------------------------")

        evaluator.evaluate(gen_resp_ids, batch['resp'], log = accelerator.is_local_main_process)

####################
subprocess.run(["cp", "-r", "data/redial/.", "data/redial_gen/"], check=True)
subprocess.run(["python", "data/redial_gen/merge.py", "--gen_file_prefix", "dialogpt5e4"], check=True)
from_pred_output_resp_to_input()

#####################
kg = DBpedia(dataset= 'redial_gen', debug=False).get_entity_kg_info()
prompt_encoder = KGPrompt(
    model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
    n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases= 8,
    edge_index=kg['edge_index'], edge_type=kg['edge_type'],
    n_prefix_rec= 20
)


prompt_encoder.load('output_dir/rec/best')
prompt_encoder = prompt_encoder.to(device)

fix_modules = [model, text_encoder]
for module in fix_modules:
    module.requires_grad_(False)

# optim & amp
modules = [prompt_encoder]
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for model in modules for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.01,
    },
    {
        "params": [p for model in modules for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr= 1e-3)
# data
train_dataset = CRSRecDataset(
    dataset= 'redial_gen', split='train', debug= False,
    tokenizer=tokenizer, context_max_length= 128, use_resp= False,
    prompt_tokenizer=text_tokenizer, prompt_max_length= 128,
    entity_max_length= 32,
)
shot_len = int(len(train_dataset) * 1)
train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
assert len(train_dataset) == shot_len
valid_dataset = CRSRecDataset(
    dataset= 'redial_gen', split='valid', debug= False,
    tokenizer=tokenizer, context_max_length= 128, use_resp= False,
    prompt_tokenizer=text_tokenizer, prompt_max_length= 128,
    entity_max_length= 32,
)
test_dataset = CRSRecDataset(
    dataset= 'redial_gen', split='sample_input', debug= False,
    tokenizer=tokenizer, context_max_length= 128, use_resp= False,
    prompt_tokenizer=text_tokenizer, prompt_max_length= 128,
    entity_max_length= 32,
)
data_collator = CRSRecDataCollator(
    tokenizer=tokenizer, device=device, debug= False,
    context_max_length= 128, entity_max_length= 32,
    pad_entity_id=kg['pad_entity_id'],
    prompt_tokenizer=text_tokenizer, prompt_max_length= 128,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size= 64,
    collate_fn=data_collator,
    shuffle=True
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=64,
    collate_fn=data_collator,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    collate_fn=data_collator,
)
evaluator = RecEvaluator()
prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
)
# step, epoch, batch size
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / 1)
if max_train_steps is None:
    max_train_steps = 4 * num_update_steps_per_epoch
else:
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

total_batch_size = 32 * accelerator.num_processes * 1
completed_steps = 0
# lr_scheduler
lr_scheduler = get_linear_schedule_with_warmup(optimizer, 530, max_train_steps)
lr_scheduler = accelerator.prepare(lr_scheduler)
# training info
logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {32}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {1}")
logger.info(f"  Total optimization steps = {max_train_steps}")
# Only show the progress bar once on each machine.
# progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Training..")

# save model with best metric
metric, mode = 'loss', -1
assert mode in (-1, 1)
if mode == 1:
    best_metric = 0
else:
    best_metric = float('inf')
best_metric_dir = os.path.join('outputdir/rec', 'best')
os.makedirs(best_metric_dir, exist_ok=True)


# test
test_loss = []
total_list = []
prompt_encoder.eval()
for batch in tqdm(test_dataloader, desc='Testing..'):
    with torch.no_grad():
        token_embeds = text_encoder(**batch['prompt']).last_hidden_state
        prompt_embeds = prompt_encoder(
            entity_ids=batch['entity'],
            token_embeds=token_embeds,
            output_entity=True,
            use_rec_prefix=True
        )
        batch['context']['prompt_embeds'] = prompt_embeds
        # batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
        batch['context']['entity_embeds'] = accelerator.unwrap_model(prompt_encoder).get_entity_embeds()

        outputs = model(**batch['context'], rec=True)
        test_loss.append(float(outputs.rec_loss))
        logits = outputs.rec_logits[:, kg['item_ids']]
        ranks = torch.topk(logits, k=50, dim=-1).indices.tolist() #?
        ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
        labels = batch['context']['rec_labels']
        total_list.extend(
            [
                {
                    'gt': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
                    'recommendation': ranks[i].tolist() if isinstance(ranks[i], torch.Tensor) else ranks[i]
                }
            ]
            for i in range(len(labels))
        )
        evaluator.evaluate(ranks, labels)

