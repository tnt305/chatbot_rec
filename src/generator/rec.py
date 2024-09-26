import sys
import os
import json
from collections import OrderedDict
# Thêm đường dẫn của thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoTokenizer, AutoModel
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from dataset_dbpedia import DBpedia
from evaluate_rec import RecEvaluator
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# def load_recommendation_model(model_path, tokenizer_path, text_tokenizer, text_encoder_path, prompt_encoder_path):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#     tokenizer.add_special_tokens(gpt2_special_tokens_dict)
#     # Load model
#     model = PromptGPT2forCRS.from_pretrained(model_path)
#     model.resize_token_embeddings(len(tokenizer))
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model = model.to(device)

#     # Load text encoder
#     text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
#     text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
#     text_encoder = AutoModel.from_pretrained(text_encoder_path)
#     text_encoder.resize_token_embeddings(len(text_tokenizer))
#     text_encoder = text_encoder.to(device)

#     # Load KG
#     kg = DBpedia(dataset="redial_gen").get_entity_kg_info()

#     # Load prompt encoder
#     prompt_encoder = KGPrompt(
#         model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
#         n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=8,
#         edge_index=kg['edge_index'], edge_type=kg['edge_type'],
#         n_prefix_rec= 20
#     )
#     prompt_encoder.load(prompt_encoder_path)
#     prompt_encoder = prompt_encoder.to(device)
#     # prompt_encoder = accelerator.prepare(prompt_encoder)

#     return model, tokenizer, text_tokenizer, text_encoder, prompt_encoder

# def generate_recommendation(model, tokenizer, text_tokenizer,  text_encoder, prompt_encoder, accelerator):
#     # Thiết lập các tham số
#     ## khởi tạo cùng chat_utils
#     # accelerator = Accelerator(device_placement=False, mixed_precision = 'fp16')
#     # device = accelerator.device

#     args = {
#         "dataset": "redial_gen",
#         "split": "sample_input",
#         "num_workers": 4,
#         "context_max_length": 128,
#         "entity_max_length": 32,
#         "prompt_max_length": 128,
#         "n_prefix_rec": 20,
#         "num_bases": 8,
#         "per_device_eval_batch_size": 64,
#     }

#     # Tải KG
#     kg = DBpedia(dataset=args["dataset"]).get_entity_kg_info()

#     # Chuẩn bị dataset và dataloader
#     dataset = CRSRecDataset(
#         args["dataset"], 
#         args["split"], 
#         tokenizer = tokenizer,
#         context_max_length=args["context_max_length"],
#         prompt_tokenizer = text_tokenizer, 
#         prompt_max_length=args["prompt_max_length"],
#         entity_max_length= args["entity_max_length"],
#     )
#     data_collator = CRSRecDataCollator(
#         tokenizer=tokenizer, 
#         device=accelerator.device,
#         context_max_length=args["context_max_length"],
#         entity_max_length=args["entity_max_length"],
#         pad_entity_id=kg['pad_entity_id'],
#         prompt_tokenizer= text_tokenizer, 
#         prompt_max_length=args["prompt_max_length"]
#     )
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args["per_device_eval_batch_size"],
#         num_workers=args["num_workers"],
#         collate_fn=data_collator,
#     )

#     # Chuẩn bị model và dataloader với Accelerator
#     prompt_encoder = accelerator.prepare(prompt_encoder)

#     # Sinh recommend
#     # model.eval()
#     prompt_encoder.eval()
#     # text_encoder.eval()

#     recommendations = []
#     with torch.no_grad():
#         for batch in tqdm(dataloader):
#             token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#             prompt_embeds = prompt_encoder(
#                 entity_ids=batch['entity'],
#                 token_embeds=token_embeds,
#                 output_entity=True,
#                 use_rec_prefix=True
#             )
#             batch['context']['prompt_embeds'] = prompt_embeds
#             batch['context']['entity_embeds'] = accelerator.unwrap_model(prompt_encoder).get_entity_embeds()

#             outputs = model(**batch['context'], rec=True)
#             logits = outputs.rec_logits[:, kg['item_ids']]
#             ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
#             ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
#             labels = batch['context']['rec_labels']

#             recommendations.extend(
#                 [
#                     {
#                         'gt': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
#                         'recommendation': ranks[i]
#                     }
#                     for i in range(len(labels))
#                 ]
#             )

#             # evaluator.evaluate(ranks, labels)
#     unique_recommendations = list(OrderedDict((json.dumps(item, sort_keys=True), item) for item in recommendations).values())

#     # Lưu các recommendation duy nhất vào file JSON
#     output_file = 'recommendations.json'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(unique_recommendations, f, ensure_ascii=False, indent=2)

#     with open('/home/thiendc/InferConverRec/src/recommendations.json', 'r') as f:
#         rec = json.load(f)
    
#     return rec[-1]  # Trả về recommend đầu tiên và metrics
from torch.nn import DataParallel
from torch.cuda.amp import autocast

class GenerateRecommendation():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(device_placement=False, mixed_precision='fp16')
        self.kg = DBpedia(dataset= 'redial', debug= False).get_entity_kg_info()

        # Initialize tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained('utils/dialogpt')
        self.tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        self.text_tokenizer = AutoTokenizer.from_pretrained('utils/roberta')
        self.text_tokenizer.add_special_tokens(prompt_special_tokens_dict)

        # Initialize models
        self.model = PromptGPT2forCRS.from_pretrained('utils/dialogpt_model')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(self.device)
        # self.model = DataParallel(self.model).to(self.device)

        self.text_encoder = AutoModel.from_pretrained('utils/roberta_model')
        self.text_encoder.resize_token_embeddings(len(self.text_tokenizer))
        self.text_encoder = self.text_encoder.to(self.device)
        # self.text_encoder = DataParallel(self.text_encoder).to(self.device)

        # Initialize prompt encoder
        self.prompt_encoder = KGPrompt(
            self.model.config.n_embd, self.text_encoder.config.hidden_size,
            self.model.config.n_head, self.model.config.n_layer, 2,
            n_entity= self.kg['num_entities'], num_relations=self.kg['num_relations'],
            num_bases= 8, edge_index=self.kg['edge_index'],
            edge_type=self.kg['edge_type'], n_prefix_rec= 20
        )
       
        self.prompt_encoder.load('/home/thiendc/InferConverRec/src/output_dir/rec1/best')
        self.prompt_encoder = self.prompt_encoder.to(self.device)
        # self.prompt_encoder = DataParallel(self.prompt_encoder).to(self.device)

        # Initialize data collator
        self.data_collator = CRSRecDataCollator(
            tokenizer=self.tokenizer, device= self.device, debug= False,
            context_max_length= 128,
            entity_max_length= 32,
            pad_entity_id= self.kg['pad_entity_id'],
            prompt_tokenizer=self.text_tokenizer,
            prompt_max_length= 200,
        )

        # Initialize evaluator
        self.evaluator = RecEvaluator()

    def prepare_dataset(self, dataset = 'redial_gen', split='test'):
        return CRSRecDataset(
            dataset= dataset, split=split, debug= False,
            tokenizer=self.tokenizer, context_max_length= 128,
            use_resp= False, prompt_tokenizer=self.text_tokenizer,
            prompt_max_length= 200,
            entity_max_length= 32,
        )

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size = 64,
            collate_fn=self.data_collator,
        )

    def generate_recommendation(self):
        dataset = self.prepare_dataset(dataset = 'redial_gen', split='sample_input')
        dataloader = self.get_dataloader(dataset)

        self.prompt_encoder.eval()
        # self.model.eval()
        # self.text_encoder.eval()

        total_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Generating recommendations...'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with autocast():
                    token_embeds = self.text_encoder(**batch['prompt']).last_hidden_state
                    prompt_embeds = self.prompt_encoder(
                        entity_ids=batch['entity'],
                        token_embeds=token_embeds,
                        output_entity=True,
                        use_rec_prefix=True
                    )
                    batch['context']['prompt_embeds'] = prompt_embeds
                    batch['context']['entity_embeds'] = self.prompt_encoder.module.get_entity_embeds()

                    outputs = self.model(**batch['context'], rec=True)
                    logits = outputs.rec_logits[:, self.kg['item_ids']]
                    ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                    ranks = [[self.kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                    labels = batch['context']['rec_labels']
                
                total_list.extend(
                    [
                        {
                            'gt': labels[i].tolist() if isinstance(labels[i], torch.Tensor) else labels[i],
                            'recommendation': ranks[i]
                        }
                        for i in range(len(labels))
                    ]
                )

                # Clear cache
                torch.cuda.empty_cache()
        return total_list

    def evaluate(self, recommendations, labels):
        return self.evaluator.evaluate(recommendations, labels)

    def save_results(self, results, filename='results.json'):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
