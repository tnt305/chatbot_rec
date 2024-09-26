import numpy as np
import json
import os
import re
import gradio 
import random
from src.chat_utils import *

from src.generator.conv import GenerateConversation

os.environ['MKL_THREADING_LAYER']= 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

apologise_error_404 = [
    "Sorry, I couldn't find any information on that.",
    "I apologize, but I don't have details on that movie.",
    "Unfortunately, I couldn't locate that entity."
]

# conv_module =  GenerateConversation()
# # rec_module = GenerateRecommendation()
# def run_inference():
#     # subprocess.run([
#     #     "python", "infer_conv.py",
#     #     "--dataset", "redial",
#     #     "--split", "sample_input",
#     #     "--tokenizer", "utils/dialogpt",
#     #     "--model", "utils/dialogpt_model",
#     #     "--text_tokenizer", "utils/roberta",
#     #     "--text_encoder", "utils/roberta_model",
#     #     "--n_prefix_conv", "20",
#     #     "--prompt_encoder", "/home/thiendc/InferConverRec/src/output_dir/conv/dialogpt1e3_2nd/best",
#     #     "--per_device_eval_batch_size", "128",
#     #     "--context_max_length", "200",
#     #     "--resp_max_length", "183",
#     #     "--prompt_max_length", "128",
#     #     "--entity_max_length", "32"
#     # ], check=True)

#     predict_conversation = conv_module.generate_conversations()
#     print('Đây là conversation', predict_conversation)
#     # Copy files
#     subprocess.run(["cp", "-r", "data/redial/.", "data/redial_gen/"], check=True)
    
#     # # Run merge.py
#     # subprocess.run(["python", "data/redial_gen/merge.py", "--gen_file_prefix", "dialogpt5e4"], check=True)
    
#     from_pred_output_resp_to_input()
#     # rec_module.generate_recommendation()
#     subprocess.run([
#         "accelerate", "launch", "--num_processes", "2",
#          "infer_rec.py",
#         "--dataset", "redial_gen",
#         "--tokenizer", "microsoft/DialoGPT-small",
#         "--model", "microsoft/DialoGPT-small",
#         "--text_tokenizer", "roberta-base",
#         "--text_encoder", "roberta-base",
#         "--n_prefix_rec", "20",
#         "--prompt_encoder", "./output_dir/rec1/best",
#         "--per_device_eval_batch_size", "64",
#         "--gradient_accumulation_steps", "1",
#         "--num_warmup_steps", "530",
#         "--context_max_length", "128",
#         "--prompt_max_length", "200",
#         "--entity_max_length", "32",
#     ], check=True)


# def chat(input_str):
#     doesEntityExist = input_to_jsonl(input_str)
#     print("***************")
#     print("Does entity exist:", doesEntityExist)
#     print("***************")
#     if doesEntityExist:
#         run_inference()
#         return read_from_jsonl_to_user()
#     else:
#         return random.choice(apologise_error_404)

def chatbot(user_input, chat_history):
    try:
        response = chat(user_input)
        chat_history.append((user_input, response))
        return response
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        chat_history.append((user_input, error_message))
        return error_message

demo = gradio.ChatInterface(
    fn= chatbot,
    title="Movie Recommendation Chatbot",
    description="Chat about movies and get recommendations. Mention movies by enclosing them in dollar signs, like $Movie Title$",
    examples=[  "Hey there! I'm looking for something similar to $Zombieland (2009)$ Loved that movie", 
                "Do you have any movies similar to $The avengers$?", 
                "Can I check with you if you have any movies that has $Brad Pitt$ like $Fight Club$ ?",
                "Hi I like funny movies like $Airplane! (1980)$"
                "have you seen $Fried Green Tomatoes (1991)$ or something more recent would be Water for $Elephants (2011)$"],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear Chat",
)

if __name__ == "__main__":
    demo.launch(share = True)
    
