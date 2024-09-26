from src.generator.conv import load_conversation_model, generate_conversation
from src.generator.rec import load_recommendation_model, generate_recommendation
from accelerate import Accelerator
import json
import random

# Khởi tạo Accelerator
accelerator = Accelerator(device_placement=False, mixed_precision='fp16')
device = accelerator.device

# Đường dẫn cho các model và tokenizer
CONV_MODEL_PATH = "/home/thiendc/InferConverRec/save/redial/gpt2_conv/checkpoint-5000"
CONV_TOKENIZER_PATH = "/home/thiendc/InferConverRec/save/redial/gpt2_conv"
REC_MODEL_PATH = "/home/thiendc/InferConverRec/save/redial/gpt2_rec/checkpoint-5000"
REC_TOKENIZER_PATH = "/home/thiendc/InferConverRec/save/redial/gpt2_rec"
TEXT_ENCODER_PATH = "/home/thiendc/InferConverRec/save/redial/bert"
CONV_PROMPT_ENCODER_PATH = "/home/thiendc/InferConverRec/save/redial/prompt_conv/checkpoint-5000/prompt_encoder.pth"
REC_PROMPT_ENCODER_PATH = "/home/thiendc/InferConverRec/save/redial/prompt_rec/checkpoint-5000/prompt_encoder.pth"

# Load models
conv_model, conv_tokenizer, conv_text_encoder, conv_prompt_encoder = load_conversation_model(
    CONV_MODEL_PATH, CONV_TOKENIZER_PATH, TEXT_ENCODER_PATH, CONV_PROMPT_ENCODER_PATH
)
rec_model, rec_tokenizer, rec_text_encoder, rec_prompt_encoder = load_recommendation_model(
    REC_MODEL_PATH, REC_TOKENIZER_PATH, TEXT_ENCODER_PATH, REC_PROMPT_ENCODER_PATH
)

# Prepare models with Accelerator
conv_model, conv_text_encoder, conv_prompt_encoder = accelerator.prepare(
    conv_model, conv_text_encoder, conv_prompt_encoder
)
rec_model, rec_text_encoder, rec_prompt_encoder = accelerator.prepare(
    rec_model, rec_text_encoder, rec_prompt_encoder
)

# Các biến global cần thiết
apologise_error_404 = [
    "Sorry, I don't seem to know the person or movie you are talking about!",
    "Ah, my circuits seem to have been fried. I can't find the movie / person you are referring to!",
    "Sorry, error 404, that movie / person has not been found.",
    "I apologise, but can you please check your spelling. I can't comprehend what you are saying!",
]

def chat(input_str):
    single_user_path = '/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl'
    doesEntityExist = input_to_jsonl(input_str)
    
    if doesEntityExist:
        # Đọc dữ liệu từ JSON
        with open(single_user_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Generate conversation
        conversation = generate_conversation(
            input_str,
            conv_model, conv_tokenizer, conv_text_encoder, conv_prompt_encoder,
            accelerator
        )
        
        # Update JSON with conversation
        data['resp'] = conversation
        
        # Generate recommendation
        recommendation = generate_recommendation(
            input_str,
            rec_model, rec_tokenizer, rec_text_encoder, rec_prompt_encoder,
            accelerator
        )
        
        # Update JSON with recommendation
        data['rec'] = recommendation[:5]  # Lấy top 5 đề xuất
        
        # Process response and recommendation
        response = data['resp']
        recommended_movies = data['rec']

        # Replace <movie> in response with recommended movie names
        for movie_id in recommended_movies:
            if '<movie>' in response:
                movie_name = id_entity.get(str(movie_id), "Unknown movie")
                response = response.replace('<movie>', movie_name, 1)
        
        # Cập nhật JSON file
        with open(single_user_path, 'w') as json_file:
            json.dump(data, json_file)

        return response
    else:
        return random.choice(apologise_error_404)

# Các hàm phụ trợ khác (như input_to_jsonl, get_entities_and_ids, etc.) giữ nguyên