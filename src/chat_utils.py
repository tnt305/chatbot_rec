%cd src
import random
import json
import os
import re
import subprocess
import multiprocessing
from collections import defaultdict
from accelerate import Accelerator
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from refinement import *
from generator.conv import GenerateConversation
from generator.rec import GenerateRecommendation
from generator.hooks import hook_sentences ,promotional_sentences, pick_verbs, question_type
from fuzzywuzzy import fuzz
multiprocessing.set_start_method('spawn', force=True)

def input2jsonl(input_str, entity2id_json):
    save_path = './data/redial/sample_input_data_processed.jsonl'
    is_exist = os.path.exists(save_path)

    current_str = ''

    if is_exist:
        with open(save_path, 'r') as json_file:
            current_str = json.load(json_file)
            current_str['context'].append(input_str)
    else:
        current_str = {"context": [input_str], "resp": "", "rec": [], "entity": []}
    
    entities = re.findall(r'\$.*?\$', input_str)
    print(entities)
    if entities:
        entities = [entity.replace('$', '').lower() for entity in entities]
        # list_entities = set(get_entities_and_ids(entity2id_json))

        matched_entities = cosine_similarity_entities(entities)
        print('Đây là các thực thể được lựa chọn', matched_entities)
        entity_id = get_entities_and_ids('./data/redial_gen/entity2id.json')
        if matched_entities:
            entity_ids = [entity_id[entity] for entity in matched_entities if entity in get_entities_and_ids(entity2id_json)]
            print("-------------------------------------------------------------------")
            print(f"Movies that you {random.choice(pick_verbs)} earlier: {', '.join(matched_entities)}")
            print(f"Corresponding IDs : {', '.join(map(str, entity_ids))}")
            # print("-------------------------------------------------------------------")
            current_str['rec'].extend(entity_ids)

            does_movie_exist = True
        else:
            does_movie_exist = False
    else:
        does_movie_exist = False
    
    with open(save_path, 'w') as outfile:
        jout = json.dumps(current_str)
        outfile.write(jout)
    
    return does_movie_exist

def get_entities_and_ids(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    entity_id = {}
    id_entity = {}
    for k, v in data.items():
        k = k.split("/")[-1]
        k = k.replace(">", "")
        # if "(" and ")" in k:
        #     k=re.sub("[\(\[].*?[\)\]]", "", k)
        k = k.replace("_", " ")
        k = k.strip()
        # k = k.lower()

        entity_id[k] = v
        id_entity[v] = k
    return entity_id

def cosine_similarity_entities(entities):
    matched_entities = []
    entity_names = list(get_entities_and_ids('./data/redial_gen/entity2id.json').keys())
    
    # Tạo vector TF-IDF cho tất cả thực thể trong entity_names và entities
    vectorizer = TfidfVectorizer().fit(entity_names + entities)
    
    # Vector hóa danh sách entity
    entity_vectors = vectorizer.transform(entities)
    entity_name_vectors = vectorizer.transform(entity_names)

    # Duyệt qua từng thực thể
    for entity_vector in entity_vectors:
        # Tính cosine similarity giữa entity hiện tại và tất cả entity_names
        similarities = cosine_similarity(entity_vector, entity_name_vectors).flatten()

        # Sắp xếp theo độ tương đồng giảm dần
        sorted_similarities = sorted(zip(entity_names, similarities), key=lambda x: x[1], reverse=True)

        # Lấy top 10 kết quả có độ tương đồng > 0.5 (50%)
        top_matches = [entity_name for entity_name, similarity in sorted_similarities[:10] if similarity > 0.6]
        matched_entities.extend(top_matches)

    return matched_entities


def chat(input_str):
    input_str =  input()
    global global_conv_module
    doesEntityExist = input2jsonl(input_str, entity2id_json = './data/redial_gen/entity2id.json')    
    
    print("***************")
    print("Does entity exist:", doesEntityExist)
    print("***************")
    
    if doesEntityExist:
        # Initialize conv_module if it hasn't been initialized yet
        initialize_conv_module()
        # Use the global conv_module
        predict_conversation = global_conv_module.generate_conversations()
        print('Đây là conversation', predict_conversation) # list
    
        save_path = './data/redial/sample_input_data_processed.jsonl'
        try:
            with open(save_path, 'r') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError:
            # If the file is empty or invalid JSON, create a new dictionary
            data = {"context": [], "resp": "", "rec": [], "entity": []}
    
        if isinstance(predict_conversation, list):
            predict_conversation = ' '.join(predict_conversation)
    
        predict_conversation = rewrite2(predict_conversation)
        predict_conversation = predict_conversation.replace("<|endoftext|>", "")
        predict_conversation = predict_conversation.replace("System: ", "")
        
        if not any(q in predict_conversation for q in question_type):
            if any(word in ['yes', 'no'] for word in predict_conversation.split(".")):
                predict_conversation = ".".join([i for i in predict_conversation.split(".") if 'yes' not in i and 'no' not in i])
    
        if predict_conversation.strip() == "":
            predict_conversation = random.choice(hook_sentences)
        
        if is_valid_sentence(predict_conversation):
            predict_conversation = rewrite(predict_conversation)
    
        data['resp'] = predict_conversation
    
        with open(save_path, 'w') as json_file:
            json.dump(data, json_file)
    
        
        subprocess.run([
            "accelerate", "launch", 
            "--num_processes", "2",  # Chính xác 4 processes cho 4 GPU
            "infer_rec.py",
            "--dataset", "redial_gen",
            "--tokenizer", "/home/thiendc/InferConverRec/src/utils/dialogpt",
            "--model", "/home/thiendc/InferConverRec/src/utils/dialogpt_model",
            "--text_tokenizer", "/home/thiendc/InferConverRec/src/utils/roberta",
            "--text_encoder", "/home/thiendc/InferConverRec/src/utils/roberta_model",
            "--n_prefix_rec", "20",
            "--prompt_encoder", "/home/thiendc/InferConverRec/src/output_dir/rec1/best",
            "--per_device_eval_batch_size", "16",
            "--gradient_accumulation_steps", "2",
            "--context_max_length", "128",
            "--prompt_max_length", "128",
            "--entity_max_length", "32",
        ], check=True)
    
        with open('./data/redial_gen/movie_ids.json', 'r') as moviesl:
            list_movies = json.load(moviesl)
    
        with open('./recommedations.json', 'r') as f:
            movie_rec = json.load(f)
    
        recommendation_counts = defaultdict(int)
    
        # Duyệt qua từng item trong movie_rec và đếm số lần xuất hiện của mỗi recommendation
        for item in movie_rec:
            recommendations = item[0]['recommendation']
            for rec in recommendations:
                recommendation_counts[rec] += 1
    
        # Sắp xếp các recommendation dựa trên số lần xuất hiện (tần suất) và tạo list unique gt
        sorted_recommendations = sorted(recommendation_counts.keys(), key=lambda x: recommendation_counts[x], reverse=True)
        # Nhóm các unique 'gt' vào list
        unique_gt = list({item[0]['gt'] for item in movie_rec})
    
        # In kết quả
        print("Unique GTs:", unique_gt)
        print("Ranked Recommendations:", sorted_recommendations)
    
        if predict_conversation.count("<movie>") >= 1:
            data['rec'] = sorted_recommendations[:predict_conversation.count("<movie>")]
        else:
            data['rec'] = sorted_recommendations[:3]
    
        response = data['resp']
        recommended_movies = data['rec']
    
        if "<movie>" in response:
            for movie_id in recommended_movies:
                movie_name = {v:k for k, v in get_entities_and_ids('./data/redial/entity2id.json').items()}.get(movie_id, str(movie_id))
                response = response.replace("<movie>", movie_name, 1)
        else:
            movie_mask = ', '.join(['<movie>'] * len(data['rec']))
            response = f"{response} {random.choice(promotional_sentences)} {movie_mask}"
        
        print('response', response)
    else:
        response = random.choice(apologise_error_404)
        with open('/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl', 'r') as file:
            line = file.readline().strip()
        data = json.loads(line)
        data['context'].append(response)
        with open('/home/thiendc/InferConverRec/src/data/redial/sample_input_data_processed.jsonl', 'w') as file:
            json.dump(data, file)
        # clear_cached()
        return response


