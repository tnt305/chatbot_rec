# import json
# from collections import defaultdict
# import pickle as pkl
# from tqdm.auto import tqdm
# # import os
# # os.chdir("data/redial")
# # current_dir = "/kaggle/working/InferConverRec/data/redial"

# def get_item_set(file):
#     entity = set()
#     with open(file, 'r', encoding='utf-8') as f:
#         for line in tqdm(f):
#             line = json.loads(line)
#             for message in line['messages']:
#                 for e in message['movie']:
#                     entity.add(e)
#     return entity


# def load_kg(path):
#     print('load kg')
#     kg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in tqdm(f):
#             tuples = line.strip().split()
#             if tuples and len(tuples) == 4 and tuples[-1] == ".":
#                 h, r, t = tuples[:3]
#                 kg[h].append((r, t))
#     return kg


# def extract_subkg(kg, seed_set, n_hop):
#     """extract subkg from seed_set by n_hop

#     Args:
#         kg (dict): {head entity: [(relation, tail entity)]}
#         seed_set (list or set): [entity]
#         n_hop (int):

#     Returns:
#         subkg (dict): {head entity: [(relation, tail entity)]}, extended by n_hop
#     """
#     print('extract subkg')

#     subkg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
#     subkg_hrt = set()  # {(head_entity, relation, tail_entity)}

#     ripple_set = None
#     for hop in range(n_hop):
#         memories_h = set()  # [head_entity]
#         memories_r = set()  # [relation]
#         memories_t = set()  # [tail_entity]

#         if hop == 0:
#             tails_of_last_hop = seed_set  # [entity]
#         else:
#             tails_of_last_hop = ripple_set[2]  # [tail_entity]

#         for entity in tqdm(tails_of_last_hop):
#             for relation_and_tail in kg[entity]:
#                 h, r, t = entity, relation_and_tail[0], relation_and_tail[1]
#                 if (h, r, t) not in subkg_hrt:
#                     subkg_hrt.add((h, r, t))
#                     subkg[h].append((r, t))
#                 memories_h.add(h)
#                 memories_r.add(r)
#                 memories_t.add(t)

#         ripple_set = (memories_h, memories_r, memories_t)

#     return subkg


# def kg2id(kg):
#     entity_set = all_entity
#     with open(f'./relation_set.json', encoding='utf-8') as f:
#         relation_set = json.load(f)

#     for head, relation_tails in tqdm(kg.items()):
#         for relation_tail in relation_tails:
#             if relation_tail[0] in relation_set:
#                 entity_set.add(head)
#                 entity_set.add(relation_tail[1])

#     entity2id = {e: i for i, e in enumerate(entity_set)}
#     print(f"# entity: {len(entity2id)}")
#     relation2id = {r: i for i, r in enumerate(relation_set)}
#     relation2id['self_loop'] = len(relation2id)
#     print(f"# relation: {len(relation2id)}")

#     kg_idx = {}
#     for head, relation_tails in kg.items():
#         if head in entity2id:
#             head = entity2id[head]
#             kg_idx[head] = [(relation2id['self_loop'], head)]
#             for relation_tail in relation_tails:
#                 if relation_tail[0] in relation2id and relation_tail[1] in entity2id:
#                     kg_idx[head].append((relation2id[relation_tail[0]], entity2id[relation_tail[1]]))

#     return entity2id, relation2id, kg_idx


# all_entity = set()
# file_list = [
#     f'redial/test_data_dbpedia_raw.jsonl',
#     f'redial/valid_data_dbpedia_raw.jsonl',
#     f'redial/train_data_dbpedia_raw.jsonl',
# ]
# for file in file_list:
#     all_entity |= get_item_set(file)
# print(f'# all entity: {len(all_entity)}')

# with open('dbpedia/kg.pkl', 'rb') as f:
#     kg = pkl.load(f)
# subkg = extract_subkg(kg, all_entity, 2)
# entity2id, relation2id, subkg = kg2id(subkg)

# with open(f'./dbpedia_subkg.json', 'w', encoding='utf-8') as f:
#     json.dump(subkg, f, ensure_ascii=False)
# with open(f'./entity2id.json', 'w', encoding='utf-8') as f:
#     json.dump(entity2id, f, ensure_ascii=False)
# with open(f'./relation2id.json', 'w', encoding='utf-8') as f:
#     json.dump(relation2id, f, ensure_ascii=False)

# # relation > 500: edge: 172644, #relation: 45, #entity: 50593
# # relation > 1000: edge: 139854, #relation: 22, #entity: 44708
import json
import random
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm
import os
import argparse

os.chdir("./redial")


def get_item_set(file):
    entity = set()
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for message in line['messages']:
                for e in message['movie']:
                    entity.add(e)
    return entity


def load_kg(path):
    print('load kg')
    kg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            tuples = line.strip().split()
            if tuples and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                kg[h].append((r, t))
    return kg


def extract_subkg(kg, seed_set, n_hop):
    """extract subkg from seed_set by n_hop

    Args:
        kg (dict): {head entity: [(relation, tail entity)]}
        seed_set (list or set): [entity]
        n_hop (int):

    Returns:
        subkg (dict): {head entity: [(relation, tail entity)]}, extended by n_hop
    """
    print('extract subkg')

    subkg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
    subkg_hrt = set()  # {(head_entity, relation, tail_entity)}

    ripple_set = None
    for hop in range(n_hop):
        memories_h = set()  # [head_entity]
        memories_r = set()  # [relation]
        memories_t = set()  # [tail_entity]

        if hop == 0:
            tails_of_last_hop = seed_set  # [entity]
        else:
            tails_of_last_hop = ripple_set[2]  # [tail_entity]

        for entity in tqdm(tails_of_last_hop):
            for relation_and_tail in kg[entity]:
                h, r, t = entity, relation_and_tail[0], relation_and_tail[1]
                if (h, r, t) not in subkg_hrt:
                    subkg_hrt.add((h, r, t))
                    subkg[h].append((r, t))
                memories_h.add(h)
                memories_r.add(r)
                memories_t.add(t)

        ripple_set = (memories_h, memories_r, memories_t)

    return subkg


def kg2id(kg):
    entity_set = all_entity
    with open('relation_set.json', encoding='utf-8') as f:
        relation_set = json.load(f)

    for head, relation_tails in tqdm(kg.items()):
        for relation_tail in relation_tails:
            if relation_tail[0] in relation_set:
                entity_set.add(head)
                entity_set.add(relation_tail[1])

    entity2id = {e: i for i, e in enumerate(entity_set)}
    print(f"# entity: {len(entity2id)}")
    relation2id = {r: i for i, r in enumerate(relation_set)}
    relation2id['self_loop'] = len(relation2id)
    print(f"# relation: {len(relation2id)}")

    kg_idx = {}
    for head, relation_tails in kg.items():
        if head in entity2id:
            head = entity2id[head]
            kg_idx[head] = [(relation2id['self_loop'], head)]
            for relation_tail in relation_tails:
                if relation_tail[0] in relation2id and relation_tail[1] in entity2id:
                    kg_idx[head].append((relation2id[relation_tail[0]], entity2id[relation_tail[1]]))

    return entity2id, relation2id, kg_idx


def random_drop(sub_kg, drop_rate):
    edges = list()
    for item in sub_kg.items():
        head = item[0]
        for body in item[1]:
            edge = list()
            edge.append(head)
            edge.append(body[0])
            edge.append(body[1])
            edges.append(edge)

    random.seed(42)
    sub_kg = random.choices(edges, k=int(len(edges) * drop_rate))
    sub_kg_dict = defaultdict(list)
    for edge in sub_kg:
        sub_kg_dict[edge[0]].append(edge[1:])
    return sub_kg_dict


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--hop', type=int, required=True)
    parser.add_argument('--drop', type=float, default=1.0)
    args, _ = parser.parse_known_args()
    print(f"Extract {args.hop}-hop subkg. Drop rate: {args.drop} ({type(args.drop)}).")

    all_entity = set()
    file_list = [
        'test_data_dbpedia_raw.jsonl',
        'valid_data_dbpedia_raw.jsonl',
        'train_data_dbpedia_raw.jsonl',
    ]
    for file in file_list:
        all_entity |= get_item_set(file)
    print(f'# all entity: {len(all_entity)}')

    with open('../dbpedia/kg.pkl', 'rb') as f:
        kg = pkl.load(f)
    subkg = extract_subkg(kg, all_entity, args.hop)  # NOTE: n-hop parameter

    # One-hop random drop
    if args.drop < 1.0:
        subkg = random_drop(subkg, args.drop)

    entity2id, relation2id, subkg = kg2id(subkg)

    with open('dbpedia_subkg.json', 'w', encoding='utf-8') as f:
        json.dump(subkg, f, ensure_ascii=False)
    with open('entity2id.json', 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, ensure_ascii=False)
    with open('relation2id.json', 'w', encoding='utf-8') as f:
        json.dump(relation2id, f, ensure_ascii=False)