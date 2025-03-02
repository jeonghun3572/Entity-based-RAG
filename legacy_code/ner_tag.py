import json
import pdb
import re
import spacy
from transformers import AutoTokenizer

from itertools import permutations

def generate_all_permutations(input_string):
    words = input_string.split()
    permutations_list = list(permutations(words))
    result_strings = [' '.join(permutation) for permutation in permutations_list]
    return result_strings


def subfinder(mylist, pattern, first_only=False):
    matches_indx = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches_indx.append((i, i+len(pattern)))
            if first_only:
                break
    return matches_indx


def update_labels(labels, matched_ranges):
    for range_i in matched_ranges:
        if labels[range_i[0]] == 0:
            labels[range_i[0]] = 1  # [B]eginning
            for i in range(range_i[0]+1, range_i[1]):
                labels[i] = 2  # [I]nside
    return labels


def update_bio_labels_below(labels, source, match_patterns, tokens, tokenizer, first_only=False):
    entity = []
    for pattern in match_patterns:
        flag = False
        if " " in pattern:
            all_pattern = generate_all_permutations(pattern)
            for pat in all_pattern:
                found = re.findall(re.escape(pat), source, re.IGNORECASE)

                for matched in found:
                    if matched[0] != ' ':
                        matched_space = ' ' + matched
                    matched_ids = tokenizer.encode(matched, add_special_tokens=False)
                    matched_ranges = subfinder(tokens, matched_ids, first_only)
                    matched_ids_space = tokenizer.encode(matched_space, add_special_tokens=False)
                    matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
                    labels = update_labels(labels, matched_ranges)
                    labels = update_labels(labels, matched_ranges_space)
                    flag = True
        else:
            found = re.findall(re.escape(pattern), source, re.IGNORECASE)
            
            for matched in found:
                if matched[0] != ' ':
                    matched_space = ' ' + matched
                matched_ids = tokenizer.encode(matched, add_special_tokens=False)
                matched_ranges = subfinder(tokens, matched_ids, first_only)
                matched_ids_space = tokenizer.encode(matched_space, add_special_tokens=False)
                matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
                labels = update_labels(labels, matched_ranges)
                labels = update_labels(labels, matched_ranges_space)
                flag = True

        if flag==True:
            entity.append(pattern)

    match_patterns = re.findall(r'<obj>(.*?)</obj>', source)
    match_patterns = list(set(match_patterns))
    for pattern in match_patterns:
        flag = False
        found = re.findall(rf'\b{re.escape(pattern)}\b', source, re.IGNORECASE)
        for matched in found:
            if matched[0] != ' ':
                matched_space = ' ' + matched
            matched_ids = tokenizer.encode(matched, add_special_tokens=False)
            matched_ranges = subfinder(tokens, matched_ids, first_only)
            matched_ids_space = tokenizer.encode(matched_space, add_special_tokens=False)
            matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
            labels = update_labels(labels, matched_ranges)
            labels = update_labels(labels, matched_ranges_space)
            flag = True
        
        if flag==True:
            entity.append(pattern)

    
    f_entity = []
    seen = set()
    for ent in entity:
        if ent.lower() not in seen:
            f_entity.append(ent)
            seen.add(ent.lower())

    return labels, f_entity


def main():
    with open('datasets/medquad-5_tag/medquad-val-sum.json') as f:
        json_data = json.load(f)

    spacy.require_gpu(0)
    nlp = spacy.load("en_core_sci_scibert")

    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-x-large')
    special_tokens_dict = {'additional_special_tokens': ['<obj>','</obj>','<ref>','</ref>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    total = []
    for data in json_data:
        question = data['question']
        passage = data['passage']
        summary = data['summary']
        entity = data['entity']

        tokens = tokenizer.encode(passage, max_length=4096, truncation=True)
        labels = [0] * len(tokens)

        labels, f_entity = update_bio_labels_below(labels, passage, t_entity, tokens, tokenizer, first_only=False)
        assert len(tokens) == len(labels)

        temp={}
        temp['passage'] = passage
        temp['summary'] = summary
        temp['ner_labels'] = labels
        total.append(temp)

    with open(f"medquad-val-tag-masking.jsonl", encoding="utf-8", mode="w") as f:
        for i in total:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
