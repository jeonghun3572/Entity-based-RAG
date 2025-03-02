import json
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
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

def update_bio_labels(labels, source, match_patterns, tokens, encoder, first_only=False):
    entity = []
    for pattern in match_patterns:
        pattern = pattern.strip()
        flag = False
        if " " in pattern:
            all_pattern = generate_all_permutations(pattern)
            for pat in all_pattern:
                found = re.findall(re.escape(pat), source, re.IGNORECASE)
                for matched in found:
                    if matched[0] != ' ':
                        matched_space = ' ' + matched
                    matched_ids = encoder.encode(matched, add_special_tokens=False)
                    matched_ranges = subfinder(tokens, matched_ids, first_only)
                    matched_ids_space = encoder.encode(matched_space, add_special_tokens=False)
                    matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
                    labels = update_labels(labels, matched_ranges)
                    labels = update_labels(labels, matched_ranges_space)
                    flag = True
        else:
            found = re.findall(re.escape(pattern), source, re.IGNORECASE)
            for matched in found:
                if matched[0] != ' ':
                    matched_space = ' ' + matched
                matched_ids = encoder.encode(matched, add_special_tokens=False)
                matched_ranges = subfinder(tokens, matched_ids, first_only)
                matched_ids_space = encoder.encode(matched_space, add_special_tokens=False)
                matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
                labels = update_labels(labels, matched_ranges)
                labels = update_labels(labels, matched_ranges_space)
                flag = True

        if flag:
            entity.append(pattern)


    # match_patterns = re.findall(r'<ent>(.*?)</ent>', source)
    # match_patterns = list(set(match_patterns))
    # for pattern in match_patterns:
    #     flag = False
    #     found = re.findall(rf'\b{re.escape(pattern)}\b', source, re.IGNORECASE)
    #     for matched in found:
    #         if matched[0] != ' ':
    #             matched_space = ' ' + matched
    #         matched_ids = encoder.encode(matched, add_special_tokens=False)
    #         matched_ranges = subfinder(tokens, matched_ids, first_only)
    #         matched_ids_space = encoder.encode(matched_space, add_special_tokens=False)
    #         matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
    #         labels = update_labels(labels, matched_ranges)
    #         labels = update_labels(labels, matched_ranges_space)
    #         flag = True
        
    #     if flag==True:
    #         entity.append(pattern)

    f_entity = []
    seen = set()
    for ent in entity:
        if ent.lower() not in seen:
            f_entity.append(ent)
            seen.add(ent.lower())

    return labels, f_entity

def entity_match(ent, source):
    ent_split = ent.split()
    result = []
    for l in range(len(ent_split), 1, -1):
        for start_i in range(len(ent_split) - l + 1):
            sub_ent = " ".join(ent_split[start_i:start_i+l])
            if re.search(re.escape(sub_ent), source, re.IGNORECASE):
                result.append(sub_ent)
        if result:
            break
    if result:
        return result
    else:
        for token in ent_split:
            if token.lower() not in STOP_WORDS or token == "US":
                if re.search(re.escape(token), source, re.IGNORECASE):
                    result.append(token)
        return result

def fix_missing_period(line):
    dm_single_close_quote = u'\u2019'
    dm_double_close_quote = u'\u201d'
    END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]
    if "@highlight" in line or line == "" or line[-1] in END_TOKENS:
        return line
    return line + "."

def main():
    with open('./val_medquad-merge-bart-origin.json') as f:
        json_data = json.load(f)

    spacy.require_gpu(3)
    nlp1 = spacy.load("en_core_sci_scibert")
    encoder = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')

    total = []
    for data in json_data:
        question = data['question']
        answer = data['answer']
        passage = data['passage']
        summary = data['summary']
        ctxs = data['ctxs']

        entity = []
        doc_q = [nlp1(question)]

        for docq in doc_q:
            for ent_q in docq.ents:
                entity.append(ent_q.text)

        t_entity = []
        seen = set()
        for ent in entity:
            if ent.lower() not in seen:
                t_entity.append(ent)
                seen.add(ent.lower())

        t_entity.sort(key=len, reverse=True)

        tokens = encoder.encode(passage, max_length=1024, truncation=True)
        labels = [0] * len(tokens)

        labels, f_entity = update_bio_labels(labels, passage, t_entity, tokens, encoder, first_only=False)
        labels = [1 if x == 2 else x for x in labels]

        temp = {
            'question': question,
            'answer': answer,
            'entity': f_entity,
            'passage': passage,
            'summary': summary,
            'ctxs': ctxs
        }
        total.append(temp)
        assert len(tokens) == len(labels)

    with open('./val_medquad-merge-bart-origin1.json', mode='w', encoding="utf-8") as f:
        json.dump(total, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
