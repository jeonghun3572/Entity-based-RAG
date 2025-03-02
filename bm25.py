import json
import re
from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi
from spacy.lang.en.stop_words import STOP_WORDS

def simple_preprocessing(text, stop_words):
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9]', ' ', text)
    text = text.split()
    return [word for word in text if word not in stop_words]

def process_data(data, stop_words):
    corpus = []
    tokenized_corpus = []
    for ctx in data['ctxs']:
        passage = ctx['title'] + "\n" + ctx['text']
        post_passage = simple_preprocessing(passage, stop_words)
        tokenized_corpus.append(post_passage)
        corpus.append(passage)
    return corpus, tokenized_corpus

def retrieve_passages(bm25, query, corpus, n):
    tokenized_query = query.split(" ")
    return bm25.get_top_n(tokenized_query, corpus, n=n)

def main():
    with open("./datasets/bioasq-0_origin/bioasq-test.json") as f:
        json_data = json.load(f)

    stop_words = list(STOP_WORDS)
    total = []

    for data in tqdm(json_data):
        temp = {}
        corpus, tokenized_corpus = process_data(data, stop_words)
        bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)

        query = data['question']
        temp['question'] = query
        temp['answer'] = data['answer']

        retrieved_passage = retrieve_passages(bm25, query, corpus, n=5)
        passage_notitle = " ".join([p.split('\n', 1)[1] for p in retrieved_passage])
        temp['passage'] = "\n\n".join(retrieved_passage).strip()
        temp['passage_notitle'] = passage_notitle.strip()

        temp['passage_top1'] = "\n\n".join(retrieve_passages(bm25, query, corpus, n=1)).strip()
        temp['passage_top10'] = "\n\n".join(retrieve_passages(bm25, query, corpus, n=10)).strip()

        total.append(temp)

    with open("./datasets/bioasq-1_bm25/bioasq-test.json", 'w', encoding="UTF-8") as f:
        json.dump(total, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
