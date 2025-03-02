import re
import torch
import string
import evaluate
import datasets
import itertools
import collections
import numpy as np
from evaluate import logging
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def em(samples, pred_answers):
    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        num_all_answers += 1
        num_correct_answers += 1 if np.count_nonzero(compute_exact(sample, pred_answer)) != 0 else 0
        
    return num_correct_answers / (num_all_answers + 1e-16)


def f1(samples, pred_answers):
    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    f1_score = 0
    for sample, pred_answer in zip(samples, pred_answers):
        num_all_answers += 1
        f1 = compute_f1(sample, pred_answer)
        f1_score += f1
        # num_correct_answers += max([compute_f1(gold_entity, pred_answer) for gold_entity in gold_entities])
        
    return f1_score/(num_all_answers + 1e-16)


def accuracy(samples, pred_answers, kg, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]

    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = set(list(itertools.chain(*
            [[entity['mention'] for entity in sample['answer_entities'] if entity['mention'] not in [None, '']]] + \
            [kg.get_aliases(entity['name']) for entity in sample['answer_entities'] if aliases == True and kg is not None and entity['name'] is not None]
        )))
        if len(gold_entities) == 0: continue

        num_all_answers += 1
        num_correct_answers += 1 \
            if np.count_nonzero([normalize_answer(gold_entity) in normalize_answer(pred_answer) for gold_entity in gold_entities]) != 0 \
            else 0
        
    return num_correct_answers / (num_all_answers + 1e-16)


def compute_rouge(predictions, references):
    flag = -100
    while True:
        try:
            rouge = evaluate.load('rouge')
            results = rouge.compute(predictions=predictions, references=references)
            flag = 100
            if flag == 100:
                break
        except:
            flag = -100
    return results


def compute_ppl(predictions, model, tokenizer, batch_size: int = 2, add_start_token: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding='longest',
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {np.mean(ppls)}