import gc
import os
import json
import wandb
import torch
import numpy as np
import transformers
from accelerate import Accelerator
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup, AutoConfig

import src
import src.utils
import src.metrics
from src import normalize_text
from src.options import Options
from bart_model import BART_class_head


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datapaths, normalize=False, global_rank=-1, world_size=-1, maxload=None):
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)
        
    def __len__(self):
        return len(self.data)

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break
        return examples, counter

    def __getitem__(self, index):
        example = self.data[index]
        return {
            "passage": self.normalize_fn(example['passage']),
            "summary": self.normalize_fn(example['summary']),
            "tagging": example['tagging'],
            "entity": example['entity'],
        }


class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        passage = [ex['passage'] for ex in batch]
        summary = [ex['summary'] for ex in batch]
        tagging = [ex['tagging'] for ex in batch]
        entity = [ex['entity'] for ex in batch]

        p_out = self.tokenizer.batch_encode_plus(
            passage,
            max_length=1024,
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        s_out = self.tokenizer.batch_encode_plus(
            summary,
            max_length=1024,
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        tag = self.tokenizer.pad(
            {"input_ids": tagging},
            padding='longest',
            return_attention_mask=False,
            return_tensors="pt",
        )

        return {
            "p_ids": p_out['input_ids'],
            "p_mask": p_out['attention_mask'],
            "s_ids": s_out['input_ids'],
            "s_mask": s_out['attention_mask'],
            "summary": summary,
            "entity": entity,
            "tagging": tag['input_ids'],
        }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    options = Options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    os.makedirs(opt.output_dir, exist_ok=True)
    logger = src.utils.init_logger(opt)

    if opt.wandb_run:
        wandb.init(project=opt.wandb_proj)
        wandb.run.name = opt.wandb_run_name

    id2label = {0: "O", 1: "B", 2: "I"}
    label2id = {"O": 0, "B": 1, "I": 2}

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    cfg = AutoConfig.from_pretrained('facebook/bart-large-cnn', num_labels=len(id2label), id2label=id2label, label2id=label2id)
    model = BART_class_head(cfg)
    model_temp = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    model.load_state_dict(model_temp.state_dict(), strict=False)
    del model_temp

    special_tokens_dict = {'additional_special_tokens': ['<ent>', '</ent>', '<desc>', '</desc>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = Dataset(datapaths=opt.train_data, normalize=True, maxload=opt.maxload)
    eval_dataset = Dataset(datapaths=opt.eval_data, normalize=True, maxload=opt.maxload)
    collator = Collator(tokenizer)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=opt.per_device_train_batch_size, num_workers=opt.num_workers, collate_fn=collator, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=opt.per_device_eval_batch_size, num_workers=opt.num_workers, collate_fn=collator, drop_last=True)

    num_training_steps_per_batch = len(train_dataloader)
    num_training_steps = opt.num_train_epochs * num_training_steps_per_batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(num_training_steps_per_batch * 0.1), num_training_steps=num_training_steps)

    accelerator = Accelerator(gradient_accumulation_steps=opt.gradient_accumulation_steps)
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, scheduler)

    train_step = 0
    valid_step = 0
    best_loss = float('inf')
    best_loss1 = float('inf')
    best_loss2 = float('inf')

    for epoch in range(opt.num_train_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_dataloader
                logger.info(f"****** Start Train | epoch: {epoch} ******")
            else:
                model.eval()
                dataloader = eval_dataloader
                val_loss = 0
                val_loss1, val_loss2 = 0, 0
                logger.info(f"****** START EVAL | epoch: {epoch} ******")

            for _, batch in enumerate(dataloader):
                with accelerator.accumulate(model):
                    p_ids, p_mask = batch['p_ids'], batch['p_mask']
                    s_ids, s_mask = batch['s_ids'], batch['s_mask']
                    tagging = batch['tagging']
                    assert p_ids.shape == tagging.shape

                    labels = torch.where(s_ids == tokenizer.pad_token_id, -100, s_ids)
                    ner_labels = torch.where(p_ids <= 3, -100, tagging)

                    decoder_ids = shift_tokens_right(s_ids, tokenizer.pad_token_id, model.config.decoder_start_token_id)
                    decoder_mask = torch.where(decoder_ids == tokenizer.pad_token_id, 0, 1)

                    output, ner_loss = model(
                        input_ids=p_ids,
                        attention_mask=p_mask,
                        decoder_input_ids=decoder_ids,
                        decoder_attention_mask=decoder_mask,
                        labels=labels,
                        ner_labels=ner_labels,
                    )
                    sum_loss = output['loss'] * opt.loss1_factor
                    ner_loss = ner_loss * opt.loss2_factor
                    total_loss = sum_loss + ner_loss

                    if phase == "train":
                        if opt.wandb_run:
                            wandb.log({"train_loss": total_loss, "sum_loss": sum_loss, "ner_loss": ner_loss, "train_step": train_step})
                        if train_step % opt.log_freq == 0:
                            logger.info(f"{train_step} | lr: {scheduler.get_last_lr()[0]} | loss: {total_loss}")

                        accelerator.backward(total_loss)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        train_step += 1
                    else:
                        if opt.wandb_run:
                            wandb.log({"valid_loss": total_loss, "valid_sum_loss": sum_loss, "valid_ner_loss": ner_loss, "valid_step": valid_step})
                        val_loss += total_loss / len(dataloader)
                        val_loss1 += sum_loss / len(dataloader)
                        val_loss2 += ner_loss / len(dataloader)
                        valid_step += 1

            if phase == "val":
                if val_loss1 < best_loss1 and val_loss2 < best_loss2:
                    src.utils.save(model, optimizer, scheduler, opt, opt.output_dir, f"epoch{epoch}")
                    logger.info(f"****** Saved model | epoch: {epoch} ******")
                    best_loss = val_loss
                    best_loss1 = val_loss1
                    best_loss2 = val_loss2

                pred_entity_isin, entity_num = 0, 0
                pred, summary, entity = [], [], []
                for batch in dataloader:
                    p_ids, p_mask = batch['p_ids'], batch['p_mask']
                    summary_ids = model.generate(input_ids=p_ids, attention_mask=p_mask, num_beams=opt.num_beams, early_stopping=True)
                    pred_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    pred.append(pred_summary)
                    summary.append(batch['summary'])
                    entity.append(batch['entity'])

                predictions = [item for sublist in pred for item in sublist]
                summaries = [item for sublist in summary for item in sublist]
                entities = [item for sublist in entity for item in sublist]

                results = src.metrics.compute_rouge(predictions, summaries)
                rouge1 = results['rouge1']
                rouge2 = results['rouge2']
                rougeL = results['rougeL']

                for i, entity in enumerate(entities):
                    for ent in entity:
                        entity_num += 1
                        if ent in predictions[i]:
                            pred_entity_isin += 1
                
                our_pred_ent_ratio = pred_entity_isin / entity_num

                if opt.wandb_run:
                    wandb.log({
                        "val Rouge1": rouge1,
                        "val Rouge2": rouge2,
                        "val RougeL": rougeL,
                        "val entity_ratio": our_pred_ent_ratio,
                    })

    logger.info("****** End Train! ******")


if __name__ == "__main__":
    main()
