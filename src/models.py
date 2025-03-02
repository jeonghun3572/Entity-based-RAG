import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model, TaskType
from transformers import LlamaForCausalLM, LlamaTokenizer

import src.utils
from src.retriever import Retriever


class LM:
    def get_perplexity_data(self, text) -> Optional[dict]:
        raise NotImplementedError

    def initialize_retriever(self, args):
        self.args = args
        if args.do_retrieval:
            self.retriever = Retriever(args)
        else:
            self.retriever = None

class LlamaLM(LM):
    def __init__(self, context_len=1024, max_seq_len=2048, verbose=False, batch_size=16, optimizer=None, args=None):
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose
        self.wb = src.utils.WaitBlocker()
        self.tmp = 1
        self.batch_size=batch_size
        self.optimzer=optimizer
        self.args = args

        torch.set_grad_enabled(False)
        self.model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.bfloat16, device_map='balanced')
        self.tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.peft_config = LoraConfig(r=64,
                                    lora_alpha=16,
                                    target_modules=['q_proj', 'v_proj'],
                                    lora_dropout=0.05,
                                    bias="none",
                                    task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, self.peft_config)

    def forward_training(self, text):
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = src.utils.get_rolling_token_windows(token_list=input_ids,
                                                                    prefix_token=self.tokenizer.eos_token_id,
                                                                    max_seq_len=self.max_seq_len,
                                                                    context_len=self.context_len,)
        batch_loss = []
        batch_index = 0

        # Remaining windows: input_tokens are context, pred_tokens are prediction
        for input_tokens, pred_tokens in tqdm(rolling_token_windows):
            retriever_loss = self.forward_training_single(input_tokens, pred_tokens)
            batch_loss.append(retriever_loss)
            if batch_index == self.batch_size:
                batch_loss = torch.stack(batch_loss)
                batch_loss = torch.mean(batch_loss)
                batch_loss.backward()
                batch_loss = []
                batch_index = 0
                self.optimizer.step()
                self.optimizer.zero_grad()


    def forward_training_single(self, input_tokens, pred_tokens):
        # query_id = input_tokens[:-len(pred_tokens)]
        query_id = input_tokens
        print("len(context):", len(query_id), "len(pred_tokens):", len(pred_tokens))
        query = self.tokenizer.decode(query_id)
        docs, scores = self.retriever.retrieve_passage([query])
        plain_docs = [doc["text"] for doc in docs]

        # encode the retrieved docs
        questions_embedding = self.retriever.embed_queries([query])
        passages_embedding = self.retriever.embed_queries(plain_docs)

        retriever_score = torch.einsum("id, ijd->ij", [questions_embedding, passages_embedding])


        all_gold_score = []
        for i in range(len(docs)):
            doc_str = plain_docs[i]
            doc_encodings = self.tokenizer.encode(doc_str)
            input_tokens_tmp = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
            block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens)
            gold_score = block_output["logprobs"]
            all_gold_score.append(gold_score)

        all_gold_score = torch.FloatTensor(all_gold_score)
        retriever_loss = self.kldivloss(retriever_score, gold_score)

        return retriever_loss

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.args.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.args.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)

    # noinspection DuplicatedCode
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text=text)["input_ids"]
        rolling_token_windows = src.utils.get_rolling_token_windows(token_list=input_ids,
                                                                    prefix_token=self.tokenizer.eos_token_id,
                                                                    max_seq_len=self.max_seq_len,
                                                                    context_len=self.context_len,)

        # noinspection PyListCreation
        all_logprobs = []
        all_positions = []

        # Remaining windows: input_tokens are context, pred_tokens are prediction
        for input_tokens, pred_tokens in tqdm(rolling_token_windows):
            query_id = input_tokens[:-len(pred_tokens)]

            # do retrieval
            if self.args.do_retrieval and (query_id != []):
                query = self.tokenizer.decode(query_id)
                docs, scores = self.retriever.retrieve_passage([query])[0]
                plain_docs = [doc["text"] for doc in docs]

                if self.args.ensemble == 0:
                    doc_str = "\n".join(plain_docs)
                    print(f"query: {[query]}\nretrieved doc: {[doc_str]}")
                    doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                    input_tokens = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1) # list (165)
                    print("retrieve + context: ", len(input_tokens)-len(pred_tokens))
                else:
                    '''
                    a + b + c = log(e^log(a) + e^log(b) + e^log(c))
                    '''
                    logprobs_list = []
                    block_output = None
                    assert self.args.ensemble <= len(plain_docs)

                    for i in range(self.args.ensemble):
                        doc_str = plain_docs[i]
                        doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                        input_tokens_tmp = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
                        block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens)
                        logprobs_list.append(block_output["logprobs"])
                        # sum(np.isinf(block_output["logprobs"]))

                    # block_output["logprobs"] = np.log(np.mean(np.exp(logprobs_list), axis=0))
                    # len(logprobs_list) = number of ensemble
                    # block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list), dim=0) - np.log(len(logprobs_list))
                    # apply softmax to scores 

                    scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
                    scores = torch.log(torch.FloatTensor(scores)).reshape(-1, 1)
                    scores = scores.repeat(1, len(logprobs_list[0]))
                    block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list)+scores, dim=0) 
                    block_output["logprobs"] = block_output["logprobs"].numpy()
            else:
                block_output = self.get_token_logprobs(input_tokens=input_tokens, pred_tokens=pred_tokens)
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)

        return {"logprobs": all_logprobs,
                "positions": all_positions,
                "length": len(all_logprobs),
                "utf8_length": len(text.encode('utf-8')),}

    @torch.no_grad
    def get_token_logprobs(self, input_tokens, pred_tokens):
        # token_ids = input_tokens + [pred_tokens[-1]] # TODO
        input_tokens = torch.tensor(input_tokens).to(self.model.device)
        pred_tokens = torch.tensor(pred_tokens).to(self.model.device)
        input_tokens = input_tokens.unsqueeze(dim=0) # torch.Size([1, 165])
        pred_tokens = pred_tokens.unsqueeze(dim=0) # torch.Size([1, 165])
        with self.wb.check_valid():
            output = self.model(input_tokens, return_dict=True)
        # loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        output.logits = output.logits.squeeze() # torch.Size([165, 32000])
        input_tokens = input_tokens.squeeze()
        pred_tokens = pred_tokens.squeeze()
        logprobs = logit_to_logprob(output.logits[-len(pred_tokens):].detach().cpu().numpy()) # TODO
        # pred_tokens = pred_tokens.squeeze()
        # input_tokens = input_tokens.squeeze()
        # neg_logprobs = loss_fct(output.logits[-len(pred_tokens):], pred_tokens,).detach().cpu().numpy()

        if self.verbose:
            print("\nContext:", len(self.tokenizer.convert_ids_to_tokens(input_tokens))) # 165
            print("Predicting:", len(self.tokenizer.convert_ids_to_tokens(pred_tokens)))
            print("Perplexity:", np.exp(-logprobs.mean())) # TODO
            # print("Perplexity:", np.exp(neg_logprobs.mean()))
            print()
        positions = np.arange(len(input_tokens) - len(pred_tokens), len(input_tokens)) # (165,)
        return {"logprobs": logprobs, "positions": positions} # TODO
        # return {"logprobs": -neg_logprobs, "positions": positions,}


def logit_to_logprob(x):
    return np.log(np.exp(x) / np.sum(np.exp(x)))


def create_model():
    model = LlamaLM()
    return model