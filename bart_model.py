import torch
import numpy as np
import transformers
from transformers import BartForConditionalGeneration

class BartClassificationHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, inner_dim)
        self.dropout = torch.nn.Dropout(p=pooler_dropout)
        self.out_proj = torch.nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BART_class_head(BartForConditionalGeneration):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.config.num_labels = 3
        self.config.attention_dropout = 0.1
        self.config.classifier_dropout = 0.1
        self.classfication_head = BartClassificationHead(
            self.config.d_model,
            self.config.d_model,
            self.config.num_labels,
            self.config.classifier_dropout,
        )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ner_labels=None,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if ner_labels is not None:
            ner_logits = self.classfication_head(model_output['encoder_last_hidden_state'])
            if isinstance(ner_labels, torch.Tensor) or isinstance(ner_labels, np.ndarray):
                loss_fct = torch.nn.CrossEntropyLoss()
                ner_loss = loss_fct(ner_logits.view(-1, self.config.num_labels), ner_labels.view(-1))
                return model_output, ner_loss
            else:
                return model_output, ner_logits
        else:
            return model_output