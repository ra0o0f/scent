from scent.data.scent_data import (
    END_OF_TITLE_TOKEN,
    MENTION_START_TOKEN,
    NODE_END_TOKEN,
    NODE_START_TOKEN,
)
from scent.models.tokengt import GraphFeatureTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AutoConfig,
    PreTrainedModel,
)
from typing import Dict, Optional, Tuple, Set, List


class InputEmbedder(nn.Module):
    def __init__(self, word_embeddings, num_entity_embeddings):
        super().__init__()

        self.word_embeddings = word_embeddings

        self.entity_token_embeddings = nn.Embedding(
            num_entity_embeddings, self.word_embeddings.embedding_dim
        )

        with torch.no_grad():
            mean_emb = self.word_embeddings.weight.mean(dim=0)
            self.entity_token_embeddings.weight.data.copy_(
                mean_emb.unsqueeze(0).repeat(num_entity_embeddings, 1)
            )

            self.entity_token_embeddings.weight.data += (
                torch.randn_like(self.entity_token_embeddings.weight) * 0.02
            )

        self.vocab_size = num_entity_embeddings + self.word_embeddings.num_embeddings

    def forward(self, input_ids):

        entity_token_mask = input_ids >= self.word_embeddings.num_embeddings

        word_input_ids = input_ids.masked_fill(entity_token_mask, 0)
        word_embs = self.word_embeddings(word_input_ids)

        entity_ids = (input_ids - self.word_embeddings.num_embeddings).masked_fill(
            ~entity_token_mask, 0
        )
        entity_embs = self.entity_token_embeddings(entity_ids)

        embs = torch.where(entity_token_mask.unsqueeze(-1), entity_embs, word_embs)

        return embs


class ScentHead(nn.Module):
    def __init__(self, lm_head, entity_token_embeddings):
        super().__init__()

        self.lm_head = lm_head
        self.gelu = nn.GELU()

        self.entity_token_embeddings = entity_token_embeddings
        self.entity_bias = nn.Parameter(
            torch.zeros((entity_token_embeddings.num_embeddings))
        )

    def forward(self, features):

        x = self.lm_head.dense(features)
        x = self.gelu(x)
        x = self.lm_head.layer_norm(x)
        lm_logits = self.lm_head.decoder(x)

        entity_logits = F.linear(
            x, self.entity_token_embeddings.weight, self.entity_bias
        )

        # unified logits [bs, seq, 50265 + 4]
        return torch.cat([lm_logits, entity_logits], dim=-1)


class ScentEncoder(nn.Module):
    def __init__(
        self,
        config: AutoConfig,
        lm_model: RobertaForMaskedLM,
        tokenizer: RobertaTokenizer,
        graph_training: bool,
        whitening_stats=None,
        # lora_r: int = 32,
        # lora_alpha: int = 64,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.lm_model = lm_model

        self.register_buffer("whitening_mu", whitening_stats["mu"])
        self.register_buffer("whitening_W", whitening_stats["W"])

        self.roberta = self.lm_model.roberta

        text_peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            # target_modules=["query", "value"],
            target_modules=["query", "value", "key", "output.dense"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
        )

        graph_peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value", "key", "output.dense"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
        )

        self.peft_roberta = get_peft_model(
            self.roberta, text_peft_config, adapter_name="text_adapter"
        )
        self.peft_roberta.add_adapter("graph_adapter", graph_peft_config)

        # self.peft_roberta.base_model.model.embeddings.word_embeddings.weight.requires_grad = True
        # for param in self.lm_model.lm_head.parameters():
        #     param.requires_grad = True

        self.roberta_encoder = self.peft_roberta.base_model.model.encoder
        self.roberta_embedding = self.peft_roberta.base_model.model.embeddings
        self.roberta_word_embedding = (
            self.peft_roberta.base_model.model.embeddings.word_embeddings
        )
        self.roberta_lm_head = self.lm_model.lm_head

        # make sure the new ones are added to "add_special_tokens" method
        num_entity_embeddings = 4
        self.node_start_id = self.tokenizer.convert_tokens_to_ids(NODE_START_TOKEN)
        self.node_end_id = self.tokenizer.convert_tokens_to_ids(NODE_END_TOKEN)
        self.eot_id = self.tokenizer.convert_tokens_to_ids(END_OF_TITLE_TOKEN)
        self.mention_start_id = self.tokenizer.convert_tokens_to_ids(
            MENTION_START_TOKEN
        )

        self.gelu = nn.GELU()

        self.input_embedder = InputEmbedder(
            self.roberta_word_embedding, num_entity_embeddings=num_entity_embeddings
        )

        self.scent_head = ScentHead(
            lm_head=self.roberta_lm_head,
            entity_token_embeddings=self.input_embedder.entity_token_embeddings,
        )

        self.entity_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.entity_loss_fn = nn.CosineEmbeddingLoss()

        ### graph

        self.graph_training = graph_training

        self.graph_feature_tokenizer = GraphFeatureTokenizer(
            rand_node_id=False,
            rand_node_id_dim=64,
            orf_node_id=True,
            orf_node_id_dim=64,
            lap_node_id=False,
            lap_node_id_k=8,
            lap_node_id_sign_flip=False,
            lap_node_id_eig_dropout=0.0,
            type_id=True,
            hidden_dim=config.hidden_size,
            n_layers=config.num_hidden_layers,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.graph_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.graph_loss_fn = nn.CosineEmbeddingLoss()

    def forward_entity_feature(self, feature_batch):

        if feature_batch["input_ids"].size(0) == 0:
            return torch.empty(
                (0, self.config.hidden_size),
                dtype=self.roberta_word_embedding.weight.dtype,
                device=self.roberta_word_embedding.weight.device,
            )
        with self.peft_roberta.disable_adapter():

            if self.whitening_mu is not None:

                outputs = self.roberta(**feature_batch, output_hidden_states=True)

                first_layer = outputs.hidden_states[0]
                last_layer = outputs.hidden_states[-1]
                avg_repr = (first_layer + last_layer) / 2.0

                mask = (
                    feature_batch["attention_mask"]
                    .unsqueeze(-1)
                    .expand(avg_repr.size())
                    .float()
                )
                sum_embeddings = torch.sum(avg_repr * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                raw_embeddings = sum_embeddings / sum_mask
                entity_emb = torch.matmul(
                    raw_embeddings - self.whitening_mu, self.whitening_W
                )
                # print('entity_emb',entity_emb.dtype)

            else:
                out = self.roberta(**feature_batch, return_dict=True)
                entity_emb = out.last_hidden_state[:, 0, :]

        return entity_emb

    def _forward_graph_internal(self, batch, mask_input):
        self.peft_roberta.set_adapter("graph_adapter")

        clean_node_features = self.forward_entity_feature(batch["node_feature_batch"])
        batch["edge_feature"] = self.forward_entity_feature(batch["edge_feature_batch"])

        self.peft_roberta.set_adapter("graph_adapter")

        aligned_targets = batch["node_ids"].repeat_interleave(batch["node_num"])
        target_indices_mask = batch["n_id"] == aligned_targets

        if mask_input:
            masked_node_features = clean_node_features.clone()
            mask_token_emb = self.roberta_word_embedding.weight[
                self.tokenizer.mask_token_id
            ]
            masked_node_features[target_indices_mask] = mask_token_emb.to(
                torch.bfloat16
            )
            batch["node_feature"] = masked_node_features
        else:
            batch["node_feature"] = clean_node_features

        padded_feature, padding_mask, padded_index = self.graph_feature_tokenizer(batch)

        attention_mask = (~padding_mask).long()
        extended_attention_mask = self.lm_model.get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device
        )

        outputs = self.roberta_encoder(
            padded_feature, attention_mask=extended_attention_mask, return_dict=True
        )
        sequence_output = outputs.last_hidden_state

        # remove special tokens (graph_token, null_token)
        seq_out_trimmed = sequence_output[:, 2:, :]
        seq_padding_mask = padding_mask[:, 2:]

        is_node = padded_index[..., 0] == padded_index[..., 1]
        node_indices_in_seq = padded_index[..., 0]

        graph_offsets = batch["graph_offsets"]
        node_indices_in_seq = node_indices_in_seq + graph_offsets.unsqueeze(1)

        is_target_node = target_indices_mask[node_indices_in_seq]

        final_sequence_mask = is_node & is_target_node & (~seq_padding_mask)

        encoded_nodes = seq_out_trimmed[final_sequence_mask]

        indices_used = node_indices_in_seq[final_sequence_mask]
        target_features = clean_node_features[indices_used]

        if mask_input:
            batch["node_feature"] = clean_node_features

        return encoded_nodes, target_features

    def forward_graph_unmasked(self, batch):

        training_mode = self.training
        self.eval()

        with torch.no_grad():
            encoded_nodes, _ = self._forward_graph_internal(batch, mask_input=False)

        self.train(training_mode)

        return encoded_nodes

    def forward_graph(self, batch):

        encoded_nodes, target_features = self._forward_graph_internal(
            batch, mask_input=True
        )

        predictions = self.graph_decoder(encoded_nodes)

        graph_loss = torch.tensor(0.0, device=encoded_nodes.device)

        if predictions.shape[0] > 0:
            target_ones = torch.ones(predictions.shape[0], device=predictions.device)
            graph_loss = self.graph_loss_fn(predictions, target_features, target_ones)

        return {"graph_loss": graph_loss}

    def forward_text(self, batch):

        self.peft_roberta.set_adapter("text_adapter")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        inputs_embeds = self.input_embedder(input_ids=input_ids)

        embedding_output = self.roberta_embedding(inputs_embeds=inputs_embeds)
        extended_attention_mask = self.lm_model.get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device
        )
        outputs = self.roberta_encoder(
            embedding_output, attention_mask=extended_attention_mask, return_dict=True
        )
        # outputs = self.roberta(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     return_dict=True,
        #     input_ids=None
        # )

        sequence_output = outputs.last_hidden_state

        prediction_scores = self.scent_head(sequence_output)

        mlm_loss = self.mlm_loss_fn(
            prediction_scores.view(-1, self.input_embedder.vocab_size), labels.view(-1)
        )

        entity_loss = torch.tensor(0.0, device=input_ids.device)

        if self.graph_training:
            node_start_indices = (input_ids == self.node_start_id).nonzero(
                as_tuple=False
            )
            if node_start_indices.size(0) != 0:

                node_hidden_states = sequence_output[
                    node_start_indices[:, 0], node_start_indices[:, 1]
                ]
                node_embs = self.entity_decoder(node_hidden_states)

                self.peft_roberta.set_adapter("graph_adapter")
                node_embs_targets = self.forward_graph_unmasked(batch)
                self.peft_roberta.set_adapter("text_adapter")

                target_ones = torch.ones(node_embs.shape[0], device=input_ids.device)
                entity_loss = self.entity_loss_fn(
                    node_embs, node_embs_targets, target=target_ones
                )

        text_loss = mlm_loss + entity_loss

        return {
            "text_loss": text_loss,
            "mlm_loss": mlm_loss,
            "entity_loss": entity_loss,
        }

    def forward(self, batch):

        if self.graph_training:
            graph_output = self.forward_graph(batch)

        output = self.forward_text(batch)

        output["loss"] = output["text_loss"]

        if self.graph_training:
            output["loss"] += graph_output["graph_loss"]
            output = output | graph_output

        return output
