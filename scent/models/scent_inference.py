import re
from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.scent_model import ScentEncoder
from data.scent_data import (
    END_OF_TITLE_TOKEN,
    MENTION_START_TOKEN,
    NODE_END_TOKEN,
    NODE_START_TOKEN,
    ScentDataCollator,
)

from tqdm import tqdm

from models.utils import move_batch_to_device


class ScentInference:
    def __init__(
        self,
        model: ScentEncoder,
        tokenizer,
        candidate_set: Dict[str, set],
        node_to_idx: Dict[str, int],
        idx_to_node: Dict[int, str],
        graph_collator: ScentDataCollator = None,
        mask_buffer_len: int = 50,
        batch_size=32,
        device=None,
        alpha: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_set = candidate_set
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.graph_collator = graph_collator
        self.mask_buffer_len = mask_buffer_len
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha

        self.eot_id = self.tokenizer.convert_tokens_to_ids(END_OF_TITLE_TOKEN)
        self.node_start_id = self.tokenizer.convert_tokens_to_ids(NODE_START_TOKEN)
        self.mention_start_id = self.tokenizer.convert_tokens_to_ids(NODE_END_TOKEN)
        self.node_end_id = self.tokenizer.convert_tokens_to_ids(MENTION_START_TOKEN)
        self.mask_id = self.tokenizer.mask_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

        self.model = self.model.to(self.device)
        self.model.eval()

        self.candidate_texts = list(candidate_set.keys())
        self.text_to_cand_idx = {t: i for i, t in enumerate(self.candidate_texts)}
        assert len(self.candidate_texts) == len(self.text_to_cand_idx)

        node_uris = set(candidate_set.values())
        self.unique_uris = list(node_uris)

        for uri in self.unique_uris:
            if uri not in self.node_to_idx:
                raise ValueError(
                    f"URI found in candidate set but missing from node_to_idx: {uri}"
                )

        self.uri_to_rep_idx = {}
        self.graph_representations = None
        self.candidate_ids = None
        self.candidate_mask = None

    def build_graph_representations(self):

        if not self.model.graph_training:
            print("Graph training disabled, skipping graph reps.")
            return

        self.model.peft_roberta.set_adapter("graph_adapter")

        ordered_node_ids = [self.node_to_idx[uri] for uri in self.unique_uris]

        print(
            f"Building graph representations for {len(ordered_node_ids)} nodes using subgraphs.."
        )

        all_reps = []

        loader = DataLoader(
            ordered_node_ids,
            batch_size=self.batch_size,
            collate_fn=lambda x: torch.tensor(x, dtype=torch.long),
        )

        with torch.no_grad():
            for batch_node_ids in tqdm(loader):
                graph_batch = self.graph_collator.graph_collate(batch_node_ids)

                graph_batch["node_ids"] = batch_node_ids

                move_batch_to_device(graph_batch, self.device, graph_training=True)

                batch_reps = self.model.forward_graph_unmasked(graph_batch)

                all_reps.append(batch_reps.cpu())

        if all_reps:
            self.graph_representations = torch.cat(all_reps, dim=0).to(self.device)
            self.uri_to_rep_idx = {uri: i for i, uri in enumerate(self.unique_uris)}
        else:
            self.graph_representations = torch.empty(
                (0, self.model.config.hidden_size)
            ).to(self.device)
            self.uri_to_rep_idx = {}

    def tokenize_candidates(self):

        batch_encoding = self.tokenizer(
            self.candidate_texts,
            padding="max_length",
            truncation=True,
            max_length=self.mask_buffer_len,
            return_tensors="pt",
            add_special_tokens=False,
        )

        candidate_ids = batch_encoding["input_ids"].to(self.device)
        candidate_mask = batch_encoding["attention_mask"].to(self.device)

        seq_lengths = candidate_mask.sum(dim=1)

        eot_indices = torch.clamp(seq_lengths, max=self.mask_buffer_len - 1).long()

        batch_range = torch.arange(candidate_ids.shape[0], device=candidate_ids.device)
        candidate_ids[batch_range, eot_indices] = self.eot_id

        candidate_mask[batch_range, eot_indices] = 1

        self.candidate_ids = candidate_ids
        self.candidate_mask = candidate_mask.float()

    def score_candidates(self, prediction_scores, input_ids):

        bs, seq_len, vocab_size = prediction_scores.shape
        num_candidates, _ = self.candidate_ids.shape

        log_probs = F.log_softmax(prediction_scores, dim=-1)

        start_pos = (input_ids == self.node_start_id).nonzero(as_tuple=True)[1]
        offsets = torch.arange(
            1, self.mask_buffer_len + 1, device=input_ids.device
        ).unsqueeze(0)
        mask_indices = start_pos.unsqueeze(1) + offsets
        batch_indices = (
            torch.arange(bs, device=input_ids.device)
            .unsqueeze(1)
            .expand(-1, self.mask_buffer_len)
        )

        relevant_log_probs = log_probs[batch_indices, mask_indices, :]

        cand_ids_t = self.candidate_ids.t()  # [mask_len, num_cand]
        cand_ids_expanded = cand_ids_t.unsqueeze(0).expand(bs, -1, -1)

        gathered_scores = torch.gather(relevant_log_probs, 2, cand_ids_expanded)

        cand_mask_t = self.candidate_mask.t()  # [mask_len, num_cand]
        cand_mask_expanded = cand_mask_t.unsqueeze(0).expand(bs, -1, -1)

        masked_scores = gathered_scores * cand_mask_expanded

        sum_scores = masked_scores.sum(dim=1)  # [bs, num_candidates]

        lengths = self.candidate_mask.sum(dim=1).unsqueeze(0)  # [1, num_candidates]
        final_scores = sum_scores / lengths

        return final_scores

    def _compute_joint_scores(
        self, input_ids, prediction_scores, node_embs=None, top_k=10
    ):

        batch_size = input_ids.size(0)

        text_scores = self.score_candidates(
            prediction_scores, input_ids
        )  # [bs, num_cand_texts]

        results = []

        for b in range(batch_size):
            # relax slightly to allow for graph re-ranking to pull up lower text items)
            if self.model.graph_training:
                k_prime = min(top_k * 10, len(self.candidate_texts))
            else:
                k_prime = top_k
            top_text_scores, top_text_indices = torch.topk(text_scores[b], k=k_prime)

            batch_results = []

            for score, idx in zip(top_text_scores, top_text_indices):
                text_idx = idx.item()
                entity_label = self.candidate_texts[text_idx]
                s_prob = score.item()

                uri = self.candidate_set[entity_label]

                total_score = s_prob
                s_sim = 0.0

                if (
                    self.model.graph_training
                    and node_embs is not None
                    and self.graph_representations is not None
                ):
                    if uri in self.uri_to_rep_idx:
                        rep_idx = self.uri_to_rep_idx[uri]

                        cand_vec = self.graph_representations[rep_idx].unsqueeze(0)
                        pred_vec = node_embs[b].unsqueeze(0)

                        s_sim = F.cosine_similarity(pred_vec, cand_vec).item()

                        # Joint Score
                        total_score = s_prob + (self.alpha * s_sim)

                batch_results.append(
                    {
                        "entity_label": entity_label,
                        "uri": uri,
                        "score": total_score,
                        "s_prob": s_prob,
                        "s_sim": s_sim,
                    }
                )

            batch_results.sort(key=lambda x: x["score"], reverse=True)
            results.append(batch_results[:top_k])

        return results

    def _prepare_text_input(self, text_list: List[str]):

        pattern = re.compile(r"(.*)(<mention_start>)(.*)(<mention_end>)(.*)")

        batch_input_ids = []

        for text in text_list:
            match = pattern.search(text)
            if not match:
                raise ValueError(
                    f"Input text must contain <mention_start>...<mention_end> tags. got: {text}"
                )

            left_text = match.group(1).strip()
            mention_text = match.group(3).strip()
            right_text = match.group(5).strip()

            left_ids = self.tokenizer.encode(left_text, add_special_tokens=False)
            mention_ids = self.tokenizer.encode(mention_text, add_special_tokens=False)
            right_ids = self.tokenizer.encode(right_text, add_special_tokens=False)

            mask_segment = [self.mask_id] * self.mask_buffer_len

            input_ids = (
                [self.cls_id]
                + left_ids
                + [self.mention_start_id]
                + mention_ids
                + [self.node_start_id]
                + mask_segment
                + [self.node_end_id]
                + right_ids
                + [self.sep_id]
            )

            batch_input_ids.append({"input_ids": input_ids})

        padded_outputs = self.tokenizer.pad(
            batch_input_ids, padding=True, return_tensors="pt"
        )

        return padded_outputs.to(self.device)

    def predict(self, text: Union[str, List[str]], top_k=5):

        if isinstance(text, str):
            text = [text]

        inputs = self._prepare_text_input(text)

        with torch.no_grad():

            self.model.peft_roberta.set_adapter("text_adapter")

            inputs_embeds = self.model.input_embedder(input_ids=inputs["input_ids"])

            outputs = self.model.roberta_encoder(
                self.model.roberta_embedding(inputs_embeds=inputs_embeds),
                attention_mask=self.model.lm_model.get_extended_attention_mask(
                    inputs["attention_mask"],
                    inputs["attention_mask"].shape,
                    self.device,
                ),
                return_dict=True,
            )
            sequence_output = outputs.last_hidden_state

            prediction_scores = self.model.scent_head(sequence_output)

            node_embs = None
            if self.model.graph_training:
                node_start_indices = (
                    inputs["input_ids"] == self.node_start_id
                ).nonzero(as_tuple=False)

                if len(node_start_indices) == inputs["input_ids"].size(0):

                    node_hidden_states = sequence_output[
                        node_start_indices[:, 0], node_start_indices[:, 1]
                    ]
                    node_embs = self.model.entity_decoder(node_hidden_states)

        results = self._compute_joint_scores(
            inputs["input_ids"], prediction_scores, node_embs, top_k=top_k
        )

        return results

    def evaluate(self, dataset, top_k=1, return_logs: bool = False):

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=ScentDataCollator(self.tokenizer),
            num_workers=1,
        )

        correct_count = 0
        total_count = 0

        results_log = []

        print(f"Evaluating on {len(dataset)} samples...")

        with torch.no_grad():
            for batch in tqdm(dataloader):

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_node_ids = batch["node_ids"].to(self.device)

                self.model.peft_roberta.set_adapter("text_adapter")

                inputs_embeds = self.model.input_embedder(input_ids=input_ids)
                outputs = self.model.roberta_encoder(
                    self.model.roberta_embedding(inputs_embeds=inputs_embeds),
                    attention_mask=self.model.lm_model.get_extended_attention_mask(
                        attention_mask, attention_mask.shape, self.device
                    ),
                    return_dict=True,
                )
                sequence_output = outputs.last_hidden_state
                prediction_scores = self.model.scent_head(sequence_output)

                node_embs = None
                if self.model.graph_training:
                    node_start_indices = (input_ids == self.node_start_id).nonzero(
                        as_tuple=False
                    )
                    node_hidden_states = sequence_output[
                        node_start_indices[:, 0], node_start_indices[:, 1]
                    ]
                    node_embs = self.model.entity_decoder(node_hidden_states)

                batch_predictions = self._compute_joint_scores(
                    input_ids, prediction_scores, node_embs, top_k=top_k
                )

                decoded_inputs = []
                if return_logs:
                    decoded_inputs = self.tokenizer.batch_decode(
                        input_ids, skip_special_tokens=False
                    )

                for i, preds in enumerate(batch_predictions):
                    target_global_id = target_node_ids[i].item()

                    target_uri = self.idx_to_node[target_global_id]

                    found = False
                    for p in preds:
                        if p["uri"] == target_uri:
                            found = True
                            break

                    if found:
                        correct_count += 1
                    total_count += 1

                    if return_logs:
                        results_log.append(
                            {
                                "input_text": decoded_inputs[i],
                                "target_uri": target_uri,
                                "predictions": preds,
                            }
                        )

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Accuracy@{top_k}: {accuracy:.4f} ({correct_count}/{total_count})")

        if return_logs:
            return accuracy, results_log

        return accuracy
