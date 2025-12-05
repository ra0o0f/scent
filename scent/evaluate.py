from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from scent.data.scent_data import END_OF_TITLE_TOKEN, NODE_END_TOKEN, NODE_START_TOKEN, ScentDataCollator, ScentDataset

from scent.models.scent_model import ScentEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ScentInference:
    def __init__(
        self, 
        model: ScentEncoder, 
        tokenizer, 
        candidate_set: Dict[str, set],
        node_to_idx: Dict[str, int],
        idx_to_node: Dict[int, str],
        mask_buffer_len: int = 50,
        batch_size=32,
        device=None,
        alpha: float = 1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_set = candidate_set
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.mask_buffer_len = mask_buffer_len
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha

        self.eot_id = self.tokenizer.convert_tokens_to_ids(END_OF_TITLE_TOKEN)
        self.node_start_id = self.tokenizer.convert_tokens_to_ids(NODE_START_TOKEN)
        self.node_end_id = self.tokenizer.convert_tokens_to_ids(NODE_END_TOKEN)
        self.mask_id = self.tokenizer.mask_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        
        self.model = self.model.to(self.device)
        self.model.eval()

        self.candidate_texts = list(candidate_set.keys())

        self.text_to_cand_idx = {t: i for i, t in enumerate(self.candidate_texts)}

        node_uris = set()
        for uris in candidate_set.values():
            node_uris.update(uris)
        
        self.unique_uris = list(node_uris)
        self.node_titles = [uri.split('/')[-1].replace('_', ' ') for uri in self.unique_uris]
        
        self.uri_to_rep_idx = {uri: i for i, uri in enumerate(self.unique_uris)}

        self.graph_representations = None
        self.candidate_ids = None
        self.candidate_mask = None

    def build_graph_representations(self):
        
        self.model.peft_roberta.set_adapter("graph_adapter")
    
        all_reps = []
        
        print(f"building graph index: {len(self.node_titles)} nodes...")
        with torch.no_grad():
            for i in tqdm(range(0, len(self.node_titles), self.batch_size)):
                batch_texts = self.node_titles[i : i + self.batch_size]
                
                feature_batch = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    add_special_tokens=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                static_node_features = self.model.forward_entity_feature(feature_batch)
                
                batch_n = static_node_features.shape[0]
                hidden_dim = static_node_features.shape[1]
                
                target_graph_batch = {
                    "node_feature": static_node_features,
                    "edge_feature": torch.empty((0, hidden_dim), device=self.device),
                    "node_num": torch.ones(batch_n, dtype=torch.long, device=self.device),
                    "edge_index": torch.empty((2, 0), dtype=torch.long, device=self.device),
                    "edge_num": torch.zeros(batch_n, dtype=torch.long, device=self.device)
                }
                
                padded_feature, padding_mask, _ = self.model.graph_feature_tokenizer(target_graph_batch)
                
                attention_mask = (~padding_mask).long()
                extended_attention_mask = self.model.lm_model.get_extended_attention_mask(
                    attention_mask, attention_mask.shape, attention_mask.device
                )
                
                outputs = self.model.roberta_encoder(
                    padded_feature, 
                    attention_mask=extended_attention_mask, 
                    return_dict=True
                )
                
                batch_reps = outputs.last_hidden_state[:, 2, :]
                all_reps.append(batch_reps)

        self.graph_representations = torch.cat(all_reps, dim=0)

    def tokenize_candidates(self):
        
        batch_encoding = self.tokenizer(
            self.candidate_texts,
            padding='max_length',
            truncation=True,
            max_length=self.mask_buffer_len, 
            return_tensors='pt',
            add_special_tokens=False
        )

        candidate_ids = batch_encoding['input_ids'].to(self.device)
        candidate_mask = batch_encoding['attention_mask'].to(self.device)

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
        offsets = torch.arange(1, self.mask_buffer_len + 1, device=input_ids.device).unsqueeze(0)
        mask_indices = start_pos.unsqueeze(1) + offsets
        batch_indices = torch.arange(bs, device=input_ids.device).unsqueeze(1).expand(-1, self.mask_buffer_len)
        
        # [bs, mask_buffer_len, vocab]
        relevant_log_probs = log_probs[batch_indices, mask_indices, :]

        cand_ids_t = self.candidate_ids.t() # [mask_len, num_cand]
        cand_ids_expanded = cand_ids_t.unsqueeze(0).expand(bs, -1, -1)
        
        gathered_scores = torch.gather(relevant_log_probs, 2, cand_ids_expanded)
        
        cand_mask_t = self.candidate_mask.t() # [mask_len, num_cand]
        cand_mask_expanded = cand_mask_t.unsqueeze(0).expand(bs, -1, -1)
        
        masked_scores = gathered_scores * cand_mask_expanded
        
        sum_scores = masked_scores.sum(dim=1)
        
        lengths = self.candidate_mask.sum(dim=1).unsqueeze(0)
        final_scores = sum_scores / lengths
        
        return final_scores

    def _compute_joint_scores(self, input_ids, prediction_scores, node_embs=None, top_k=10):
        
        batch_size = input_ids.size(0)
        
        text_scores = self.score_candidates(prediction_scores, input_ids)
        
        results = []
        
        for b in range(batch_size):
            
            k_prime = min(top_k * 2, len(self.candidate_texts))
            top_text_scores, top_text_indices = torch.topk(text_scores[b], k=k_prime)
            
            batch_results = []
            
            for score, idx in zip(top_text_scores, top_text_indices):
                text_idx = idx.item()
                text_str = self.candidate_texts[text_idx]
                s_prob = score.item()
                
                possible_uris = self.candidate_set.get(text_str, set())
                
                if not possible_uris:
                    continue

                for uri in possible_uris:
                    total_score = s_prob
                    s_sim = 0.0
                    
                    if self.model.graph_training and node_embs is not None and self.graph_representations is not None:
                        
                        if uri in self.uri_to_rep_idx:
                            rep_idx = self.uri_to_rep_idx[uri]
                            
                            cand_vec = self.graph_representations[rep_idx].unsqueeze(0)
                            
                            pred_vec = node_embs[b].unsqueeze(0)
                            
                            s_sim = F.cosine_similarity(pred_vec, cand_vec).item()
                            
                            total_score = s_prob + (self.alpha * s_sim)
                    
                    batch_results.append({
                        "mention": text_str,
                        "uri": uri,
                        "score": total_score,
                        "s_prob": s_prob,
                        "s_sim": s_sim
                    })

                    # print(batch_results)
            
            batch_results.sort(key=lambda x: x['score'], reverse=True)
            results.append(batch_results[:top_k])
            
        return results

    def _prepare_text_input(self, text_list: List[str]):

        processed_texts = []
        for text in text_list:
            if "<predict>" not in text:
                raise ValueError(f"input text must contain <predict> token. got: {text}")
            
            mask_str = f"{NODE_START_TOKEN}" + (f"{self.tokenizer.mask_token}" * self.mask_buffer_len) + f"{NODE_END_TOKEN}"
            processed_text = text.replace("<predict>", mask_str)
            processed_texts.append(processed_text)

        inputs = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        return inputs

    def predict(self, text: Union[str, List[str]], top_k=5):
        
        if isinstance(text, str):
            text = [text]
            
        inputs = self._prepare_text_input(text)
        
        with torch.no_grad():
            
            self.model.peft_roberta.set_adapter("text_adapter")
            
            inputs_embeds = self.model.input_embedder(input_ids=inputs['input_ids'])
            
            outputs = self.model.roberta_encoder(
                self.model.roberta_embedding(inputs_embeds=inputs_embeds),
                attention_mask=self.model.lm_model.get_extended_attention_mask(
                    inputs['attention_mask'], inputs['attention_mask'].shape, self.device
                ),
                return_dict=True
            )
            sequence_output = outputs.last_hidden_state
            
            prediction_scores = self.model.scent_head(sequence_output)
            
            node_embs = None
            if self.model.graph_training:
                node_start_indices = (inputs['input_ids'] == self.node_start_id).nonzero(as_tuple=False)
                
                if len(node_start_indices) == inputs['input_ids'].size(0):
                    
                    node_hidden_states = sequence_output[node_start_indices[:, 0], node_start_indices[:, 1]]
                    node_embs = self.model.entity_decoder(node_hidden_states)

        results = self._compute_joint_scores(
            inputs['input_ids'], 
            prediction_scores, 
            node_embs, 
            top_k=top_k
        )
        
        return results

    def evaluate(self, dataset: ScentDataset, top_k=1):
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=ScentDataCollator(self.tokenizer),
            num_workers=4
        )
        
        correct_count = 0
        total_count = 0
        
        print(f"samples: {len(dataset)} ")
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_node_ids = batch['node_ids'].to(self.device) 
                
                self.model.peft_roberta.set_adapter("text_adapter")
                
                inputs_embeds = self.model.input_embedder(input_ids=input_ids)
                outputs = self.model.roberta_encoder(
                    self.model.roberta_embedding(inputs_embeds=inputs_embeds),
                    attention_mask=self.model.lm_model.get_extended_attention_mask(
                        attention_mask, attention_mask.shape, self.device
                    ),
                    return_dict=True
                )
                sequence_output = outputs.last_hidden_state
                prediction_scores = self.model.scent_head(sequence_output)
                
                node_embs = None
                if self.model.graph_training:
                    node_start_indices = (input_ids == self.node_start_id).nonzero(as_tuple=False)
                    node_hidden_states = sequence_output[node_start_indices[:, 0], node_start_indices[:, 1]]
                    node_embs = self.model.entity_decoder(node_hidden_states)
                
                batch_predictions = self._compute_joint_scores(
                    input_ids, 
                    prediction_scores, 
                    node_embs, 
                    top_k=top_k
                )
                
                for i, preds in enumerate(batch_predictions):
                    target_global_id = target_node_ids[i].item()
                    
                    target_uri = self.idx_to_node[target_global_id]
                    
                    found = False
                    for p in preds:
                        if p['uri'] == target_uri:
                            found = True
                            break
                    
                    if found:
                        correct_count += 1
                    total_count += 1
                    
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Accuracy@{top_k}: {accuracy:.4f} ({correct_count}/{total_count})")
        return accuracy
