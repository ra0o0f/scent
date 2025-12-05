import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, Set, List
from transformers import AutoTokenizer
from torch_geometric.loader import NeighborLoader

NODE_START_TOKEN = "<node_start>"
NODE_END_TOKEN = "<node_end>"
END_OF_TITLE_TOKEN = "<end_of_title>"
MASK_TOKEN = "<mask>"

class ScentDataset(Dataset):
    def __init__(
        self, 
        aida_data: List[Dict], 
        node_to_idx: Dict[str, int],
        tokenizer: AutoTokenizer,
        mask_buffer_len: int = 50,
        max_seq_len: int = 512,
        ignore_label_id: int = -100
    ):
        self.data = aida_data
        self.node_to_idx = node_to_idx
        self.tokenizer = tokenizer
        self.mask_buffer_len = mask_buffer_len
        self.max_seq_len = max_seq_len
        self.ignore_label_id = ignore_label_id
        
        special_tokens = {
            'additional_special_tokens': [NODE_START_TOKEN, NODE_END_TOKEN, END_OF_TITLE_TOKEN]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.samples, self.candidate_set = self._build_index()

        self.node_start_id = self.tokenizer.convert_tokens_to_ids(NODE_START_TOKEN)
        self.node_end_id = self.tokenizer.convert_tokens_to_ids(NODE_END_TOKEN)
        self.eot_id = self.tokenizer.convert_tokens_to_ids(END_OF_TITLE_TOKEN)
        self.mask_id = self.tokenizer.mask_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
    
    def _build_index(self):
        samples = []
        candidate_set = {}
        for doc_idx, doc in enumerate(self.data):
            annotations = doc.get('annotations', [])
            if not annotations:
                continue
                
            i = 0
            while i < len(annotations):
                annot = annotations[i]
                yago_entity = annot.get('yago_entity')
                
                if yago_entity and yago_entity != '--NME--':
                    full_uri = f'http://yago-knowledge.org/resource/{yago_entity}'
                    mention = annot.get('mention')
                    
                    if full_uri in self.node_to_idx:
                        start_idx = i
                        end_idx = i

                        if mention not in candidate_set:
                            candidate_set[mention] = set()
                        candidate = candidate_set[mention]
                        candidate.add(full_uri)
                        
                        while end_idx + 1 < len(annotations):
                            next_annot = annotations[end_idx + 1]
                            if next_annot.get('yago_entity') == yago_entity:
                                end_idx += 1
                            else:
                                break
                        
                        samples.append((doc_idx, start_idx, end_idx, full_uri))
                        
                        i = end_idx + 1
                        continue
                
                i += 1
        return samples, candidate_set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        doc_idx, start_idx, end_idx, node_uri = self.samples[idx]
        
        doc = self.data[doc_idx]
        annotations = doc['annotations']
        target_annot = annotations[start_idx]
        
        target_mention = target_annot['mention']
        node_id = self.node_to_idx[node_uri]

        left_tokens_raw = [a.get('token', '') for a in annotations[:start_idx]]
        right_tokens_raw = [a.get('token', '') for a in annotations[end_idx+1:]]
        
        left_text = " ".join(left_tokens_raw)
        right_text = " ".join(right_tokens_raw)

        left_ids = self.tokenizer.encode(left_text, add_special_tokens=False, truncation=True, max_length=2048)
        right_ids = self.tokenizer.encode(right_text, add_special_tokens=False, truncation=True, max_length=2048)
        mention_ids = self.tokenizer.encode(target_mention, add_special_tokens=False)
        
        fixed_cost = 4 + self.mask_buffer_len
        context_budget = self.max_seq_len - fixed_cost
        
        len_l = len(left_ids)
        len_r = len(right_ids)
        
        if len_l + len_r <= context_budget:
            take_l = len_l
            take_r = len_r
        else:
            
            # take_r <= len_r  =>  (budget - take_l) <= len_r  =>  take_l >= budget - len_r
            min_l = max(0, context_budget - len_r)
            max_l = min(len_l, context_budget)
            
            take_l = random.randint(min_l, max_l)
            take_r = context_budget - take_l
            
        final_left = left_ids[-take_l:] if take_l > 0 else []
        
        final_right = right_ids[:take_r] if take_r > 0 else []

        mask_segment = [self.mask_id] * self.mask_buffer_len
        
        input_ids = (
            [self.cls_id] + 
            final_left + 
            [self.node_start_id] + 
            mask_segment + 
            [self.node_end_id] + 
            final_right + 
            [self.sep_id]
        )

        labels = [self.ignore_label_id] * len(input_ids)
        
        mask_start_idx = 1 + len(final_left) + 1
        
        target_seq = mention_ids + [self.eot_id]
        
        if len(target_seq) > self.mask_buffer_len:
            target_seq = target_seq[:self.mask_buffer_len]
            
        for i, tid in enumerate(target_seq):
            labels[mask_start_idx + i] = tid
            
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "node_id": node_id
        }

class ScentDataCollator:
    def __init__(self, tokenizer, sample_graph_nodes=False, pyg_data=None, idx_to_node=None, idx_to_rel=None, graph_num_neighbors=[3,3]):
        self.tokenizer = tokenizer

        self.sample_graph_nodes = sample_graph_nodes

        if self.sample_graph_nodes:

            self.idx_to_node = idx_to_node
            self.idx_to_rel = idx_to_rel

            self.neighbor_loader = NeighborLoader(
                pyg_data,
                num_neighbors=graph_num_neighbors,
                batch_size=1,
                input_nodes=None, 
                shuffle=False,
                replace=False,
                disjoint=True,
                num_workers=0,
                weight_attr='edge_degree_weights_squared'
            )

    def node_name(self, n_id):
        return self.idx_to_node[n_id].split('/')[-1]

    def edge_name(self, e_id):
        return self.idx_to_rel[e_id].split('/')[-1]
    
    def graph_collate(self, starting_node_ids):

        batch = self.neighbor_loader.collate_fn(starting_node_ids)

        device = batch.edge_index.device
    
        node_graph_idx = batch.batch
        _, node_sort_idx = torch.sort(node_graph_idx)

        batch.batch = batch.batch[node_sort_idx]
        batch.n_id = batch.n_id[node_sort_idx]

        node_map = torch.empty(batch.num_nodes, dtype=torch.long, device=device)
        node_map[node_sort_idx] = torch.arange(batch.num_nodes, device=device)
        batch.edge_index = node_map[batch.edge_index]

        source_node_indices = batch.edge_index[0]
        edge_graph_idx = batch.batch[source_node_indices]

        _, edge_sort_idx = torch.sort(edge_graph_idx, stable=False)
        batch.edge_index = batch.edge_index[:, edge_sort_idx]
        batch.edge_attr = batch.edge_attr[edge_sort_idx]
        edge_graph_idx = edge_graph_idx[edge_sort_idx]

        node_num = torch.bincount(batch.batch)
        edge_num = torch.bincount(edge_graph_idx, minlength=len(node_num))

        cum_nodes = torch.cumsum(node_num, 0)
        graph_offsets = torch.cat([torch.tensor([0], device=device), cum_nodes[:-1]])
        edge_offsets = graph_offsets[edge_graph_idx]

        edge_index = batch.edge_index - edge_offsets

        #node_data = node_features[batch.n_id].unsqueeze(1)
        e_id = batch.edge_attr.long()
        #edge_data = rel_features[e_id].unsqueeze(1)

        node_feature_seqs = [f"{self.node_name(nid.item())}" for nid in batch.n_id]
        edge_feature_seqs = [f"{self.edge_name(eid.item())}" for eid in e_id]

        node_feature_batch = self.tokenizer(
            node_feature_seqs, 
            padding=True, 
            add_special_tokens=True, 
            return_tensors='pt'
        )

        if len(edge_feature_seqs) > 0:
            edge_feature_batch = self.tokenizer(
                edge_feature_seqs, 
                padding=True, 
                add_special_tokens=True, 
                return_tensors='pt'
            )
        else:
            edge_feature_batch = {
                'input_ids': torch.zeros((0, 0), dtype=torch.long),
                'attention_mask': torch.zeros((0, 0), dtype=torch.long)
            }

        return {
            'node_num': node_num,
            'edge_num': edge_num,
            'edge_index': edge_index,
            'edge_graph_idx': edge_graph_idx,
            'n_id': batch.n_id,
            'e_id': e_id,
            'node_feature_batch': node_feature_batch,
            'edge_feature_batch': edge_feature_batch,
            'graph_offsets': graph_offsets
        }
        
    def __call__(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
        node_ids = [x["node_id"] for x in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        attention_mask_padded = (input_ids_padded != self.tokenizer.pad_token_id).long()
        
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        node_ids_tensor = torch.tensor(node_ids, dtype=torch.long)

        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "node_ids": node_ids_tensor
        }
        
        if not self.sample_graph_nodes:
            return batch
        
        graph_batch = self.graph_collate(node_ids_tensor)
        
        return batch | graph_batch