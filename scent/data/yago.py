import re
import os
import gzip
import csv
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch_geometric.utils import to_networkx, sort_edge_index
from torch_geometric.data import Data

def parse_aida_dataset(file_path):
    # train_data = {}
    # testa_data = {}
    # testb_data = {}
    train_data = []
    testa_data = []
    testb_data = []

    all_ids = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        current_doc_id = None
        current_annotations = []
        current_split = None

        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if line.startswith('-DOCSTART-'):
                if current_doc_id is not None:
                    doc_item = {
                        'id': current_doc_id,
                        'annotations': current_annotations
                    }

                    # if current_split == 'train':
                    #     train_data[current_doc_id] = doc_item
                    # elif current_split == 'testa':
                    #     testa_data[current_doc_id] = doc_item
                    # elif current_split == 'testb':
                    #     testb_data[current_doc_id] = doc_item

                    if current_split == 'train':
                        train_data.append(doc_item)
                    elif current_split == 'testa':
                        testa_data.append(doc_item)
                    elif current_split == 'testb':
                        testb_data.append(doc_item)

                match = re.search(r'\((.*?)\)', line)
                if not match:
                    continue

                id_part = match.group(1).split(' ')[0]

                if 'testa' in id_part:
                    current_split = 'testa'
                elif 'testb' in id_part:
                    current_split = 'testb'
                else:
                    current_split = 'train'
                
                current_doc_id = id_part

                if current_doc_id in all_ids:
                    raise ValueError(f"duplicate document ID found: {current_doc_id}")
                all_ids.add(current_doc_id)
                
                current_annotations = []
            
            elif line and current_doc_id is not None:
                parts = line.split('\t')
                
                token_data = {'token': parts[0]}

                if len(parts) >= 4:
                    token_data['bio_tag'] = parts[1]
                    token_data['mention'] = parts[2]
                    if parts[3] != 'null':
                        # token_data['yago_entity'] = parts[3]
                        raw_yago = parts[3]
                        decoded_yago = raw_yago.encode('utf-8').decode('unicode_escape')
                        token_data['yago_entity'] = decoded_yago
                
                if len(parts) >= 6:
                    if parts[4] != 'null':
                        token_data['wikipedia_url'] = parts[4]
                    if parts[5] != 'null':
                        token_data['numeric_id'] = parts[5]

                if len(parts) == 7:
                    if parts[6] != 'null':
                        token_data['freebase_mid'] = parts[6]
                
                current_annotations.append(token_data)

        if current_doc_id is not None:
            doc_item = {
                'id': current_doc_id,
                'annotations': current_annotations
            }

            # if current_split == 'train':
            #     train_data[current_doc_id] = doc_item
            # elif current_split == 'testa':
            #     testa_data[current_doc_id] = doc_item
            # elif current_split == 'testb':
            #     testb_data[current_doc_id] = doc_item

            if current_split == 'train':
                train_data.append(doc_item)
            elif current_split == 'testa':
                testa_data.append(doc_item)
            elif current_split == 'testb':
                testb_data.append(doc_item)


    return train_data, testa_data, testb_data


def parse_yago_ntx(filepath, is_label_file=False, target_lang='en'):

    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        
        reader = csv.reader(f, delimiter='\t', quotechar='"', escapechar='\\')
        
        for line in reader:
            
            if len(line) < 3: continue

            subj, pred, obj = line[0], line[1], line[2]

            subj = subj.strip('<>')
            pred = pred.strip('<>')
            
            if subj.startswith('<<'): continue

            if is_label_file:
                
                if 'label' not in pred and 'name' not in pred:
                    continue
                
                if '@' in obj:
                    text_part, lang_part = obj.rsplit('@', 1)
                    
                    if target_lang and lang_part != target_lang:
                        continue
                        
                    obj = text_part
                else:
                    pass

                obj = obj.strip('"')

            elif obj.startswith('<'):
                 obj = obj.strip('<>')

            yield subj, pred, obj

def prepare_yago_data(directory_name):
    nodes_info = {}
    print("loading labels...")
    for subj, pred, obj in parse_yago_ntx(os.path.join(directory_name, 'yago-wd-labels.nt.gz'), is_label_file=True):
        
        if subj not in nodes_info: 
            nodes_info[subj] = obj

    edges_info = []
    print("loading facts...")
    for subj, pred, obj in parse_yago_ntx(os.path.join(directory_name, 'yago-wd-facts.nt.gz')):
        
        if subj in nodes_info and obj in nodes_info:
            edges_info.append((subj, pred, obj))

    return nodes_info, edges_info

def store_graph_as_dataframe(
        nodes_info, 
        edges_info, 
        dataset_dir, 
        node_path, 
        edge_path,
        rel_path
    ):
    df_nodes = pd.DataFrame(list(nodes_info.items()), columns=['url', 'name'])
    df_edges = pd.DataFrame(edges_info, columns=['subject', 'predicate', 'object'])
    df_nodes = df_nodes.sort_values('url').reset_index(drop=True)
    unique_rels = sorted(df_edges['predicate'].unique())
    df_rels = pd.DataFrame(unique_rels, columns=['predicate'])
    df_nodes.to_feather(os.path.join(dataset_dir, node_path))
    df_edges.to_feather(os.path.join(dataset_dir, edge_path))
    df_rels.to_feather(os.path.join(dataset_dir, rel_path))

def load_graph_as_dataframe(
        dataset_dir, 
        node_path, 
        edge_path,
        rel_path
    ):
    df_nodes = pd.read_feather(os.path.join(dataset_dir, node_path))
    df_edges = pd.read_feather(os.path.join(dataset_dir, edge_path))
    df_rels = pd.read_feather(os.path.join(dataset_dir, rel_path))

    return df_nodes, df_edges, df_rels

def node_name(idx_to_node, n_id):
    return idx_to_node[n_id].split('/')[-1]

def edge_name(idx_to_rel, e_id):
    return idx_to_rel[e_id].split('/')[-1]

def prepare_graph_indices(df_nodes, df_rels):
    idx_to_node = df_nodes['url'].values 
    
    node_to_idx = pd.Series(
        data=df_nodes.index, 
        index=df_nodes['url']
    ).to_dict()
    
    idx_to_rel = df_rels['predicate'].values
    
    rel_to_idx = pd.Series(
        data=df_rels.index, 
        index=df_rels['predicate']
    ).to_dict()

    return node_to_idx, idx_to_node, rel_to_idx, idx_to_rel


def create_orphan_nodes(df_nodes, not_found, dataset_path, node_path):
    print(f'current nodes: {len(df_nodes)}')
    df_new = pd.DataFrame(list(not_found.items()), columns=['url', 'name'])
    if len(df_new) != 0:
        df_nodes = pd.concat([df_nodes, df_new])
        df_nodes = df_nodes.sort_values('url').reset_index(drop=True)
        df_nodes.to_feather(os.path.join(dataset_path, node_path))
    print(f'updated nodes: {len(df_nodes)}')
    return df_nodes

def store_pyg_tensors(df_nodes, df_edges, df_rels, dataset_dir, pyg_path):
    node_to_idx, idx_to_node, rel_to_idx, idx_to_rel = prepare_graph_indices(df_nodes, df_rels)

    src_list = []
    dst_list = []
    edge_type_list = []

    for src, rel, dst in df_edges.values:
        src_list.append(node_to_idx[src])
        dst_list.append(node_to_idx[dst])
        edge_type_list.append(rel_to_idx[rel])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)

    pyg_tensors = {
        "edge_index": edge_index,
        "edge_type": edge_type
    }

    save_file(pyg_tensors, os.path.join(dataset_dir, pyg_path))

def load_pyg_data(pyg_tensors, df_nodes):

    num_nodes = len(df_nodes)

    data = Data(
        edge_index=pyg_tensors["edge_index"],
        edge_attr=pyg_tensors["edge_type"],
        num_nodes=num_nodes
    )

    if data.edge_index is not None:
        data.edge_index, data.edge_attr = sort_edge_index(
            data.edge_index, 
            data.edge_attr, 
            sort_by_row=True
        )

    data.validate(raise_on_error=True)

    return data
