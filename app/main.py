import os
import yaml
import argparse
from pathlib import Path


from scent.data.yago import create_orphan_nodes, parse_aida_dataset, prepare_graph_indices, prepare_yago_data, store_graph_as_dataframe, store_pyg_tensors



parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='config path',
    default='./configs/config.yaml')
parser.add_argument(
    '--mode', type=str,
    help='script mode',
    required=True)

def main(args, config):

    dataset_path = config['data']['dataset_path']

    if args.mode == 'preprocess':

        print('Preparing AIDA-YAGO2..')

        train_set, testa_set, testb_set = parse_aida_dataset(config['data']['aida_yago2_path'])

        print('Preparing YAGO4 facts and labels..')

        nodes_info, edges_info = prepare_yago_data(
                labels_path=config['data']['yago4_labels_path'],
                facts_path=config['data']['yago4_facts_path']
            )
        
        print('Storing YAGO4 as feathers..')

        df_nodes, df_edges, df_rels = store_graph_as_dataframe(
            nodes_info=nodes_info,
            edges_info=edges_info,
            dataset_dir=dataset_path,
            node_path=config['data']['node_path'],
            edge_path=config['data']['edge_path'],
            rel_path=config['data']['rel_path']
        )

        node_to_idx, _, _, _ = prepare_graph_indices(
            df_nodes=df_nodes,
            df_rels=df_rels
        )

        print('Adding missing nodes from AIDA-YAGO2..')

        df_nodes = create_orphan_nodes(
            aida_dataset=train_set+testa_set+testb_set,
            df_nodes=df_nodes,
            node_to_idx=node_to_idx,
            dataset_path=dataset_path,
            node_path=config['data']['node_path']
        )
        
        print('Storing graph as COO tensors..')

        store_pyg_tensors(
            df_nodes=df_nodes, 
            df_edges=df_edges, 
            df_rels=df_rels, 
            dataset_dir=dataset_path, 
            pyg_path=config['data']['pyg_path']
        )

        print('Preprocessing finished.')




if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    main(args, config)
