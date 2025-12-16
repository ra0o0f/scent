import os
from data.shadowlink import parse_shadowlink_dataset, shadowlink_post_process
from data.whitening import extract_embeddings, get_whitening_params
from data.wiki import enrich_yagograph_with_wikidata, read_wikidata
import torch
from data.yago import (
    aida_post_process,
    create_orphan_nodes,
    # get_aida_textual_content,
    parse_aida_dataset,
    prepare_graph_indices,
    prepare_yago_data,
    store_graph_as_dataframe,
    store_pyg_tensors,
)
from models.utils import get_lm_model


def store_whitening_stats(
    config,
):
    print("Whitening params is being calculated..")

    lm_model, tokenizer, lm_config = get_lm_model(config["model"]["lm_model_name"])

    aida_wikidata = read_wikidata(
        os.path.join(
            config["data"]["dataset_path"], config["data"]["aida_wikidata_path"]
        )
    )
    train_set, _, _ = parse_aida_dataset(config["data"]["aida_yago2_path"])

    # aida_contents = get_aida_textual_content(
    #     train_set=train_set, aida_wikidata=aida_wikidata
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm_model = lm_model.to(device)

    raw_embeddings = extract_embeddings(
        aida_wikidata, tokenizer=tokenizer, lm_model=lm_model, device=device
    )

    raw_embeddings = raw_embeddings.to(device)

    mu, W = get_whitening_params(raw_embeddings)

    torch.save(
        {"mu": mu, "W": W},
        os.path.join(
            config["data"]["dataset_path"], config["data"]["aida_whitening_params_path"]
        ),
    )

    print("Whitening params stored.")
    print("mu", mu.shape)
    print("W", W.shape)


def preprocess(args, config):

    dataset_path = config["data"]["dataset_path"]

    print("Preparing AIDA-YAGO2..")

    train_set, testa_set, testb_set = parse_aida_dataset(
        config["data"]["aida_yago2_path"]
    )

    print("Preparing YAGO4 facts and labels..")

    nodes_info, edges_info = prepare_yago_data(
        labels_path=config["data"]["yago4_labels_path"],
        facts_path=config["data"]["yago4_facts_path"],
    )

    print("Storing YAGO4 as feathers..")

    df_nodes, df_edges, df_rels = store_graph_as_dataframe(
        nodes_info=nodes_info,
        edges_info=edges_info,
        dataset_dir=dataset_path,
        node_path=config["data"]["node_path"],
        edge_path=config["data"]["edge_path"],
        rel_path=config["data"]["rel_path"],
    )

    node_to_idx, _, _, _ = prepare_graph_indices(df_nodes=df_nodes, df_rels=df_rels)

    print("Adding missing nodes from AIDA-YAGO2..")

    df_nodes = aida_post_process(
        train_set=train_set,
        testa_set=testa_set,
        testb_set=testb_set,
        df_nodes=df_nodes,
        node_to_idx=node_to_idx,
        dataset_path=dataset_path,
        node_path=config["data"]["node_path"],
        wikidata_path=config["data"]["wikidata_path"],
        aida_wikidata_path=config["data"]["aida_wikidata_path"],
    )

    print("Processing Shadowlink dataset..")

    top_data, shadow_data, tail_data = parse_shadowlink_dataset(
        config["data"]["shadowlink_path"]
    )

    node_to_idx, _, _, _ = prepare_graph_indices(df_nodes=df_nodes, df_rels=df_rels)

    df_nodes = shadowlink_post_process(
        top_data=top_data,
        shadow_data=shadow_data,
        tail_data=tail_data,
        df_nodes=df_nodes,
        node_to_idx=node_to_idx,
        dataset_path=dataset_path,
        wikidata_path=config["data"]["wikidata_path"],
        shadowlink_wikidata_path=config["data"]["shadowlink_wikidata_path"],
        node_path=config["data"]["node_path"],
    )

    print("Storing graph as COO tensors..")

    store_pyg_tensors(
        df_nodes=df_nodes,
        df_edges=df_edges,
        df_rels=df_rels,
        dataset_dir=dataset_path,
        pyg_path=config["data"]["pyg_path"],
    )

    print("Processing yagograph with wikidata..")

    node_to_idx, idx_to_node, _, _ = prepare_graph_indices(
        df_nodes=df_nodes, df_rels=df_rels
    )

    enrich_yagograph_with_wikidata(
        list(idx_to_node),
        wikidata_zip_path=config["data"]["wikidata_path"],
        output_path=os.path.join(
            config["data"]["dataset_path"], config["data"]["graph_yago_wiki_path"]
        ),
    )

    print("Preprocessing finished.")

    store_whitening_stats(config)
