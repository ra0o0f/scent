import torch
import os
from models.scent_model import ScentEncoder
from models.utils import get_lm_model
from data.yago import (
    load_graph_as_dataframe,
    load_pyg_data,
    parse_aida_dataset,
    postproces_pyg_data,
    prepare_graph_indices,
)
from data.wiki import load_yagograph_with_wikidata, read_wikidata
from data.shadowlink import parse_shadowlink_dataset
from data.scent_data import ScentDataCollator, ScentDataset, ShadowScentDataset
from models.scent_inference import ScentInference


def evaluate(args, config):

    eval_mode = args.eval_mode
    eval_checkpoints = args.eval_mode
    graph_training = args.graph_training

    device = torch.device(config["train"]["device"])
    mask_buffer_len = config["eval"]["mask_buffer_len"]
    max_seq_len = config["eval"]["max_seq_len"]
    graph_num_neighbors = [10, 3]

    batch_size = config["eval"]["batch_size"]
    num_workers = config["eval"]["num_workers"]

    whitening_stats = torch.load(
        os.path.join(
            config["data"]["dataset_path"], config["data"]["aida_whitening_params_path"]
        )
    )

    checkpoint = torch.load(eval_checkpoints, map_location=device)

    ### prepare data

    train_set, testa_set, testb_set = parse_aida_dataset(
        config["data"]["aida_yago2_path"]
    )
    df_nodes, df_edges, df_rels = load_graph_as_dataframe(
        dataset_dir=config["data"]["dataset_path"],
        node_path=config["data"]["node_path"],
        edge_path=config["data"]["edge_path"],
        rel_path=config["data"]["rel_path"],
    )
    pyg_data = load_pyg_data(
        dataset_dir=config["data"]["dataset_path"],
        df_nodes=df_nodes,
        pyg_path=config["data"]["pyg_path"],
    )

    postproces_pyg_data(pyg_data)

    node_to_idx, idx_to_node, rel_to_idx, idx_to_rel = prepare_graph_indices(
        df_nodes=df_nodes, df_rels=df_rels
    )

    yagograph_wikidata = load_yagograph_with_wikidata(
        os.path.join(
            config["data"]["dataset_path"], config["data"]["graph_yago_wiki_path"]
        )
    )

    if eval_mode == "aida":

        train_set, testa_set, testb_set = parse_aida_dataset(
            config["data"]["aida_yago2_path"]
        )
        wikidata = read_wikidata(
            os.path.join(
                config["data"]["dataset_path"], config["data"]["aida_wikidata_path"]
            )
        )

        dataset = ScentDataset(
            aida_data=train_set,
            node_to_idx=node_to_idx,
            tokenizer=tokenizer,
            mask_buffer_len=mask_buffer_len,
            max_seq_len=max_seq_len,
        )

        test_dataset = ScentDataset(
            aida_data=testa_set + testb_set,
            node_to_idx=node_to_idx,
            tokenizer=tokenizer,
            mask_buffer_len=mask_buffer_len,
            max_seq_len=max_seq_len,
        )

        aida_candidate_set = dataset.candidate_set | test_dataset.candidate_set

        datacollator_wikidata = wikidata
        inference_candidate_set = aida_candidate_set

    if eval_mode == "shadowlink":

        top_data, shadow_data, tail_data = parse_shadowlink_dataset(
            config["data"]["shadowlink_path"]
        )

        shadowlink_wikidata = read_wikidata(
            os.path.join(
                config["data"]["dataset_path"],
                config["data"]["shadowlink_wikidata_path"],
            )
        )

        shadowlink_top_ds = ShadowScentDataset(
            data=top_data,
            wiki_to_yago_map=shadowlink_wikidata,
            node_to_idx=node_to_idx,
            tokenizer=tokenizer,
            mask_buffer_len=mask_buffer_len,
        )

        shadowlink_shadow_ds = ShadowScentDataset(
            data=shadow_data,
            wiki_to_yago_map=shadowlink_wikidata,
            node_to_idx=node_to_idx,
            tokenizer=tokenizer,
            mask_buffer_len=mask_buffer_len,
        )

        shadowlink_tail_ds = ShadowScentDataset(
            data=tail_data,
            wiki_to_yago_map=shadowlink_wikidata,
            node_to_idx=node_to_idx,
            tokenizer=tokenizer,
            mask_buffer_len=mask_buffer_len,
        )

        shadowlink_candidate_set = (
            shadowlink_top_ds.candidate_set
            | shadowlink_shadow_ds.candidate_set
            | shadowlink_tail_ds.candidate_set
        )

        datacollator_wikidata = shadowlink_wikidata
        inference_candidate_set = shadowlink_candidate_set

    ### eval

    lm_model, tokenizer, lm_config = get_lm_model(config["model"]["lm_model_name"])

    scent_encoder = ScentEncoder(
        config=lm_config,
        lm_model=lm_model,
        tokenizer=tokenizer,
        graph_training=graph_training,
        whitening_stats=whitening_stats,
    ).to(device)

    scent_encoder.load_state_dict(checkpoint["model_state_dict"])

    collate_fn = ScentDataCollator(
        tokenizer,
        sample_graph_nodes=graph_training,
        pyg_data=pyg_data,
        idx_to_node=idx_to_node,
        idx_to_rel=idx_to_rel,
        graph_num_neighbors=graph_num_neighbors,
        yagograph_wikidata=yagograph_wikidata,
        wikidata=datacollator_wikidata,
    )

    scent_inference = ScentInference(
        model=scent_encoder,
        tokenizer=tokenizer,
        idx_to_node=idx_to_node,
        node_to_idx=node_to_idx,
        graph_collator=collate_fn,
        mask_buffer_len=mask_buffer_len,
        batch_size=batch_size,
        device=device,
        candidate_set=inference_candidate_set,
    )

    scent_inference.build_graph_representations()
    scent_inference.tokenize_candidates()

    if eval_mode == "aida":
        print("Evaluating AIDA-YAGO2 Test dataset..")
        aida_acc = scent_inference.evaluate(test_dataset, top_k=1, return_logs=False)

    if eval_mode == "shadowlink":
        print("Evaluating Shadowlink Top dataset..")
        top_acc = scent_inference.evaluate(
            shadowlink_top_ds, top_k=1, return_logs=False
        )

        print("Evaluating Shadowlink Shadow dataset..")
        shadow_acc = scent_inference.evaluate(
            shadowlink_shadow_ds, top_k=1, return_logs=False
        )

        print("Evaluating Shadowlink Tail dataset..")
        tail_acc = scent_inference.evaluate(
            shadowlink_tail_ds, top_k=1, return_logs=False
        )
