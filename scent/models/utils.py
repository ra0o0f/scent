from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AutoConfig,
    PreTrainedModel,
)


def get_lm_model(lm_model_name):
    lm_config = AutoConfig.from_pretrained(lm_model_name)
    tokenizer = RobertaTokenizer.from_pretrained(lm_model_name)
    lm_model = RobertaForMaskedLM.from_pretrained(lm_model_name, config=lm_config)

    return lm_model, tokenizer, lm_config


def move_batch_to_device(batch, device, graph_training=True):
    for key in ["input_ids", "attention_mask", "labels", "node_ids"]:
        if key in batch:
            batch[key] = batch[key].to(device)

    if graph_training:
        for key in [
            "node_num",
            "edge_num",
            "edge_index",
            "edge_graph_idx",
            "n_id",
            "e_id",
            "graph_offsets",
        ]:
            if key in batch:
                batch[key] = batch[key].to(device)

        if "node_feature_batch" in batch:
            batch["node_feature_batch"]["input_ids"] = batch["node_feature_batch"][
                "input_ids"
            ].to(device)
            batch["node_feature_batch"]["attention_mask"] = batch["node_feature_batch"][
                "attention_mask"
            ].to(device)

        if "edge_feature_batch" in batch:
            batch["edge_feature_batch"]["input_ids"] = batch["edge_feature_batch"][
                "input_ids"
            ].to(device)
            batch["edge_feature_batch"]["attention_mask"] = batch["edge_feature_batch"][
                "attention_mask"
            ].to(device)

    return batch
