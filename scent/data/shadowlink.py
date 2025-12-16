import json
import yaml
import os

from .yago import create_orphan_nodes

from .wiki import enrich_with_wikidata


def parse_shadowlink_dataset(shadowlink_path):

    tail_path = os.path.join(shadowlink_path, "Tail.json")
    shadow_path = os.path.join(shadowlink_path, "Shadow.json")
    top_path = os.path.join(shadowlink_path, "Top.json")

    with open(tail_path, "r") as f:
        tail_data = json.load(f)
    with open(shadow_path, "r") as f:
        shadow_data = json.load(f)
    with open(top_path, "r") as f:
        top_data = json.load(f)

    return top_data, shadow_data, tail_data


def shadowlink_post_process(
    top_data,
    shadow_data,
    tail_data,
    df_nodes,
    node_to_idx,
    dataset_path,
    wikidata_path,
    shadowlink_wikidata_path,
    node_path,
):

    wiki_data = {}
    shadow_dataset = top_data + shadow_data + tail_data

    not_found = {}

    for item in shadow_dataset:
        numeric_id = str(item["wiki_id"])

        yago_id = (
            f"http://yago-knowledge.org/resource/{item['entity_name'].replace(' ','_')}"
        )

        if yago_id not in node_to_idx:
            not_found[yago_id] = item["entity_name"]

        if numeric_id not in wiki_data:
            wiki_data[numeric_id] = {
                "numeric_id": numeric_id,
                "yago_id": yago_id,
                "name": item["entity_name"],
            }

    df_nodes = create_orphan_nodes(
        not_found=not_found,
        df_nodes=df_nodes,
        dataset_path=dataset_path,
        node_path=node_path,
    )

    print("Preparing wikidata for Shadowlink..")
    enrich_with_wikidata(wiki_data=wiki_data, wikidata_path=wikidata_path)

    with open(
        os.path.join(dataset_path, shadowlink_wikidata_path), "w", encoding="utf-8"
    ) as out_f:
        for item in wiki_data.values():
            out_f.write(json.dumps(item) + "\n")

    return df_nodes
