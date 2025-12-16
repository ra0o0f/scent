import zipfile
import json
import io
import re
import pandas as pd
import os


def get_first_sentence(text):

    if not text:
        return ""
    # lookbehind for abbreviations (e.g., U.S., Mr.) to prevent false splits
    split_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
    parts = re.split(split_pattern, text, maxsplit=1)
    return parts[0]


def enrich_yagograph_with_wikidata(yago_urls, wikidata_zip_path, output_path):

    slug_to_url = {url.split("/")[-1]: url for url in yago_urls}

    results_map = {}

    with zipfile.ZipFile(wikidata_zip_path, "r") as z:
        for filename in z.namelist():
            if not filename.startswith("enwiki"):
                continue

            print(f"Processing {filename}...")

            with z.open(filename) as f:
                with io.TextIOWrapper(f, encoding="utf-8") as text_file:
                    for line in text_file:
                        try:
                            data = json.loads(line)

                            wiki_url = data.get("url", "")
                            if not wiki_url:
                                continue

                            wiki_slug = wiki_url.split("/")[-1]

                            if wiki_slug in slug_to_url:
                                full_yago_url = slug_to_url[wiki_slug]

                                abstract = data.get("abstract", "")
                                short_abstract = get_first_sentence(abstract)

                                results_map[full_yago_url] = {
                                    "yago_url": full_yago_url,
                                    "wiki_id": data.get("identifier"),
                                    "name": data.get("name"),
                                    "short_abstract": short_abstract,
                                }

                        except json.JSONDecodeError:
                            continue

    df = pd.DataFrame(list(results_map.values()))
    df.set_index("yago_url", inplace=True)
    df.to_feather(output_path)


def load_yagograph_with_wikidata(path):
    return pd.read_feather(path)


def enrich_with_wikidata(wiki_data, wikidata_path):

    with zipfile.ZipFile(wikidata_path, "r") as z:

        for filename in z.namelist():

            if not filename.startswith("enwiki"):
                continue

            print(f"Processing {filename}..")

            with z.open(filename) as f:
                with io.TextIOWrapper(f, encoding="utf-8") as text_file:

                    for i, line in enumerate(text_file):

                        data = json.loads(line)

                        name = data.get("name")
                        identifier = str(data.get("identifier"))

                        if identifier not in wiki_data:
                            continue

                        abstract = data.get("abstract", "")
                        short_abstract = get_first_sentence(abstract)

                        if name:
                            wiki_data[identifier]["name"] = name
                        wiki_data[identifier]["short_abstract"] = short_abstract


def read_wikidata(jsonl_path):
    data_list = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data_list[item["numeric_id"]] = item

    return data_list
