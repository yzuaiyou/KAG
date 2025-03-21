# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.

import os
import json
import hashlib
import shutil
import random

import chromadb

import pandas as pd
from neo4j import GraphDatabase

from kag.builder.runner import BuilderChainRunner
from kag.common.conf import KAG_CONFIG


def load_finqa_data() -> map:
    """
    load data
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, "data", "test.json")
    with open(file_name, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print("finqa data list len " + str(len(data_list)))
    for _idx, data in enumerate(data_list):
        data["index"] = _idx

    file_to_qa_map = {}
    for data in data_list:
        finqa_filename = data["filename"]
        if finqa_filename not in file_to_qa_map:
            file_to_qa_map[finqa_filename] = []
        file_to_qa_map[finqa_filename].append(data)
    return file_to_qa_map


def split_to_list(txt_list, limit_len) -> list:
    rst = []
    tmp = ""
    for l in txt_list:
        if len(tmp) < limit_len:
            tmp += l
            continue
        rst.append(tmp)
        tmp = ""
    if len(tmp) > 0:
        rst.append(tmp)
    return rst


def convert_data(item, limitlen) -> list:
    prev_text_list = item["pre_text"]
    prev_text_list = [s for s in prev_text_list if s != "."]

    post_text_list = item["post_text"]
    post_text_list = [s for s in post_text_list if s != "."]

    table_row_list = item["table"]
    columns = table_row_list[0]
    data = table_row_list[1:]
    table_df = pd.DataFrame(data=data, columns=columns)
    table_md_str = table_df.to_markdown(index=False)

    return (
        split_to_list(prev_text_list, limitlen)
        + [table_md_str]
        + split_to_list(post_text_list, limitlen)
    )


def save_data(item, collection):
    file_name = item["filename"]
    print(file_name)
    chunk_len = 500
    documents = convert_data(item, chunk_len)
    documents = [f"{i+1}. {d}" for i, d in enumerate(documents)]
    metadatas = [{"file_name": f"{file_name}_{chunk_len}"} for doc in documents]
    ids = [f"{file_name}_{chunk_len}_{i}" for i, _ in enumerate(documents)]

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chromadb_path = os.path.join(current_dir, "chunk_chromadb")
    os.makedirs(chromadb_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    collection = chroma_client.create_collection(name="chunk", get_or_create=True)

    _finqa_file_to_qa_map = load_finqa_data()
    for file_name, _item_list in _finqa_file_to_qa_map.items():
        save_data(_item_list[0], collection)
