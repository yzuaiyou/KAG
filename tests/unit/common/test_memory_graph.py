# -*- coding: utf-8 -*-
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

from kag.common.graphstore.memory_graph import MemoryGraph

def test_memory_graph():
    subgraph = {
        "resultNodes": [
            {'id': '0', 'label': 'Person', 'name': 'A', 'properties': {'_name_vector': [1.0, 0.0]}},
            {'id': '1', 'label': 'Person', 'name': 'B', 'properties': {'_name_vector': [0.0, 1.0]}},
            {'id': '2', 'label': 'Person', 'name': 'C', 'properties': {'_name_vector': [-1.0, 0.0]}},
        ],
        "resultEdges": [
            {'from': '0', 'fromType': 'Person', 'id': 101, 'label': 'knows', 'properties': {}, 'to': '1', 'toType': 'Person'},
            {'from': '0', 'fromType': 'Person', 'id': 102, 'label': 'knows', 'properties': {}, 'to': '2', 'toType': 'Person'},
        ],
    }

    subgraph2 = {
        "resultNodes": [
            {'id': '3', 'label': 'Person', 'name': 'D', 'properties': {'_name_vector': [0.0, -1.0]}},
        ],
        "resultEdges": [
            {'from': '1', 'fromType': 'Person', 'id': 201, 'label': 'knows', 'properties': {}, 'to': '3', 'toType': 'Person'},
            {'from': '2', 'fromType': 'Person', 'id': 202, 'label': 'knows', 'properties': {}, 'to': '3', 'toType': 'Person'},
        ],
    }

    memory_graph = MemoryGraph("graph_store.pkl.gz", vectorizer=None)
    with memory_graph.graph_store_lock:
        memory_graph.upsert_subgraph(subgraph)
        memory_graph.upsert_subgraph(subgraph2)
        memory_graph.flush()

    print(memory_graph._backend_graph)

    e = memory_graph.get_entity("1", "Person")
    print(e)
    assert e.name == "B"

    d = memory_graph.get_one_hop_graph("1", "Person")
    print(d)
    assert len(d.in_relations) == 1
    assert len(d.out_relations) == 1

    h = memory_graph.calculate_pagerank_scores(None, [])
    print(h)
    assert len(h) == 4

    r = memory_graph.vector_search("Person", "name", [0.866, 0.5])
    print(r)
    assert len(r) == 4
