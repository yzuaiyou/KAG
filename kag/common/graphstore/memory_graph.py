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

import os
import re
from typing import Dict, List

import filelock

from kag.solver.logic.core_modules.common.one_hop_graph import EntityData, RelationData, OneHopGraphData
from kag.solver.tools.graph_api.model.table_model import TableData


class MemoryGraph:
    def __init__(self, graph_store_path, vectorizer):
        """
        Initialize the MemoryGraph instance.

        :param graph_store_path: path of the memory graph storage file
        :param vectorizer: vectorizer, to turn strings into vectors
        """
        self._graph_store_path = graph_store_path
        self._graph_store_lock_path = graph_store_path + ".lock"
        self._graph_store_lock = filelock.FileLock(self._graph_store_lock_path)
        self._backend_graph = None
        self._vectorizer = vectorizer
        self.init_graph()

    @property
    def graph_store_lock(self):
        return self._graph_store_lock

    def init_graph(self):
        """
        Initialize the MemoryGraph instance from storage file
        """
        import igraph as ig

        with self._graph_store_lock:
            if os.path.isfile(self._graph_store_path):
                self._backend_graph = ig.Graph.Read(self._graph_store_path, "picklez")
            else:
                self._backend_graph = ig.Graph(directed=True)

    def get_entity(self, biz_id, label, **kwargs) -> EntityData:
        """
        Get data of the specified entity.

        :param biz_id: entity business id。
        :param label: entity label
        :return: data of the specified entity
        """
        vertex = self._get_vertex(biz_id, label)
        entity = self._create_entity_from_vertex(vertex, biz_id, label)
        return entity

    def get_one_hop_graph(self, biz_id, label) -> OneHopGraphData:
        """
        Get one-hop graph of the specified entity.

        :param biz_id: entity business id。
        :param label: entity label
        :return: one-hop graph of the specified entity
        """
        start_vertex = self._get_vertex(biz_id, label)
        start_entity = self._create_entity_from_vertex(start_vertex, biz_id, label)
        one_hop = OneHopGraphData(None, "s")
        one_hop.s = start_entity
        in_edges = start_vertex.in_edges()
        for in_edge in in_edges:
            source_entity = self._create_entity_from_vertex(in_edge.source_vertex)
            in_relation = self._create_relation_from_edge(in_edge, source_entity, start_entity)
            one_hop.in_relations.setdefault(in_relation.type, []).append(in_relation)
        out_edges = start_vertex.out_edges()
        for out_edge in out_edges:
            target_entity = self._create_entity_from_vertex(out_edge.target_vertex)
            out_relation = self._create_relation_from_edge(out_edge, start_entity, target_entity)
            one_hop.out_relations.setdefault(out_relation.type, []).append(out_relation)
        return one_hop

    def _get_vertex(self, biz_id, label):
        try:
            vertex = self._backend_graph.vs.find(id=biz_id, label=label)
        except (KeyError, ValueError):
            vertex = None
        if vertex is None:
            message = f"no such entity {label} {biz_id}"
            raise ValueError(message)
        return vertex

    @staticmethod
    def _create_entity_from_vertex(vertex, biz_id=None, label=None) -> EntityData:
        attributes = vertex.attributes()
        entity = EntityData()
        entity.prop = None
        entity.biz_id = attributes.get("id", biz_id)
        entity.name = attributes.get("name", "")
        entity.description = attributes.get("description", "")
        entity.type = attributes.get("label", label)
        entity.type_zh = None
        entity.score = 1.0
        return entity

    @staticmethod
    def _create_relation_from_edge(edge, from_entity: EntityData, end_entity: EntityData) -> RelationData:
        attributes = edge.attributes()
        relation = RelationData()
        relation.prop = None
        relation.from_id = from_entity.biz_id
        relation.end_id = end_entity.biz_id
        relation.from_entity = from_entity
        relation.from_type = from_entity.type
        relation.from_alias = "s"
        relation.end_type = end_entity.type
        relation.end_entity = end_entity
        relation.end_alias = "o"
        relation.type = attributes.get("label")
        relation.type_zh = None
        return relation

    def execute_dsl(self, dsl, **kwargs) -> TableData:
        """
        Execute DSL query statement.

        :param dsl: the query statement
        :param kwargs: other optional arguments
        :return: query result data as TableData
        """
        raise NotImplementedError

    def calculate_pagerank_scores(self, target_vertex_type, start_nodes: List[Dict], **kwargs) -> Dict:
        """
        Calculate PageRank scores.

        :param target_vertex_type: target vertex type (not used currently)
        :param start_nodes: list of start nodes; a start node is a dictionary (not used currently)
        :param kwargs: other optional arguments
        :return: result as a dictionary mapping node ids to scores
        """
        scores = self._backend_graph.personalized_pagerank(**kwargs)
        node_ids = self._backend_graph.vs.get_attribute_values("id")
        result = dict(zip(node_ids, scores))
        return result

    def vector_search(self, label, property_key, query_vector: list, topk=10, **kwargs):
        """
        Execute vector searching.

        :param label: entity label
        :param property_key: property key to search
        :param query_vector: the query vector (a list of float)
        :param topk: number of entities to return; default tot 10
        :param kwargs: other optional arguments
        """
        import torch

        if label == "Entity":
            nodes = self._backend_graph.vs
        else:
            try:
                nodes = self._backend_graph.vs.select(label=label)
            except (KeyError, ValueError):
                return []

        vector_field_name = self._get_vector_field_name(property_key)
        vectors = nodes.get_attribute_values(vector_field_name)
        filtered_nodes = []
        filtered_vectors = []
        for node, vector in zip(nodes, vectors):
            if vector is not None:
                filtered_nodes.append(node)
                filtered_vectors.append(vector)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        filtered_vectors = torch.tensor(filtered_vectors).to(device)
        if isinstance(query_vector, str):
            query_vector = self._vectorizer.vectorize(query_vector)
        query_vector = torch.tensor(query_vector).unsqueeze(1).to(device)
        cosine_similarity = filtered_vectors @ query_vector
        scores = 0.5 * cosine_similarity + 0.5

        top_data = scores.topk(k=min(topk, len(scores)), dim=0)
        top_indices = top_data.indices.to("cpu")
        top_values = top_data.values.to("cpu")
        items = []
        for index, score in zip(top_indices, top_values):
            node = nodes[index.item()]
            node_attributes = node.attributes()
            node_attributes["__labels__"] = [node_attributes.pop("label")]
            items.append({"node": node_attributes, "score": score.item()})
        return items

    @staticmethod
    def _get_vector_field_name(property_key: str) -> str:
        name = f"{property_key}_vector"
        name =  MemoryGraph._to_snake_case(name)
        return "_" + name

    @staticmethod
    def _to_snake_case(name: str) -> str:
        words = re.findall("[A-Za-z][a-z0-9]*", name)
        result = "_".join(words).lower()
        return result

    def text_search(self, label, property_key, query_string: str, topk=10, **kwargs):
        """
        Execute vector searching.

        :param label: entity label
        :param property_key: property key
        :param query_string: query text
        :param topk: number of entities to return; default tot 10
        :param kwargs: other optional arguments
        """
        return []

    def upsert_subgraph(self, subgraph: Dict):
        """
        Upsert the subgraph with the memory graph.

        The subgraph is a dictionary create by `kag.builder.model.sub_graph.SubGraph.to_dict()`.

        :param subgraph: subgraph to upsert
        """
        def update_vertex_attributes(v, n):
            v.update_attributes(n["properties"], id=n["id"], name=n["name"], label=n["label"])

        def update_edge_attributes(e, a):
            e.update_attributes(a["properties"],
                                id=a["id"], label=a["label"],
                                from_id=a["from"], from_type=a["fromType"],
                                to_id=a["to"], to_type=a["toType"])

        fresh_nodes = []
        for node in subgraph["resultNodes"]:
            try:
                vertex = self._backend_graph.vs.find(id=node["id"])
            except (KeyError, ValueError):
                fresh_nodes.append(node)
                continue
            update_vertex_attributes(vertex, node)
        old_num_vertices = len(self._backend_graph.vs)
        self._backend_graph.add_vertices(len(fresh_nodes))
        for k, node in enumerate(fresh_nodes):
            vertex = self._backend_graph.vs[old_num_vertices + k]
            update_vertex_attributes(vertex, node)

        fresh_arcs = []
        for arc in subgraph["resultEdges"]:
            try:
                edge = self._backend_graph.es.find(id=arc["id"])
            except (KeyError, ValueError):
                fresh_arcs.append(arc)
                continue
            update_edge_attributes(edge, arc)
        old_num_edges = len(self._backend_graph.es)
        fresh_edges = []
        for arc in fresh_arcs:
            source = self._backend_graph.vs.find(id=arc["from"])
            target = self._backend_graph.vs.find(id=arc["to"])
            fresh_edges.append((source, target))
        self._backend_graph.add_edges(fresh_edges)
        for k, arc in enumerate(fresh_arcs):
            edge = self._backend_graph.es[old_num_edges + k]
            update_edge_attributes(edge, arc)

    def flush(self):
        """
        Flush the memory graph to disk file.
        """
        with self._graph_store_lock:
            self._backend_graph.write(self._graph_store_path, "picklez")
