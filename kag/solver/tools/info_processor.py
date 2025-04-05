import json
import logging
import re
from enum import Enum
from typing import List

from kag.interface.solver.base_model import LFPlan, SubQueryResult
from kag.solver.logic.core_modules.common.one_hop_graph import KgGraph, EntityData
from knext.common.rest import ApiClient, Configuration
from knext.reasoner.rest.models.ca_pipeline import CaPipeline
from knext.reasoner.rest.models.data_edge import DataEdge
from knext.reasoner.rest.models.data_node import DataNode
from knext.reasoner.rest.models.edge import Edge
from knext.reasoner.rest.models.node import Node
from knext.reasoner.rest.models.report_pipeline_request import ReportPipelineRequest
from knext.reasoner.rest.models.sub_graph import SubGraph
from knext.reasoner.rest.reasoner_api import ReasonerApi

logger = logging.getLogger(__name__)

# `ReporterIntermediateProcessTool` 类用于在推理管道中报告中间过程。以下是该类的详细说明，包括其属性、构造函数和主要方法。

# ### 类的作用
# `ReporterIntermediateProcessTool` 类的主要作用是：
# - 报告推理管道中的中间过程和节点状态。
# - 管理节点的创建、状态更新以及报告整个管道的结构。

# ### 类的属性
# - `STATE (Enum)`: 枚举类，表示管道中节点的可能状态，包括 `WAITING`, `RUNNING`, `FINISH`, `ERROR`。
# - `ROOT_ID (int)`: 根节点的 ID，初始值为 0。
# - `report_log (bool)`: 是否报告日志。
# - `task_id (str)`: 任务 ID。
# - `project_id (str)`: 项目 ID。
# - `client (ReasonerApi)`: 与推理器交互的 API 客户端。
# - `cur_node_id (int)`: 当前节点的 ID。
# - `last_sub_question_size (int)`: 上一个子问题列表的大小。
# - `sub_query_node (list)`: 子查询节点的列表。
# - `start_node_id (int)`: 起始节点的 ID。
# - `create_pipeline_times (int)`: 创建管道的次数。
# - `language (str)`: 输出消息的语言。

# ### 构造函数
# ```python
# def __init__(self, report_log=False, task_id=None, project_id=None, host_addr=None, language="en"):
# ```
# - `report_log (bool)`: 是否报告日志。
# - `task_id (str)`: 任务 ID。
# - `project_id (str)`: 项目 ID。
# - `host_addr (str)`: API 客户端的主机地址。
# - `language (str)`: 输出消息的语言，默认为英文。

# ### 方法
# 1. `get_start_node_name()`
#    - 功能：根据当前管道创建次数获取起始节点的名称。
#    - 输出：起始节点的名称（字符串）。

# 2. `get_end_node_name()`
#    - 功能：获取结束节点的名称。
#    - 输出：结束节点的名称（字符串）。

# 3. `get_sub_question_name(index)`
#    - 功能：获取子问题节点的名称。
#    - 输入：子问题的索引（整数）。
#    - 输出：子问题节点的名称（字符串）。

# 4. `report_pipeline(question, rewrite_question_list=[])`
#    - 功能：报告整个管道，包括节点和边。
#    - 输入：
#      - `question (str)`: 原始问题。
#      - `rewrite_question_list (List[LFPlan])`: 重写问题的列表。

# 5. `report_final_answer(query, answer, state)`
#    - 功能：报告最终答案。
#    - 输入：
#      - `query (str)`: 查询。
#      - `answer (str)`: 答案。
#      - `state (STATE)`: 节点的状态。

# 6. `report_node(req_id, index, state, node_plan, kg_graph)`
#    - 功能：报告管道中的单个节点。
#    - 输入：
#      - `req_id (str)`: 请求 ID。
#      - `index (int)`: 节点索引。
#      - `state (STATE)`: 节点的状态。
#      - `node_plan (LFPlan)`: 节点的逻辑形式计划。
#      - `kg_graph (KgGraph)`: 与节点相关的知识图。

# 7. `_convert_lf_res_to_report_format(req_id, index, state, res, kg_graph)`
#    - 功能：将逻辑形式结果转换为报告格式。
#    - 输入：
#      - `req_id (str)`: 请求 ID。
#      - `index (int)`: 节点索引。
#      - `state (STATE)`: 节点的状态。
#      - `res (SubQueryResult)`: 逻辑形式查询的结果。
#      - `kg_graph (KgGraph)`: 与节点相关的知识图。
#    - 输出：子答案、上下文内容和子图（元组）。

# 8. `_convert_spo_to_graph(graph_id, spo_retrieved)`
#    - 功能：将 SPO 三元组转换为图形表示。
#    - 输入：
#      - `graph_id (str)`: 图形 ID。
#      - `spo_retrieved (list)`: 检索到的 SPO 三元组列表。
#    - 输出：子图（SubGraph）。

# 9. `_update_sub_question_recall_docs(docs)`
#    - 功能：用检索到的文档更新子问题的上下文。
#    - 输入：
#      - `docs (list)`: 检索到的文档列表。
#    - 输出：更新后的上下文内容（列表）。

# 10. `format_logs(logs)`
#     - 功能：将日志格式化为字符串。
#     - 输入：
#       - `logs (list or str)`: 要格式化的日志。
#     - 输出：格式化的日志内容（字符串）。

class ReporterIntermediateProcessTool:
    """
    A tool for reporting intermediate processes in a reasoning pipeline.

    Attributes:
        STATE (Enum): An enumeration of possible states for nodes in the pipeline.
        ROOT_ID (int): The root node ID.
        report_log (bool): Whether to report logs.
        task_id (str): The task ID.
        project_id (str): The project ID.
        client (ReasonerApi): API client for interacting with the reasoner.
        cur_node_id (int): Current node ID.
        last_sub_question_size (int): Size of the last sub-question list.
        sub_query_node (list): List of sub-query nodes.
        start_node_id (int): Starting node ID.
        create_pipeline_times (int): Number of times the pipeline has been created.
        language (str): Language for output messages.
    """

    class STATE(str, Enum):
        """Enumeration of possible states for nodes in the pipeline."""

        WAITING = "WAITING"
        RUNNING = "RUNNING"
        FINISH = "FINISH"
        ERROR = "ERROR"

    ROOT_ID = 0

    def __init__(
        self,
        report_log=False,
        task_id=None,
        project_id=None,
        host_addr=None,
        language="en",
    ):
        """
        Initialize the ReporterIntermediateProcessTool.

        Args:
            report_log (bool): Whether to report logs.
            task_id (str): The task ID.
            project_id (str): The project ID.
            host_addr (str): Host address for the API client.
            language (str): Language for output messages.
        """
        self.report_log = report_log
        self.task_id = task_id
        self.project_id = project_id
        self.client: ReasonerApi = ReasonerApi(
            api_client=ApiClient(configuration=Configuration(host=host_addr))
        )
        self.cur_node_id = self.ROOT_ID
        self.last_sub_question_size = self.ROOT_ID
        self.sub_query_node = []
        self.start_node_id = 1
        self.create_pipeline_times = 0
        self.language = language

    def get_start_node_name(self):
        """
        Get the name for the start node based on the current pipeline creation count.

        Returns:
            str: Name for the start node.
        """
        start_node_name = "问题" if self.language == "zh" else "Question"
        if self.create_pipeline_times != 0:
            start_node_name = (
                "反思问题" if self.language == "zh" else "Reflective Questioning"
            )
        return start_node_name

    def get_end_node_name(self):
        """
        Get the name for the end node.

        Returns:
            str: Name for the end node.
        """
        return "问题答案" if self.language == "zh" else "Answer"

    def get_sub_question_name(self, index):
        """
        Get the name for a sub-question node.

        Args:
            index (int): Index of the sub-question.

        Returns:
            str: Name for the sub-question node.
        """
        return f"子问题{index}" if self.language == "zh" else f"Sub Question {index}"

    def report_pipeline(self, question, rewrite_question_list: List[LFPlan] = []):
        """
        Report the entire pipeline including nodes and edges.

        Args:
            question (str): The original question.
            rewrite_question_list (List[LFPlan]): List of rewritten questions.
        """
        pipeline = CaPipeline()
        pipeline.nodes = []
        pipeline.edges = []
        self.cur_node_id += self.last_sub_question_size
        rethink_question = question
        # print(question)
        for idx, item in enumerate(rewrite_question_list, start=self.cur_node_id + 2):
            item.id = self.cur_node_id + idx

        if len(self.sub_query_node) == 0:
            end_node = Node(
                id=self.ROOT_ID,
                state=self.STATE.WAITING,
                question=rethink_question,
                answer=None,
                title=self.get_end_node_name(),
                logs=None,
            )
            self.sub_query_node.append(end_node)
            self.start_node_id = 1
        else:
            self.start_node_id = len(self.sub_query_node)

        start_node_name = self.get_start_node_name()
        question_node = Node(
            id=self.start_node_id,
            state=self.STATE.FINISH,
            question=rethink_question,
            answer=str([n.query for n in rewrite_question_list]),
            title=start_node_name,
            logs=None,
        )
        self.sub_query_node.append(question_node)

        for idx, item in enumerate(rewrite_question_list):
            cur_node = Node(
                id=len(self.sub_query_node),
                state=self.STATE.WAITING,
                question=item.query,
                answer=None,
                logs=None,
                title=self.get_sub_question_name(idx + 1),
            )
            self.sub_query_node.append(cur_node)

        # Generate edges between nodes
        for idx, item in enumerate(self.sub_query_node, start=1):
            if item.id == 0:
                continue
            if idx == len(self.sub_query_node):
                pipeline.edges.append(Edge(_from=item.id, to=0))
                break
            else:
                pipeline.edges.append(
                    Edge(_from=item.id, to=self.sub_query_node[idx].id)
                )
        pipeline.nodes = self.sub_query_node

        request = ReportPipelineRequest(task_id=self.task_id, pipeline=pipeline)
        if self.report_log:
            self.client.reasoner_dialog_report_pipeline_post(
                report_pipeline_request=request
            )
        else:
            logger.info(request)
        self.last_sub_question_size = len(rewrite_question_list)
        self.create_pipeline_times += 1

    def report_final_answer(self, query, answer, state):
        node = self.sub_query_node[0]
        node._state = state
        node._question = query
        node._answer = answer
        request = ReportPipelineRequest(task_id=self.task_id, node=node)
        if self.report_log:
            self.client.reasoner_dialog_report_node_post(
                report_pipeline_request=request
            )
        else:
            logger.info(request)

    def report_node(self, req_id, index, state, node_plan: LFPlan, kg_graph: KgGraph):
        """
        Report a single node in the pipeline.

        Args:
            req_id (str): Request ID.
            index (int): Index of the node.
            state (STATE): State of the node.
            node_plan (LFPlan): Logical form plan for the node.
            kg_graph (KgGraph): Knowledge graph associated with the node.
        """
        sub_logic_nodes_str = "\n".join([str(ln) for ln in node_plan.lf_nodes])
        # 为产品展示隐藏冗余信息
        sub_logic_nodes_str = re.sub(
            r"(\s,sub_query=[^)]+|get\([^)]+\))", "", sub_logic_nodes_str
        ).strip()
        context = [
            "## SPO Retriever",
            "#### logic_form expression: ",
            f"```java\n{sub_logic_nodes_str}\n```",
        ]
        sub_answer = None
        if node_plan.res is not None:
            sub_answer, cur_content, sub_graph = self._convert_lf_res_to_report_format(
                req_id=req_id,
                index=index,
                state=state,
                res=node_plan.res,
                kg_graph=kg_graph,
            )
            context += cur_content
        else:
            sub_graph = None

        logs = self.format_logs(context)
        report_node_id = self.start_node_id + index if index != 0 else 0
        node = self.sub_query_node[report_node_id]
        node._state = state
        node._question = node_plan.query
        node._answer = sub_answer
        node._logs = logs
        if sub_graph is not None:
            node._subgraph = [sub_graph]
        request = ReportPipelineRequest(task_id=self.task_id, node=node)
        if self.report_log:
            self.client.reasoner_dialog_report_node_post(
                report_pipeline_request=request
            )
        else:
            logger.info(request)

    def _convert_lf_res_to_report_format(
        self, req_id, index, state, res: SubQueryResult, kg_graph: KgGraph
    ):
        """
        Convert logical form result to a report format.

        Args:
            req_id (str): Request ID.
            index (int): Index of the node.
            state (STATE): State of the node.
            res (SubQueryResult): Result of the logical form query.
            kg_graph (KgGraph): Knowledge graph associated with the node.

        Returns:
            tuple: Sub-answer, context content, and sub-graph.
        """
        spo_retrieved = res.spo_retrieved
        context = []
        sub_answer = None
        if len(spo_retrieved) > 0:
            spo_answer_path = json.dumps(
                kg_graph.to_spo_path(spo_retrieved, self.language),
                ensure_ascii=False,
                indent=4,
            )
            spo_answer_path = f"```json\n{spo_answer_path}\n```"
            graph_id = f"{req_id}_{index}"
            graph_div = f"<div class='{graph_id}'></div>\n\n"
            sub_graph = self._convert_spo_to_graph(graph_id, spo_retrieved)
            context.append(graph_div)
            context.append(f"#### Triplet Retrieved:")
            context.append(spo_answer_path)
        else:
            context.append(f"#### Triplet Retrieved:")
            context.append("No triplets were retrieved.")
            sub_graph = None

        doc_retrieved = res.doc_retrieved
        context += self._update_sub_question_recall_docs(doc_retrieved)
        if state == ReporterIntermediateProcessTool.STATE.FINISH:
            context.append(f"#### answer based by {res.match_type}:")
            context.append(f"{res.sub_answer}")
            sub_answer = res.sub_answer
        return sub_answer, context, sub_graph

    def _convert_spo_to_graph(self, graph_id, spo_retrieved):
        """
        Convert SPO triples to a graph representation.

        Args:
            spo_retrieved (list): List of SPO triples.

        Returns:
            SubGraph: Graph representation of the SPO triples.
        """
        nodes = {}
        edges = []
        for spo in spo_retrieved:

            def _get_node(entity: EntityData):
                node = DataNode(
                    id=entity.to_show_id(self.language),
                    name=entity.get_short_name(),
                    label=entity.type_zh if self.language == "zh" else entity.type,
                    properties=entity.prop.get_properties_map() if entity.prop else {},
                )
                return node

            start_node = _get_node(spo.from_entity)
            end_node = _get_node(spo.end_entity)
            if start_node.id not in nodes:
                nodes[start_node.id] = start_node
            if end_node.id not in nodes:
                nodes[end_node.id] = end_node
            spo_id = spo.to_show_id(self.language)
            data_spo = DataEdge(
                id=spo_id,
                _from=start_node.id,
                from_type=start_node.label,
                to=end_node.id,
                to_type=end_node.label,
                properties=spo.prop.get_properties_map() if spo.prop else {},
                label=spo.type_zh if self.language == "zh" else spo.type,
            )
            edges.append(data_spo)
        sub_graph = SubGraph(
            class_name=graph_id, result_nodes=list(nodes.values()), result_edges=edges
        )
        return sub_graph

    def _update_sub_question_recall_docs(self, docs):
        """
        Update the context with retrieved documents for sub-questions.

        Args:
            docs (list): List of retrieved documents.

        Returns:
            list: Updated context content.
        """
        if docs is None or len(docs) == 0:
            return []
        doc_content = [f"## Chunk Retriever"]
        doc_content.extend(["|id|content|", "|-|-|"])
        for i, d in enumerate(docs, start=1):
            _d = d.replace("\n", "<br>")
            doc_content.append(f"|{i}|{_d}|")
        return doc_content

    def format_logs(self, logs):
        """
        Format logs into a string.

        Args:
            logs (list or str): Logs to be formatted.

        Returns:
            str: Formatted log content.
        """
        if not logs:
            return None
        content = ""
        if isinstance(logs, str):
            try:
                logs = json.loads(logs)
            except:
                logs = eval(logs)
        for idx, item in enumerate(logs, start=1):
            content += f"{item}\n"
        return content
