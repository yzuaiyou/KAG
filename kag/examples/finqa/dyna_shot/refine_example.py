import logging
import re
import os
import json

import chromadb

import re
import logging
from typing import List

from kag.interface import PromptABC


from kag.interface import LLMClient

from kag.common.conf import KAG_CONFIG


from kag.solver.utils import init_prompt_with_fallback

from kag.examples.finqa.solver.prompt.question_classify import FinQAQuestionClassify
from kag.examples.finqa.builder.indexer import convert_finqa_to_md_file


@PromptABC.register("default_refine_example_prompt")
class FinQARefineExamplePrompt(PromptABC):
    template_zh = """
# 任务
你是一个财经领域的专家，你的任务对参考案例进行分类。
我会给你一个问题的解题过程，以及该问题对应的五个参考案例。

# 要求
先分析输入问题及其解题思路，总结计算公式。
再逐个分析参考案例，对每个参考案例按照如下分类进行分析，输出结果。
参考案例可分为如下三类：
1. 不相关，与问题完全没有关系。
2. 与问题相关，但是解题思路和公式不一致。
3. 完全一致，问题相关性很高，解题思路和公式都一致。
注意：
```
参考案例相关性不要看时间，例如问题是2010年到2020年，案例中是其他年份，这个没关系，重点看公式。
参考案例使用的金融数据可能不同，例如计算利润百分比变化和计算营收百分比变化，这个不影响相关性，主要看解题思路和公式。
```

# 真正的输入
问题: $question
问题的参考信息:
```
$info
```
问题计算过程: $process

## 参考案例
$examples
""".strip()

    template_en = template_zh

    @property
    def template_variables(self) -> List[str]:
        return ["question", "info", "process", "examples"]

    def parse_response(self, response: str, **kwargs):
        return response


class RefineExamplePipeline:
    def __init__(self, **kwargs):
        """
        Initializes the think-and-act loop class.

        :param max_iterations: Maximum number of iteration to limit the thinking and acting loop, defaults to 3.
        :param reflector: Reflector instance for reflect tasks.
        :param reasoner: Reasoner instance for reasoning about tasks.
        :param generator: Generator instance for generating actions.
        :param memory: Assign memory store type
        """
        super().__init__(**kwargs)
        self.llm_client: LLMClient = LLMClient.from_config(
            KAG_CONFIG.all_config["chat_llm"]
        )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        chromadb_path = os.path.join(current_dir, "chromadb_v2")
        os.makedirs(chromadb_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.chroma_client.create_collection(
            name="finqa_example", get_or_create=True
        )

        self.refine_example_prompt = init_prompt_with_fallback(
            "refine_example_prompt", "default"
        )

        self.question_classify_prompt = init_prompt_with_fallback(
            "question_classify", "table"
        )

    def question_classify(self, question):
        llm: LLMClient = self.llm_client
        params = {"question": question}
        tags = llm.invoke(
            variables=params,
            prompt_op=self.question_classify_prompt,
            with_json_parse=False,
            with_except=True,
        )
        return tags

    def retrieval_examples(self, question, tags, topn=3):
        doc = question + " tags=" + str(tags)
        rsts = self.collection.query(query_texts=[doc], n_results=topn)
        examples = []
        for meta in rsts["metadatas"][0]:
            example = f"Question:{meta['question']}\nFormula:{meta['formula']}"
            examples.append(example)
            # examples.append(meta["example"])
        return examples

    def _to_example_str(self, examples):
        e_str = ""
        for i, e in enumerate(examples):
            e_str += f"\n\n### {i}\n{e}"
        return e_str.strip()

    def refine(self, start_index=0):
        idx_set = {
            385,
            130,
            771,
            265,
            268,
            910,
            911,
            401,
            658,
            403,
            916,
            534,
            408,
            281,
            285,
            414,
            548,
            427,
            940,
            686,
            47,
            178,
            56,
            826,
            187,
            317,
            446,
            64,
            961,
            66,
            834,
            708,
            324,
            70,
            455,
            329,
            974,
            472,
            602,
            992,
            98,
            100,
            631,
            1130,
            108,
            1004,
            238,
            494,
            496,
            369,
            372,
            887,
            504,
            761,
            891,
            380,
            767,
        }
        train_data_list = self.load_finqa_train_data("test")
        for i, item in enumerate(train_data_list):
            if i < start_index:
                continue
            if i not in idx_set:
                continue
            # file_path = convert_finqa_to_md_file(item)
            # with open(file_path, "r", encoding="utf-8") as file:
            #     file_content = file.read()
            try:
                _id = item["id"]
                question = item["qa"]["question"]
                tags = self.question_classify(question=question)
                examples = self.retrieval_examples(question=question, tags=tags, topn=5)
                examples_str = self._to_example_str(examples)
                info = ""
                for k, v in item["qa"]["gold_inds"].items():
                    if len(info) > 0:
                        info += "\n"
                    info += f"{k}: {v}"
                # info = str(file_content)
                process = str(item["qa"]["program_re"])
                # answer = str(item["qa"]["answer"])
                params = {
                    "question": question,
                    "info": info,
                    "process": process,
                    "examples": examples_str,
                }
                logging.info("#" * 100)
                logging.info("start,index=%d", i)
                response = self.llm_client.invoke(
                    variables=params,
                    prompt_op=self.refine_example_prompt,
                    with_json_parse=False,
                    with_except=True,
                    with_cache=False,
                )
                print(response)
            except:
                logging.exception("error")
                continue

    def load_finqa_train_data(self, _type="train") -> list:
        """
        load data
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(current_dir, "..", "builder", "data", f"{_type}.json")
        with open(file_name, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        print("finqa data list len " + str(len(data_list)))
        for _idx, data in enumerate(data_list):
            data["index"] = _idx
        print(f"type={_type},len={len(data_list)}")
        return data_list


if __name__ == "__main__":
    resp = RefineExamplePipeline()
    resp.refine(67)
