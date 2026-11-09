"""检查原生基线的题目隔离、档位边界及元数据恢复冲突保护。"""
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from timeguard.native_baseline import arm_config, select_queries, install_observers
from timeguard.native_metadata import expected_metadata, restore_allowed


class NativeBaselineTests(unittest.TestCase):
    def test_generation_input_removes_gold(self):
        rows=[{"id":"q","question":"问题","split":"dev","answer":"金标","evidence":["金标证据"]}]
        self.assertEqual(select_queries(rows,"dev"),[{"id":"q","question":"问题","split":"dev"}])

    def test_duplicate_and_empty_plan_rejected(self):
        row={"id":"q","question":"问题","split":"dev"}
        for rows,split in [([row,row],"dev"),([row],"test")]:
            with self.assertRaises(ValueError):select_queries(rows,split)

    def test_native_modes_are_distinct_from_model_effort(self):
        self.assertEqual([arm_config("agent_"+m,"")["reasoning"] for m in ("low","medium","high","ultra")],[1,2,3,4])
        self.assertEqual(arm_config("plain","reranker")["rerank_id"],"")
        self.assertEqual(arm_config("rerank","reranker")["rerank_id"],"reranker")
        self.assertEqual(arm_config("metadata_all","")["meta_data_filter"],{"method":"auto"})
        self.assertEqual(arm_config("metadata_auto","")["meta_data_filter"]["method"],"semi_auto")

    def test_metadata_restore_rejects_concurrent_change(self):
        document={"company":"公司","doc_period":2023};old={"已有字段":"原值"}
        current=expected_metadata(old,document)
        self.assertTrue(restore_allowed(current,old,document))
        self.assertFalse(restore_allowed({**current,"他人字段":"新值"},old,document))
        self.assertEqual(old,{"已有字段":"原值"})

    def test_metadata_namespace_collision_rejected(self):
        with self.assertRaises(ValueError):expected_metadata({"tg_native_company":"已有值"},{"company":"公司","doc_period":2023})

    def test_observers_preserve_native_arguments_and_return_values(self):
        result={"chunks":[]};native_retrieve=AsyncMock(return_value=result)
        retriever=SimpleNamespace(retrieval=native_retrieve,insert_citations=Mock())
        blocks=["ID: 1\n原生正文"];native_prompt=Mock(return_value=blocks)
        class Bundle:
            async def async_chat(self,*args,**kwargs):return "原生回答"
        service=SimpleNamespace(kb_prompt=native_prompt,apply_meta_data_filter=AsyncMock(),LLMBundle=Bundle,RAGTools=type("Tools",(),{}))
        install_observers(service,SimpleNamespace(chunks_format=lambda value:[]),retriever)
        args=("问题",object());kwargs={"doc_ids":["文档"],"page_size":6}
        self.assertIs(asyncio.run(retriever.retrieval(*args,**kwargs)),result)
        native_retrieve.assert_awaited_once_with(*args,**kwargs)
        info={"chunks":[]}
        self.assertIs(service.kb_prompt(info,100),blocks)
        native_prompt.assert_called_once_with(info,100)
