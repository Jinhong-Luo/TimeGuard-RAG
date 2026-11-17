"""四块对照的候选、范围与计数回归。"""
import unittest
from unittest.mock import Mock, patch

from timeguard.four_blocks import pick, scoped_search, select_four


class FourBlocksTest(unittest.TestCase):
    def test_unique_in_range(self):
        self.assertEqual(pick({"selected":[20,5,9,1]},20),[20,5,9,1])
        for ids in ([0,2,3,4],[1,2,3,21],[1,1,3,4],[True,2,3,4],[1,2]):
            with self.assertRaises(ValueError):
                pick({"selected":ids},20)

    def test_invalid_output_restores_original_four(self):
        chunks = [{"document_name":"报告","content":"正文","positions":[[1]]} for _ in range(20)]
        rt = {"base_url":"测试","token":"测试"}
        with patch("timeguard.four_blocks.post_completion",side_effect=TimeoutError()):
            log = []
            self.assertEqual(select_four(rt,"问题",chunks,log),[1,2,3,4])
            self.assertEqual(len(log),1)
            self.assertEqual(log[0]["fallback"],"TimeoutError")

    def test_scope_never_expands(self):
        client = Mock()
        config = {"dataset_ids":["库"], "retrieval":{"vector_similarity_weight":0.3}}
        with self.assertRaises(ValueError):
            scoped_search(client,config,"问题",[])
        client.request.assert_not_called()
        client.request.return_value = {"chunks":[{"document_id":"乙","id":"一"}]}
        with self.assertRaises(ValueError):
            scoped_search(client,config,"问题",["甲"])
