"""用离线元数据检查原生调用证据，防止缓存或重试被当成免费成功。"""

import json
import unittest

from timeguard.matrix import audit_summary, same_bridge_as_history


class MatrixAuditTests(unittest.TestCase):
    def record(self, request_id, success=True, effort="low"):
        return {"request_id": request_id, "requested_model": "gpt-6.1-sol@" + effort,
                "requested_effort": effort, "actual_model": "gpt-6.1-sol", "reasoning_effort": effort,
                "bridge_source_sha256": "a" * 64, "instruction_source_sha256": {"示例指令文件": "b" * 64},
                "success": success, "usage": {"totalTokens": 50}, "error_type": None if success else "TimeoutError"}

    def encoded(self, records):
        return b"\n".join(json.dumps(row).encode() for row in records) + b"\n"

    def test_native_retries_are_counted_with_failure_tokens(self):
        report = audit_summary(self.encoded([self.record("failed", False), self.record("ok")]), "low", 1)
        self.assertTrue(report["verified"])
        self.assertEqual(report["actual_attempts"], 2)
        self.assertEqual(report["failed_attempts"], 1)
        self.assertEqual(report["usage_total_tokens"], 100)

    def test_no_call_or_other_effort_cannot_prove_actual_generation(self):
        self.assertFalse(audit_summary(b"", "low", 1)["verified"])
        self.assertFalse(audit_summary(self.encoded([self.record("wrong", effort="high")]), "low", 1)["verified"])
        self.assertFalse(audit_summary(self.encoded([self.record("same"), self.record("same")]), "low", 2)["verified"])

    def test_bridge_and_instruction_fingerprints_are_required_and_frozen(self):
        original = audit_summary(self.encoded([self.record("first")]), "low", 1)
        for field, value in (("bridge_source_sha256", "c" * 64),
                             ("instruction_source_sha256", {"示例指令文件": "d" * 64})):
            changed = self.record("changed")
            changed[field] = value
            self.assertFalse(audit_summary(self.encoded([self.record("first"), changed]), "low", 2)["verified"])
            report = audit_summary(self.encoded([changed]), "low", 1)
            self.assertTrue(report["verified"])
            self.assertFalse(same_bridge_as_history(report, [original]))
        missing = self.record("missing")
        missing.pop("instruction_source_sha256")
        self.assertFalse(audit_summary(self.encoded([missing]), "low", 1)["verified"])


if __name__ == "__main__":
    unittest.main()
