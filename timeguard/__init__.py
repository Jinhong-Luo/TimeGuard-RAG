"""从仓库根目录加载后端，同时支持之前的平铺模块名和 python -m 命令。"""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from pathlib import Path
import sys

__path__.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "timeguard"))

# 固定旧版曾公开的模块名。新模块直接使用所属子包，不必在此登记。
_OLD_MODULES = {
    "server": (
        "app", "ask",
    ),
    "indexing": (
        "data", "ingest", "audit_index", "extend_corpus",
    ),
    "retrieval": (
        "scope", "filing_pages", "outline_search", "history_read",
        "page_select",
    ),
    "research": (
        "agentic_candidate", "agentic_finance", "financial_agent", "finance_route",
        "live_finance",
    ),
    "evidence": (
        "evidence_layers", "evidence_review", "fact_selection", "final_snapshot",
        "formula_guard", "message_budget",
    ),
    "llm": (
        "openai_native", "openai_transport", "tei_adapter",
    ),
    "operations": (
        "bootstrap", "gpu_image", "pull_images", "setup_service",
        "reranker", "probe",
    ),
    "evals": (
        "financebench_paired", "financebench_checks", "judge", "score",
        "select", "review_form",
    ),
    "evals.baselines": (
        "matrix", "run", "native_baseline", "native_trial",
        "native_http", "native_matrix", "native_metadata", "native_full",
        "native_recover", "r1_dev",
    ),
    "evals.trials": (
        "adaptive_final_trial", "agentic_fallback_check", "agentic_regressions", "agentic_replay",
        "app_trial", "budget_fact_trial", "fact_final_trial", "fact_tune_trial",
        "filing_trial", "four_blocks", "four_feedback", "four_protection",
        "full_feedback", "oracle", "outline_trial", "paired_render",
        "paired_trial", "rerank_trial", "retrieval_experiment", "table_trial",
        "verify_pair",
    ),
    "evals.analysis": (
        "agentic_results", "citation_audit", "fact_report", "final_report",
        "full_feedback_report", "layer_report", "legacy_citation_audit", "native_report",
        "operand_audit", "page_report", "paired_audit", "paired_report",
        "r1_report", "report_baseline", "snapshot_history", "snapshot_report",
    ),
}
_MOVED = {f"timeguard.{name}": f"timeguard.{group}.{name}"
          for group, names in _OLD_MODULES.items() for name in names}


class _LegacyModule(Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def exec_module(self, module):
        # 两种写法共用模块对象，确保会话状态、异常类型和补丁目标一致。
        sys.modules[module.__name__] = import_module(self.canonical)

    def get_code(self, fullname):
        # runpy 在 python -m 下直接取代码，需保留原命令的参数和退出状态。
        code = f"from runpy import run_module\nrun_module({self.canonical!r}, run_name='__main__', alter_sys=True)"
        return compile(code, fullname, "exec")


class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        canonical = _MOVED.get(fullname)
        if canonical:
            return spec_from_loader(fullname, _LegacyModule(canonical))
        return None


sys.meta_path.append(_LegacyFinder())
