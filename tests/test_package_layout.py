"""检查目录变更涉及的旧调用、容器源码清单和仓库资源定位。"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from timeguard.evals.baselines.native_baseline import worker_sources
from timeguard.worker_bundle import source_hashes, stage_package


class PackageLayoutTest(unittest.TestCase):
    def test_flat_import_and_command_match_canonical_entry(self):
        from timeguard import native_baseline
        from timeguard.evals.baselines import native_baseline as current

        self.assertIs(native_baseline, current)
        def help_text(module):
            return subprocess.check_output([sys.executable, "-m", module, "--help"])
        self.assertEqual(help_text("timeguard.financebench_paired"),
                         help_text("timeguard.evals.financebench_paired"))

    def test_paired_worker_has_a_standalone_package_and_no_grading_material(self):
        files = worker_sources("pair_a1", official=True)
        self.assertNotIn("evals/judge.py", files)
        self.assertTrue(all(name.endswith(".py") for name in files))
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "timeguard"
            copied = stage_package(package, files)
            self.assertEqual(copied, source_hashes(files))
            self.assertEqual(sorted(copied), sorted(p.relative_to(package).as_posix()
                                                    for p in package.rglob("*.py")))
            code = [
                "import sys",
                "from pathlib import Path",
                "sys.path.insert(0, sys.argv[1])",
                "from timeguard.evals.baselines import native_baseline",
                "from timeguard.evidence import evidence_layers",
                "from timeguard.llm import openai_native",
                "assert Path(native_baseline.__file__).resolve().is_relative_to(sys.argv[1])",
            ]
            if importlib.util.find_spec("langgraph"):
                code.append("from timeguard.research.live_finance import enhance")
            subprocess.run([sys.executable, "-I", "-c", "\n".join(code), directory],
                           cwd=directory, check=True)

    def test_catalog_finds_filing_dates_outside_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "from timeguard.retrieval.catalog import catalog; assert catalog([], {}) == []")
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, "-I", "-c", code, str(root)], cwd=directory, check=True)


if __name__ == "__main__":
    unittest.main()
