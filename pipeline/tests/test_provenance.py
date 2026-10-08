import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.experiment.fingerprint import fingerprint  # noqa: E402
from pipeline.experiment.provenance import (  # noqa: E402
    SOURCE_SUFFIXES,
    capture_run_provenance,
    git_snapshot,
    git_snapshot_optional,
    patch_digests,
    run_source_files,
    semantic_git_identity,
    source_files_under,
    toolchain_snapshot,
)


class ProvenanceTests(unittest.TestCase):
    @staticmethod
    def resolved(*, target="hetero_soc", memory="fixed", application=None):
        return {
            "simulator": {"target": target},
            "memory": {"kind": memory},
            "workload": {"application": application},
        }

    def _git(self, root, *args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def _repository(self, root):
        self._git(root, "init", "-q")
        self._git(root, "config", "user.email", "m4ia-test@example.invalid")
        self._git(root, "config", "user.name", "M4IA Test")
        (root / "model.cpp").write_text("int model = 1;\n")
        self._git(root, "add", "model.cpp")
        self._git(root, "commit", "-q", "-m", "fixture")

    def test_git_snapshot_represents_dirty_tracked_and_untracked_source_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repository(root)
            clean = git_snapshot(root, name="fixture")
            self.assertFalse(clean["dirty"])
            self.assertIsNone(clean["diff_digest"])
            self.assertIsNone(clean["untracked_digest"])

            (root / "model.cpp").write_text("int model = 2;\n")
            (root / "new.hpp").write_text("#pragma once\n")
            dirty = git_snapshot(root, name="fixture")
            self.assertTrue(dirty["dirty"])
            self.assertTrue(dirty["diff_digest"].startswith("sha256:"))
            self.assertTrue(dirty["untracked_digest"].startswith("sha256:"))

    def test_remote_url_is_provenance_not_semantic_identity(self):
        base = {
            "available": True,
            "name": "dependency",
            "repository": "https://example.invalid/upstream.git",
            "revision": "abc123",
            "dirty": False,
            "diff_digest": None,
            "untracked_digest": None,
        }
        mirror = dict(base, repository="ssh://mirror.invalid/dependency.git")
        self.assertEqual(semantic_git_identity(base), semantic_git_identity(mirror))
        self.assertEqual(
            fingerprint(semantic_git_identity(base)),
            fingerprint(semantic_git_identity(mirror)),
        )

    def test_optional_git_snapshot_reports_missing_metadata_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = git_snapshot_optional(tmp, name="runtime-copy")
        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["reason"], "git-metadata-unavailable")
        self.assertIsNone(snapshot["revision"])

    def test_source_suffixes_cover_current_and_common_cpp_spellings(self):
        self.assertTrue({".cpp", ".hpp", ".cc", ".cxx", ".hh"} <= SOURCE_SUFFIXES)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = ["dram.cpp", "dram_core.hpp", "other.cc", "more.cxx", "last.hh"]
            for name in wanted:
                (root / name).write_text(name)
            (root / "README.md").write_text("not causal source")
            self.assertEqual(
                [path.name for path in source_files_under(root)],
                sorted(wanted),
            )

    def test_patch_digests_are_deterministic_and_root_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.patch").write_text("b\n")
            (root / "a.patch").write_text("a\n")
            forward = patch_digests(root, [root / "b.patch", root / "a.patch"])
            reverse = patch_digests(root, [root / "a.patch", root / "b.patch"])
            self.assertEqual(forward, reverse)
            self.assertEqual([item["path"] for item in forward], ["a.patch", "b.patch"])

    def test_toolchain_snapshot_captures_version_and_binary_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "compiler"
            executable.write_text("#!/bin/sh\nprintf 'fixture compiler 1.0\\n'\n")
            executable.chmod(executable.stat().st_mode | 0o111)
            snapshot = toolchain_snapshot(executable)
            self.assertEqual(snapshot["executable"], "compiler")
            self.assertEqual(snapshot["version"], "fixture compiler 1.0")
            self.assertTrue(snapshot["binary_digest"].startswith("sha256:"))

    def test_run_source_set_covers_execution_and_excludes_noncausal_files(self):
        relative = {
            path.relative_to(ROOT).as_posix()
            for path in run_source_files(
                ROOT, self.resolved(), include_sweep=True,
            )
        }
        for path in (
            "pipeline/run_hetero.py",
            "pipeline/hetero_platform/mapper.py",
            "pipeline/hetero_platform/generate.py",
            "pipeline/sweep/run.py",
            "runtime/mesh/hes_host.c",
            "targets/hetero_soc.py",
        ):
            self.assertIn(path, relative)
        for path in (
            "runtime/mesh/hes_system.h",
            "runtime/mesh/host.ld",
            "runtime/mesh/snitch.ld",
            "runtime/mesh/spatz.ld",
            "runtime/mesh/memsys_expect.h",
            "targets/hetero_ara.py",
            "targets/hetero/dram.cpp",
            "targets/hetero/dram.py",
            "targets/hetero/dram_core.hpp",
        ):
            self.assertNotIn(path, relative)
        self.assertNotIn("pipeline/sweep/area.py", relative)
        self.assertNotIn("README.md", relative)
        self.assertFalse(any(path.startswith("docs/") for path in relative))
        self.assertFalse(any(path.startswith("gui/") for path in relative))
        self.assertFalse(any(path.startswith("results/") for path in relative))
        self.assertFalse(any(path.startswith("work/") for path in relative))

    def _run_fixture(self, root):
        paths = {
            "pipeline/run_hetero.py": "runner\n",
            "targets/hetero_soc.py": "cva6 adapter\n",
            "targets/hetero_ara.py": "ara adapter\n",
            "targets/hetero/dram.cpp": "real dram\n",
            "runtime/mesh/hes_system.h": "generated default\n",
            "README.md": "docs\n",
        }
        for relative, content in paths.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def _run_digest(self, root, *, target="hetero_soc", memory="fixed"):
        return capture_run_provenance(
            root, self.resolved(target=target, memory=memory),
        )["source_set"]["digest"]

    def test_run_source_digest_changes_for_causal_source_not_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_fixture(root)
            runner = root / "pipeline" / "run_hetero.py"
            readme = root / "README.md"
            first = self._run_digest(root)
            readme.write_text("second docs\n")
            second = self._run_digest(root)
            runner.write_text("second\n")
            third = self._run_digest(root)
        self.assertEqual(first, second)
        self.assertNotEqual(second, third)

    def test_generated_mesh_defaults_do_not_change_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_fixture(root)
            first = self._run_digest(root)
            (root / "runtime/mesh/hes_system.h").write_text("stale other default\n")
            second = self._run_digest(root)
        self.assertEqual(first, second)

    def test_real_dram_sources_are_memory_conditional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_fixture(root)
            fixed_before = self._run_digest(root, memory="fixed")
            real_before = self._run_digest(root, memory="lpddr5")
            (root / "targets/hetero/dram.cpp").write_text("changed real dram\n")
            fixed_after = self._run_digest(root, memory="fixed")
            real_after = self._run_digest(root, memory="lpddr5")
        self.assertEqual(fixed_before, fixed_after)
        self.assertNotEqual(real_before, real_after)

    def test_unselected_host_adapter_does_not_change_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_fixture(root)
            cva6_before = self._run_digest(root, target="hetero_soc")
            ara_before = self._run_digest(root, target="hetero_ara")
            (root / "targets/hetero_ara.py").write_text("changed ara adapter\n")
            cva6_after = self._run_digest(root, target="hetero_soc")
            ara_after = self._run_digest(root, target="hetero_ara")
            (root / "targets/hetero_soc.py").write_text("changed cva6 adapter\n")
            cva6_final = self._run_digest(root, target="hetero_soc")
            ara_final = self._run_digest(root, target="hetero_ara")
        self.assertEqual(cva6_before, cva6_after)
        self.assertNotEqual(ara_before, ara_after)
        self.assertNotEqual(cva6_after, cva6_final)
        self.assertEqual(ara_after, ara_final)


if __name__ == "__main__":
    unittest.main()
