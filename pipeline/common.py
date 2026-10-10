#!/usr/bin/env python3
"""Helpers shared by the pipeline drivers: paths, the debug trace, and the
command runner that reports what each command produced.

Split out of run.py unchanged so the per-core benchmark and the hetero_soc
driver trace their runs the same way.
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEEPLOY = ROOT / "deps" / "deeploy"
DEEPLOY_TEST = DEEPLOY / "DeeployTest"
GENERIC_LIB = DEEPLOY / "TargetLibraries" / "Generic"
GVSOC = ROOT / "deps" / "gvsoc" / "install" / "bin" / "gvsoc"
TC = ROOT / "toolchains" / "xpack-riscv-none-elf-gcc-15.2.0-1" / "bin" / "riscv-none-elf-gcc"
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNTIME = ROOT / "runtime"
TARGETS = ROOT / "targets"
WORK = ROOT / "work"
RESULTS = ROOT / "results"


# An op that ships its own evaluation set brings its own host program too: one
# that loops over the set and scores it, instead of running a single inference
# and diffing it. An op is recognised by the data header it carries.
#
#   header          the generated header that identifies the application
#   main            the host program under runtime/mesh
#   beacon          the regex for the single metrics line it prints
#   samples_flag    what --images caps, for the message
APPS = {
    "mnist": {"header": "mnist_data.h", "main": "mnist_main.c", "unit": "image"},
    "kws": {"header": "kws_data.h", "main": "kws_main.c", "unit": "clip"},
    "llm": {"header": "llm_data.h", "main": "llm_main.c", "unit": "problem"},
}


def detect_app(test_dir: Path):
    """Which application this op directory is, if any."""
    for name, app in APPS.items():
        if (test_dir / app["header"]).is_file():
            return name, app
    return None, None


# --- Main-memory device ------------------------------------------------------
#
# Which RAM main memory is (targets/hetero/memsys.py DRAM_KIND). It is a design
# choice like any other, so it reaches the boards the way every design choice
# does: through the HES_DESIGN file, which gvsoc and every generator this
# process starts inherit. --dram writes that file.

sys.path.insert(0, str(TARGETS))
from hetero.dram_presets import KINDS as DRAM_KINDS  # noqa: E402


def dram_kind() -> str:
    """The main-memory device the current HES_DESIGN selects."""
    path = os.environ.get("HES_DESIGN")
    if not path:
        return "fixed"
    return json.loads(Path(path).read_text()).get("DRAM_KIND", "fixed")


def use_dram(kind) -> str:
    """Make every simulation this process starts use main-memory device `kind`.

    Merges DRAM_KIND into the design already in HES_DESIGN (if any), writes the
    result under work/designs/ and points HES_DESIGN at it. `kind` None leaves
    the environment alone. Returns the effective kind either way.
    """
    if kind is None:
        return dram_kind()
    if kind not in DRAM_KINDS:
        raise SystemExit(f"--dram {kind}: choose one of {', '.join(DRAM_KINDS)}")
    base = os.environ.get("HES_DESIGN")
    design = json.loads(Path(base).read_text()) if base else {}
    if kind == "fixed" and "DRAM_KIND" not in design:
        return kind
    design["DRAM_KIND"] = kind
    text = json.dumps(design, indent=2, sort_keys=True) + "\n"
    out = WORK / "designs" / f"dram-{kind}-{hashlib.sha1(text.encode()).hexdigest()[:8]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    # Absolute: gvsoc runs from the run directory, not from here.
    os.environ["HES_DESIGN"] = str(out.resolve())
    return kind


DRAM_RE = re.compile(r"\[HES-DRAM\] mem=(\S+) kind=(\S+) reads=(\d+) writes=(\d+) "
                     r"bursts=(\d+) row_hits=(\d+) row_misses=(\d+) row_conflicts=(\d+) "
                     r"refreshes=(\d+) busy_ns=(\S+) latency_cycles=(\d+)")
WRITEBACK_RE = re.compile(r"\[HES-MEM\] cache=(\S+) writebacks=(\d+)")


def dram_counters(m) -> dict:
    """One [HES-DRAM] match (DRAM_RE) as a dict."""
    hits, misses, conflicts = int(m.group(6)), int(m.group(7)), int(m.group(8))
    bursts = hits + misses + conflicts
    return {
        "mem": m.group(1),
        "kind": m.group(2),
        "reads": int(m.group(3)),
        "writes": int(m.group(4)),
        "bursts": int(m.group(5)),
        "row_hits": hits,
        "row_misses": misses,
        "row_conflicts": conflicts,
        # Bursts that found their row open; None for a device without rows.
        "row_hit_rate": round(hits / bursts, 4) if bursts else None,
        "refreshes": int(m.group(9)),
        "busy_ns": float(m.group(10)),
        "latency_cycles": int(m.group(11)),
    }


DEBUG = False


def rel(p) -> str:
    """Path relative to the repo root when it is inside it, else absolute."""
    p = Path(p)
    for cand in (p, p.resolve()):
        try:
            return str(cand.relative_to(ROOT))
        except ValueError:
            pass
    return str(p)


def human_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def dbg(msg: str) -> None:
    """One line of --debug trace, on stderr so stdout stays the report."""
    if DEBUG:
        print(f"[dbg] {msg}", file=sys.stderr, flush=True)


def note_file(path: Path, why: str) -> None:
    """Trace a file written by the driver itself rather than by a command."""
    if DEBUG:
        size = human_size(path.stat().st_size) if path.is_file() else "not created"
        dbg(f"  {rel(path)}  ({size}, {why})")


def snapshot(paths) -> dict:
    """Files under each directory in `paths`, with mtimes, before a command runs."""
    seen = {}
    for p in paths:
        p = Path(p)
        seen[p] = ({f: f.stat().st_mtime_ns for f in p.rglob("*") if f.is_file()}
                   if p.is_dir() else {})
    return seen


def report_produced(produces, before: dict) -> None:
    """Trace what a command generated: named files, plus files new or rewritten
    in the directories it writes into (so a re-run still lists its outputs)."""
    if not DEBUG:
        return
    for p in produces:
        p = Path(p)
        if p.is_dir():
            old = before.get(p, {})
            touched = sorted(f for f in p.rglob("*")
                             if f.is_file() and f.stat().st_mtime_ns != old.get(f))
            for f in touched:
                dbg(f"  -> {rel(f)}  ({human_size(f.stat().st_size)})")
            if not touched:
                dbg(f"  -> {rel(p)}/  (nothing written)")
        elif p.is_file():
            dbg(f"  -> {rel(p)}  ({human_size(p.stat().st_size)})")
        else:
            dbg(f"  -> {rel(p)}  (not created)")


def sh(cmd, cwd=None, timeout=None, env=None, produces=()):
    """Run a command, capturing its output.

    `produces` names the files (or directories) the command is expected to
    generate; with --debug the command line and those files are traced.
    """
    before = {}
    if DEBUG:
        before = snapshot(produces)
        dbg(f"$ {shlex.join(str(c) for c in cmd)}")
        if cwd:
            dbg(f"  (cwd: {rel(cwd)})")

    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, timeout=timeout, env=env,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        dbg(f"  timed out after {timeout}s")
        report_produced(produces, before)
        raise

    dbg(f"  exit={r.returncode} in {time.time() - t0:.1f}s")
    report_produced(produces, before)
    return r



def set_debug(on: bool) -> None:
    """Turn the command trace on or off for this process."""
    global DEBUG
    DEBUG = on
