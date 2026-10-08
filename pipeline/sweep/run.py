#!/usr/bin/env python3
"""Sweep a group of ONNX models across a design space.

    python pipeline/sweep/run.py --models ops/mnist ops/kws --grid ofat -o results/sweep/s1

For each (design point, model) it writes one JSONL row carrying the resolved
design, the cycles, the cache counters and energy, and the modelled area -- so a
row says what was run, not what was asked for.

Structure, and why:

  * Designs are grouped by build key. Everything except VLEN is a property
    GVSoC reads from its generated config, so a whole grid usually runs against
    one existing build; the driver refuses up front rather than silently
    running the wrong VLEN if a build is missing.
  * Calibration is per resolved machine, not per cell, and cached: the engine
    cost table depends on numeric design, host and memory, not on the network.
  * Every cell gets its own copy of runtime/mesh. That is not fastidiousness --
    hes_host.h and friends include "hes_system.h" with quotes, which GCC
    resolves next to the including file before any -I, so a cell that shared
    the directory would link its own linker script against another cell's
    addresses and report plausible wrong numbers.
  * A cell whose status is not ok is recorded as infeasible with the reason,
    never dropped. A design too small to hold a layer must show up as
    infeasible rather than as fast.
"""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pipeline"))

import area as area_model          # noqa: E402
import design as design_mod        # noqa: E402
from common import detect_app  # noqa: E402
from experiment.artifact_identity import (  # noqa: E402
    build_machine_context,
    build_run_input_context,
)
from experiment.calibration_artifact import (  # noqa: E402
    cache_status,
    capture_calibration_context,
    write_metadata,
)
from experiment.discovery import current_host_profile_catalog  # noqa: E402
from experiment.provenance import capture_run_provenance  # noqa: E402
from experiment.resolve import resolve_experiment  # noqa: E402
from experiment.run_manifest import write_run_manifest  # noqa: E402
from experiment.schema import ExperimentRequest  # noqa: E402
from experiment.workload import resolve_workload_path  # noqa: E402

PYTHON = ROOT / ".venv" / "bin" / "python"
MESH = ROOT / "runtime" / "mesh"
DEEPLOY_TEST = ROOT / "deps" / "deeploy" / "DeeployTest"


# --- grids ------------------------------------------------------------------

# One factor at a time: the baseline, then each knob moved alone. This is the
# screening pass -- it says which knobs move cycles at all, and a full factorial
# over everything would be 10^5 designs for no extra information.
OFAT = {
    "HOST_NB_LANES": [2, 8],
    "SPATZ_NB_LANES": [2, 8],
    "SNITCH_NB_CORE": [5, 17],
    "SPATZ_NB_CORE": [5, 17],
    "DCACHE_SIZE": [16 * 1024, 64 * 1024, 128 * 1024],
    "ICACHE_SIZE": [8 * 1024, 32 * 1024],
    "L2_SIZE": [128 * 1024, 2048 * 1024],
    "L2_WAYS": [4, 16],
    "LINE_SIZE": [32, 128],
    "TCDM_SIZE": [0x10000, 0x40000],
    "WIDE_AXI_WIDTH": [32, 128],
}


def parse_knob(spec: str):
    """One --knob KNOB=v1,v2,... into (knob, [values]).

    Values go through int(v, 0), so 0x40000 and 262144 are both accepted -- the
    sizes read naturally either way and the design file records the same number.

    An unknown knob is refused rather than ignored: silently sweeping nothing
    would produce a grid of identical designs and a sensitivity table of zeros,
    which looks like a finding.
    """
    if "=" not in spec:
        raise SystemExit(f"--knob wants KNOB=v1,v2,...; got {spec!r}")
    knob, _, values = spec.partition("=")
    knob = knob.strip().upper()
    if knob not in design_mod.DEFAULTS:
        # Substring matching misses the realistic typo: SPATZ_LANES is neither
        # a substring nor a superstring of SPATZ_NB_LANES.
        near = difflib.get_close_matches(knob, design_mod.DEFAULTS, n=3, cutoff=0.6)
        hint = f" Did you mean {', '.join(near)}?" if near else ""
        raise SystemExit(f"unknown knob {knob!r}.{hint}\n"
                         f"  Known knobs: {', '.join(sorted(design_mod.DEFAULTS))}")
    out = []
    for v in values.split(","):
        v = v.strip()
        if not v:
            continue
        try:
            out.append(int(v, 0))
        except ValueError:
            raise SystemExit(f"--knob {knob}: {v!r} is not an integer")
    if not out:
        raise SystemExit(f"--knob {knob}: no values given")
    return knob, out


def load_designs(path):
    """An explicit list of design points, from a JSON file.

    Each entry is a partial design -- the knobs it moves -- exactly as --knob
    would build it, so a caller that wants a factorial grid or a hand-picked set
    (the GUI does both) writes the points out and the rest of the sweep does not
    know the difference. Unknown knobs are refused here, before any cell runs,
    for the same reason parse_knob refuses them.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--designs {path}: {exc}")
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise SystemExit(f"--designs {path}: want a JSON list of objects, "
                         "each mapping knob names to integers")
    out = []
    for i, d in enumerate(data):
        design = {}
        for knob, value in d.items():
            name = knob.strip().upper()
            if name not in design_mod.DEFAULTS:
                near = difflib.get_close_matches(name, design_mod.DEFAULTS, n=3, cutoff=0.6)
                hint = f" Did you mean {', '.join(near)}?" if near else ""
                raise SystemExit(f"--designs {path}: entry {i}: unknown knob {knob!r}.{hint}")
            if isinstance(value, str):
                try:
                    value = int(value, 0)
                except ValueError:
                    value = None
            if not isinstance(value, int) or isinstance(value, bool):
                raise SystemExit(f"--designs {path}: entry {i}: {knob} is not an integer")
            design[name] = value
        out.append(design)
    return out


def expand(grid_name: str, knobs=None, explicit=None):
    """The design points to run, baseline first.

    The baseline is always included, and always first: every number the report
    quotes is relative to it, so a sweep without it has nothing to compare
    against.
    """
    designs = [{}]
    if explicit is not None:
        designs.extend(explicit)
    elif knobs:
        for knob, values in knobs:
            for v in values:
                designs.append({knob: v})
    else:
        if grid_name != "ofat":
            raise SystemExit(f"unknown grid: {grid_name}")
        for knob, values in OFAT.items():
            for v in values:
                designs.append({knob: v})

    # Drop repeats by what the design actually resolves to, not by how it was
    # written. Sweeping a knob over its own baseline value -- SPATZ_NB_LANES=4
    # when 4 is the default -- otherwise runs the baseline twice and reports it
    # as two cells.
    seen, unique = set(), []
    for d in designs:
        key = design_mod.slug(d)
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique



class Progress:
    """One updating line for the whole sweep, in the style run_hetero.py uses.

    A screening grid is two dozen cells of half a minute each, so the useful
    question during a run is "how far in, and how much longer" -- which a line
    per finished cell answers only in arrears. The bar also names the phase the
    current cell is in, because the phases have very different lengths and a
    stall in one of them looks nothing like a stall in another.

    Falls back to one line per cell when stdout is not a terminal, so piping to
    a file or through `docker exec` without -t stays readable instead of filling
    with carriage returns.
    """

    WIDTH = 24

    def __init__(self, total: int, mode: str):
        self.total = total
        self.done = 0
        self.t0 = time.time()
        self.durations = []
        self.cell = ""
        self.phase = ""
        # json: one event object per line, for a program reading the sweep
        # (the GUI) rather than a person.
        self.json = mode == "json"
        if mode == "auto":
            self.bar = sys.stdout.isatty()
        else:
            self.bar = mode == "bar"
        self._emit({"event": "start", "total": total})

    def _emit(self, event: dict) -> None:
        if self.json:
            print(json.dumps(event), flush=True)

    @staticmethod
    def _clock(seconds) -> str:
        seconds = int(max(0, seconds))
        return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"

    def _eta(self):
        if not self.durations:
            return None
        # Median, not mean: one rebuilt or pathologically slow cell should not
        # drag the estimate for the rest.
        ordered = sorted(self.durations)
        typical = ordered[len(ordered) // 2]
        return typical * (self.total - self.done)

    def start(self, cell: str) -> None:
        self.cell = cell
        self.phase = "starting"
        self._emit({"event": "cell", "index": self.done, "cell": cell})
        self._render()

    def step(self, phase: str) -> None:
        self.phase = phase
        self._emit({"event": "phase", "index": self.done, "phase": phase})
        self._render()

    def finish(self, row: dict) -> None:
        self.done += 1
        if row.get("wall_s"):
            self.durations.append(row["wall_s"])
        if self.json:
            eta = self._eta()
            self._emit({"event": "row", "done": self.done, "total": self.total,
                        "eta_s": round(eta, 1) if eta is not None else None,
                        "row": row})
            return
        if not self.bar:
            cyc = (row.get("cycles_per_image") or row.get("cycles_per_clip")
                   or row.get("cycles"))
            print(f"[{self.done:3}/{self.total}] {row['design_slug'][:34]:34} "
                  f"{row['model']:8} {row['status']:12} cycles={cyc} "
                  f"wall={row.get('wall_s')}s", flush=True)
        elif row["status"] != "ok":
            # A failure must not scroll past behind the bar.
            self._clear()
            why = "; ".join(row.get("reasons", []))[:90]
            print(f"  {row['status']:10} {row['design_slug'][:34]:34} {why}", flush=True)
        self._render()

    def _clear(self) -> None:
        if self.bar:
            sys.stdout.write("\r" + " " * 110 + "\r")

    def _render(self) -> None:
        if not self.bar:
            return
        frac = self.done / self.total if self.total else 0
        filled = int(round(frac * self.WIDTH))
        bar = "\u2588" * filled + "\u2591" * (self.WIDTH - filled)
        eta = self._eta()
        tail = f"  eta ~{self._clock(eta)}" if eta else ""
        # Truncate the whole label, not the cell name: cutting the cell alone
        # left a dangling separator ("design \u00b7  \u00b7 phase").
        label = f"{self.cell} \u00b7 {self.phase}"
        if len(label) > 44:
            label = label[:43] + "\u2026"
        sys.stdout.write(f"\r  [{bar}] {self.done:2}/{self.total}  {frac * 100:3.0f}%  "
                         f"{label:44} {self._clock(time.time() - self.t0)}{tail}   ")
        sys.stdout.flush()

    def done_all(self) -> None:
        self._emit({"event": "done", "done": self.done, "total": self.total,
                    "wall_s": round(time.time() - self.t0, 1)})
        self._clear()


# --- running one cell -------------------------------------------------------

def sh(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def prepare(design, cell_dir, env):
    """Refresh the machine mesh copy, then generate its resolved header."""
    mesh = cell_dir / "mesh"
    if mesh.exists():
        shutil.rmtree(mesh)
    shutil.copytree(MESH, mesh)
    r = sh([PYTHON, ROOT / "pipeline" / "gen_system_header.py", "--out-dir", mesh], env=env)
    if r.returncode:
        raise RuntimeError(f"header generation failed:\n{r.stdout}\n{r.stderr}")
    return mesh


def calibrate(design_dir, mesh, env, host, resolved_experiment, context=None):
    """Return rates and the validated workload-independent input context."""
    rates = design_dir / "rates.json"
    metadata = design_dir / "calibration.json"
    context = context or capture_calibration_context(ROOT, resolved_experiment)
    host_profiles = current_host_profile_catalog(ROOT)
    if host not in host_profiles:
        raise RuntimeError(f"unknown calibration host profile: {host!r}")
    target = host_profiles[host]["target"]
    resolved_target = resolved_experiment.get("simulator", {}).get("target")
    if resolved_target != target:
        raise RuntimeError(
            f"calibration host {host!r} requires target {target!r}, "
            f"not resolved target {resolved_target!r}"
        )
    cached, reason = cache_status(rates, metadata, context["input_fingerprint"])
    if cached:
        return rates, context, reason

    build = design_dir / "calib"
    if build.exists():
        shutil.rmtree(build)
    rates.unlink(missing_ok=True)
    metadata.unlink(missing_ok=True)
    r = sh([PYTHON, ROOT / "pipeline" / "build_mesh.py", "--test", "mesh_calib",
            "--cluster", "cluster_main.c", "--host-extra", "hes_host.c",
            "--host", host, "--mesh-dir", mesh, "--work-dir", build], env=env)
    if r.returncode:
        raise RuntimeError(f"calibration build failed:\n{r.stdout}\n{r.stderr}")

    run_dir = build / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cenv = dict(env)
    cenv["HES_ELF_SNITCH"] = str(build / "snitch" / "snitch.elf")
    cenv["HES_ELF_SPATZ"] = str(build / "spatz" / "spatz.elf")
    cenv["PATH"] = f"{ROOT / '.venv' / 'bin'}:{cenv['PATH']}"
    r = sh([ROOT / "deps/gvsoc/install/bin/gvsoc", f"--target-dir={ROOT / 'targets'}",
            f"--target={target}", f"--binary={build / 'host' / 'host.elf'}", "run"],
           cwd=run_dir, env=cenv)
    log = run_dir / "calib.log"
    log.write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(
            f"calibration simulation failed for this design:\n{r.stdout}\n{r.stderr}"
        )

    r = sh([PYTHON, HERE / "calibrate.py", log, "-o", rates, "--host", host], env=env)
    if r.returncode:
        raise RuntimeError(f"calibration failed for this design:\n{r.stdout}\n{r.stderr}")
    write_metadata(metadata, context, rates, log)
    return rates, context, reason


def resolve_cell_experiment(design, model, host, power, images,
                            frontend=None, serial=False, dram=None):
    """Resolve requested spelling and effective execution before a cell runs."""
    try:
        sample_cap = int(images)
    except (TypeError, ValueError) as error:
        raise ValueError(f"images must be a positive integer, got {images!r}") from error
    if sample_cap <= 0:
        raise ValueError(f"images must be a positive integer, got {images!r}")

    workload_path = resolve_workload_path(model, roots=(ROOT, DEEPLOY_TEST))
    application = None
    if workload_path.is_dir():
        application, _ = detect_app(workload_path)
    effective_frontend = (frontend or "snitch") if application == "kws" else None
    effective_serial = bool(serial) if application == "kws" else False

    request = ExperimentRequest(
        workload=str(model),
        design_overrides=tuple(sorted(design.items())),
        host=host,
        frontend=effective_frontend,
        serial=effective_serial,
        power=bool(power),
        dram=dram,
    )
    resolved = resolve_experiment(
        request,
        design_api=design_mod,
        host_profiles=current_host_profile_catalog(ROOT),
        workload_path=workload_path,
        application=application,
    ).to_dict()
    expected_memory = "fixed" if dram is None else dram
    if resolved["memory"]["kind"] != expected_memory:
        raise RuntimeError(
            f"resolved memory {resolved['memory']['kind']!r} does not match "
            f"operational sweep memory {expected_memory!r}"
        )

    requested = {
        "platform": "m4ia_current",
        "model": str(model),
        "design_overrides": dict(design),
        "host": host,
        "mapping_strategy": request.mapping_strategy,
        "pin": None,
        "frontend": frontend,
        "serial": bool(serial),
        "power": bool(power),
        "dram": dram,
        "dram_overrides": {},
        "images": sample_cap,
    }
    effective = {
        "images": sample_cap,
        "application": application,
        "frontend": effective_frontend,
        "serial": effective_serial,
        "power": bool(power),
        "pin": None,
    }
    return requested, resolved, effective


def run_cell(design, model, out_dir, host, power, images, progress=None,
             frontend=None, serial=False, dram=None):
    """One (design, model) measurement."""
    def phase(name):
        if progress is not None:
            progress.step(name)

    slug = design_mod.slug(design)
    row = {"design_slug": slug, "design": design_mod.resolve(design),
           "model": Path(model).name, "host": host,
           "dram_kind": "fixed" if dram is None else dram}
    if frontend:
        row["frontend"] = frontend
    if serial:
        row["serial"] = True

    bad = design_mod.validate(design)
    if bad:
        row.update(status="invalid", reasons=bad)
        return row

    t0 = time.time()
    try:
        requested, resolved, effective = resolve_cell_experiment(
            design, model, host, power, images,
            frontend=frontend, serial=serial, dram=dram,
        )
        machine = build_machine_context(resolved)
        design_dir = out_dir / "designs" / machine["machine_key"]
        design_dir.mkdir(parents=True, exist_ok=True)
        path = design_mod.write(design, design_dir / "design.json")

        # Numeric design resolution remains owned by design.py. Memory is an
        # explicit, separately resolved machine dimension carried into the
        # operational HES_DESIGN document consumed by every subprocess.
        operational_design = json.loads(path.read_text())
        operational_design["DRAM_KIND"] = resolved["memory"]["kind"]
        if resolved["memory"]["overrides"]:
            operational_design["DRAM_OVERRIDES"] = resolved["memory"]["overrides"]
        path.write_text(json.dumps(operational_design, indent=2, sort_keys=True) + "\n")

        env = dict(os.environ)
        env["HES_DESIGN"] = str(path)
        row["machine_key"] = machine["machine_key"]
        row["dram_kind"] = resolved["memory"]["kind"]
        if effective["frontend"] is not None:
            row["effective_frontend"] = effective["frontend"]

        phase("mesh + header")
        mesh = prepare(design, design_dir, env)
        phase("calibration cache check")
        rates, calibration_context, cache_reason = calibrate(
            design_dir, mesh, env, host, resolved,
        )
        env["HES_RATES"] = str(rates)
        row["calibration_cache"] = cache_reason

        run_provenance = capture_run_provenance(
            ROOT, resolved, include_sweep=True,
        )
        calibration_provenance = calibration_context.get("provenance", {})
        run_provenance["execution_environment"] = {
            key: calibration_provenance[key]
            for key in ("dependencies", "patches", "toolchain", "simulator_binary")
            if key in calibration_provenance
        }
        run_input = build_run_input_context(
            resolved,
            calibration_input_fingerprint=calibration_context["input_fingerprint"],
            run_source_set_digest=run_provenance["source_set"]["digest"],
            execution_controls=effective,
            workload_label=Path(model).name,
        )
        artifact_key = run_input["artifact_key"]
        row["artifact_key"] = artifact_key
        row["run_input_fingerprint"] = run_input["run_input_fingerprint"]

        cell = out_dir / "cells" / artifact_key
        cell.mkdir(parents=True, exist_ok=True)
        result = cell / "result.json"
        manifest_path = cell / "manifest.json"
        # A failed new invocation must never be represented by artifacts from
        # an older invocation with the same pre-run identity.
        result.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        cmd = [PYTHON, ROOT / "pipeline" / "run_hetero.py", model,
               "--host", host, "--mesh-dir", mesh, "--tag", artifact_key,
               "--out", result, "--images", effective["images"], "-q"]
        if power:
            cmd.append("--power")
        if effective["frontend"]:
            cmd += ["--frontend", effective["frontend"]]
        if effective["serial"]:
            cmd.append("--serial")
        phase("codegen + build + simulate")
        r = sh(cmd, env=env)
        if not result.exists():
            row.update(status="failed", reasons=[(r.stdout + r.stderr)[-1500:]])
            return row
        result_doc = json.loads(result.read_text())
        res = result_doc["result"]
        row["status"] = res.get("status", "unknown")

        manifest = write_run_manifest(
            manifest_path,
            requested=requested,
            resolved={
                "experiment": resolved,
                "machine": machine,
                "execution": effective,
            },
            run_input=run_input,
            artifact_key=artifact_key,
            calibration_path=design_dir / "calibration.json",
            result_path=result,
            run_provenance=run_provenance,
            base_dir=out_dir,
        )
        row["run_fingerprint"] = manifest["run_fingerprint"]
        row["manifest"] = str(manifest_path.relative_to(out_dir))
        for k in ("cycles", "cycles_per_image", "cycles_per_clip", "accuracy",
                  "offload_failures", "maxdiff", "caches", "per_engine_cycles", "dram",
                  # KWS runs both clusters at once, so its result is a max
                  # rather than a sum. Dropping these would leave a sweep over
                  # cluster geometry unable to say whether a design improved the
                  # overlap, which is the whole point of that workload.
                  "frontend_engine", "frontend_busy", "frontend_wait",
                  "hidden_cycles", "cluster_busy", "pipelined", "images", "clips"):
            if k in res:
                row[k] = res[k]
        if row["status"] == "ok" and res.get("offload_failures"):
            row["status"] = "wrong-result"
    except Exception as exc:                        # noqa: BLE001
        row.update(status="error", reasons=[str(exc)])
        return row
    finally:
        row["wall_s"] = round(time.time() - t0, 1)

    # Area is a pure function of the design, so it costs nothing to attach and
    # makes every row self-contained for the Pareto pass.
    phase("scoring")
    try:
        parts = area_model.breakdown({"_path": str(path)})
        row["area_au"] = sum(parts.values())
        row["area_breakdown"] = parts
        row["area_coefficients_sourced"] = all(
            c["sourced"] for c in area_model.COEFFS.values())
    except Exception as exc:                        # noqa: BLE001
        row["area_error"] = str(exc)

    if "caches" in row and row["caches"]:
        pj = [c.get("dynamic_pj") for c in row["caches"]]
        row["cache_dynamic_pj"] = sum(p for p in pj if p) if any(pj) else None

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="op directories")
    ap.add_argument("--grid", default="ofat",
                    help="named grid to run when no --knob is given (default: %(default)s)")
    ap.add_argument("--designs", default=None, metavar="FILE",
                    help="run exactly these design points: a JSON list of "
                         "objects mapping knobs to values, e.g. "
                         "[{\"SPATZ_NB_LANES\": 8, \"SPATZ_NB_CORE\": 17}]. "
                         "Use it for factorial or hand-picked grids. The "
                         "baseline is still prepended. Excludes --knob/--grid.")
    ap.add_argument("--knob", action="append", default=[], metavar="KNOB=v1,v2",
                    help="sweep just this knob over these values, instead of the "
                         "named grid. Repeatable. The baseline is always included, "
                         "since every reported figure is relative to it. "
                         "e.g. --knob SPATZ_NB_LANES=2,4,8")
    ap.add_argument("-o", "--out", required=True, help="sweep output directory")
    ap.add_argument("--host", default="cva6", choices=("cva6", "ara"))
    ap.add_argument("--power", action="store_true", help="measure energy too (slower)")
    ap.add_argument("--frontend", choices=("snitch", "spatz"), default=None,
                    help="for keyword spotting: which cluster runs the MFCC "
                         "front-end while the other runs the classifier")
    ap.add_argument("--serial", action="store_true",
                    help="for keyword spotting: take turns instead of pipelining, "
                         "which is the comparison that shows what the overlap buys")
    ap.add_argument("--images", default="16", help="samples per model run")
    ap.add_argument("--dram", choices=("fixed", "lpddr4", "lpddr4x", "lpddr5", "hyperram"),
                    default=None,
                    help="main-memory device for every cell (default: fixed latency). Not a "
                         "knob: run one sweep per device, into separate --out directories")
    ap.add_argument("--limit", type=int, default=None, help="stop after N designs")
    ap.add_argument("--progress", choices=("auto", "bar", "lines", "json"), default="auto",
                    help="auto uses a bar on a terminal and one line per cell "
                         "otherwise; json prints one event object per line for a "
                         "program to read, and moves every other message to "
                         "stderr (default: %(default)s)")
    args = ap.parse_args()
    if args.designs and args.knob:
        ap.error("--designs and --knob both choose the design points; give one")

    # In json mode stdout carries only events, so a reader can parse every line.
    say = (lambda *a: print(*a, file=sys.stderr, flush=True)) \
        if args.progress == "json" else print

    # Resolved, not as given: every cell path derives from this, and a relative
    # spelling made build_mesh.py fail when it tried to report an ELF path
    # relative to the repository root.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    knobs = [parse_knob(k) for k in args.knob]
    explicit = load_designs(args.designs) if args.designs else None
    designs = expand(args.grid, knobs, explicit)
    if explicit is not None:
        say(f"sweeping {len(designs)} design points from {args.designs}")
    elif knobs:
        say("sweeping " + "; ".join(
            f"{k} over {', '.join(str(v) for v in vs)}" for k, vs in knobs))
    if args.limit:
        designs = designs[:args.limit]

    # Designs needing a build we do not have would otherwise run against
    # whatever VLEN happens to be compiled in, and report it under the label
    # they asked for. Refuse instead.
    have = design_mod.build_key({})
    needs_build = sorted({design_mod.build_key(d) for d in designs} - {have})
    if needs_build:
        raise SystemExit(
            "error: these design points need GVSoC builds that do not exist: "
            + ", ".join(needs_build)
            + "\n  VLEN is compiled into the model, so running them now would "
              "measure the built VLEN and label it as the requested one.\n"
              "  Build them first, or drop the VLEN axis from the grid.")

    jsonl = out / "sweep.jsonl"
    total = len(designs) * len(args.models)
    prog = Progress(total, args.progress)
    n = 0
    with jsonl.open("w") as fh:
        for d in designs:
            for model in args.models:
                # The slug's trailing hash disambiguates directories, not
                # humans; drop it for the display only.
                shown = design_mod.slug(d)
                if shown != "baseline":
                    shown = shown.rsplit("-", 1)[0]
                prog.start(f"{shown} \u00b7 {Path(model).name}")
                row = run_cell(d, model, out, args.host, args.power, args.images,
                               progress=prog, frontend=args.frontend,
                               serial=args.serial, dram=args.dram)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                prog.finish(row)
    prog.done_all()

    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    ok = sum(1 for r in rows if r["status"] == "ok")
    say(f"\n{n} rows -> {jsonl}")
    say(f"{ok} ok, {n - ok} not, in {Progress._clock(time.time() - prog.t0)}")


if __name__ == "__main__":
    main()
