# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Cases of the test_fixture_*.py tests, from recorded outputs or from a reference model."""

import atexit
import fnmatch
import glob
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import types


class _Absent(types.ModuleType):
    """Stands in for a module the generating interpreter does not have."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Absent(name)

    def __call__(self, *a, **k):
        return self


if __name__ == "__main__":  # the generating side imports a test file without its test modules
    sys.modules.update(fixture=sys.modules[__name__], npu=_Absent("npu"),
                       pytest=_Absent("pytest"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from npu import (I64, NPU_BF16, NPU_F16, NPU_F32, NPU_FP8, NPU_I8, NPU_I16,  # noqa: E402
                 NPU_I32, NPU_I64, NPU_U8, NPU_U16, NPU_U32, NPU_U64, OUT, call)

# type name of a case -> (entry type code, numpy storage type)
TYPES = {"float": (NPU_F32, np.float32), "float32": (NPU_F32, np.float32),
         "float16": (NPU_F16, np.uint16), "bfloat16": (NPU_BF16, np.uint16),
         "float8": (NPU_FP8, np.uint8),
         "int8": (NPU_I8, np.int8), "uint8": (NPU_U8, np.uint8),
         "int16": (NPU_I16, np.int16), "uint16": (NPU_U16, np.uint16),
         "int32": (NPU_I32, np.int32), "uint32": (NPU_U32, np.uint32),
         "int64": (NPU_I64, np.int64), "uint64": (NPU_U64, np.uint64)}
NPU_TYPE = {k: v[0] for k, v in TYPES.items()}
# the short name the golden modules use for a type
SHORT = {"float": "f32", "float32": "f32", "float16": "f16", "bfloat16": "bf16", "float8": "fp8",
         "int8": "i8", "uint8": "u8", "int16": "i16", "uint16": "u16", "int32": "i32",
         "uint32": "u32", "int64": "i64", "uint64": "u64"}
INT = {k: (*TYPES[k], SHORT[k]) for k in ("int8", "uint8", "int16", "uint16")}
HALF = {k: (TYPES[k][0], SHORT[k]) for k in ("bfloat16", "float16")}

# NPU_EMU_SIM_ROOT: recorded outputs. NPU_EMU_MODEL_REPO: a model repository whose reference model
# generates each case when it is first used (interpreter NPU_EMU_MODEL_PYTHON, cache directory
# NPU_EMU_FIXTURE_CACHE). NPU_EMU_FIXTURE_LIMIT: at most this many cases per directory.
ROOT = os.environ.get("NPU_EMU_SIM_ROOT") or None
MODEL = os.environ.get("NPU_EMU_MODEL_REPO") or None
PYTHON = os.environ.get("NPU_EMU_MODEL_PYTHON") or sys.executable
LIMIT = int(os.environ.get("NPU_EMU_FIXTURE_LIMIT") or 0)
CACHE = os.environ.get("NPU_EMU_FIXTURE_CACHE") or None
if CACHE is None:  # a directory of this run, created by the first generated case
    CACHE = os.path.join(tempfile.gettempdir(), f"npu_emu_fixture_{os.getpid()}")
    atexit.register(shutil.rmtree, CACHE, True)
HERE = os.path.dirname(os.path.abspath(__file__))

# Names of the model repository and of the recorded tree.
LAYOUT = {
    "dirs": "L2/regression/wrapper",                # regression directories of the repository
    "list": "description_population.json",          # per directory: testcase file and parts
    "generated": "_Generated.json",                 # suffix of the generated testcase file
    "template": r"_Template(_\d+)?$",               # testcase keys that are not cases
    "generator": "internal/models/python/restructured/app/tv_gen.py",
    "arguments": ("--use_json", "--key", "--part"),
    "case": "XDNA_{key}_AIE_Test_{part}",           # case directory of a generated case
    "runs": ("XDNA_*_AIE_Test_*", "AIE4_*_regression_*"),  # case directories of recorded cases
    "name": r"^(?:XDNA_(.+?)_AIE_Test|AIE4_(.+?)_regression_).*$",  # the case name in a case dir
    "subdirs": (("", ""), ("generated_cases/", "gen_"), ("perf/", "perf_")),  # (path, id prefix)
    "perf": "_perfgroup_",                          # marks the keys of the perf_ cases
    "recorded": "ofm32.txt",                        # expected output of a recorded case
    "ofm": "ofm32_ref.txt",                         # expected output of a generated case
    "tensors": "fixture.npz",                       # what extract() made of a generated case
}

_live = {}  # case directory -> [testcase file, key, part, test file with extract() or None, error]


class Skip(Exception):
    """Raised by extract() for a case the test cannot handle."""


def ofm(case):
    """Name of the expected-output file of a case."""
    return LAYOUT["ofm" if case in _live else "recorded"]


def case_name(case):
    """The case name of a case directory."""
    return re.sub(LAYOUT["name"], r"\1\2", os.path.basename(case))


def recorded_cases(d):
    """(id prefix, case directory) of every case of d that has a recorded output."""
    return [(prefix, c) for sub, prefix in LAYOUT["subdirs"]
            for c in sorted(c for runs in LAYOUT["runs"]
                            for c in glob.glob(f"{ROOT}/{d}/{sub}*/{runs}"))
            if os.path.exists(f"{c}/data/{LAYOUT['recorded']}")
            and os.path.exists(f"{c}/testcase.json")]


def model_cases(d, test_file):
    """(id prefix, case directory) of every case the model repository lists for d."""
    try:
        with open(os.path.join(MODEL, LAYOUT["dirs"], d, LAYOUT["list"])) as f:
            listing = json.load(f)
    except OSError:
        return []
    path = os.path.join(MODEL, listing["case_population"]["test_config_path"])
    out = []
    for p, generated in ((path, False), (path[:-len(".json")] + LAYOUT["generated"], True)):
        if not os.path.exists(p):
            continue
        with open(p) as f:
            keys = [k for k, v in json.load(f).items()
                    if not re.search(LAYOUT["template"], k)
                    and not (isinstance(v, dict) and v.get("case_population", {}).get("skip"))]
        for key in keys:
            prefix = "gen_" if generated else "perf_" if LAYOUT["perf"] in key else ""
            for part in listing["part_allowlist"]:
                case = os.path.join(CACHE, d, LAYOUT["case"].format(key=key, part=part))
                _live[case] = [p, key, part, test_file, None]
                out.append((prefix, case))
    return out


def cases(dirs, test_file=None):
    """pytest params (dir, case) for every case of the given dirs. With test_file the cases are
    the tensors its extract() makes of the reference model objects (see load)."""
    out = []
    live = MODEL and (test_file or not ROOT)
    if __name__ == "__main__" or not (live or ROOT and not test_file):
        return out
    seen = set()
    for d in dirs:
        found = model_cases(d, test_file) if live else recorded_cases(d)
        for prefix, c in found[:LIMIT or None]:
            cid = base = prefix + case_name(c)
            k = 2
            while cid in seen:  # several runs of one case
                cid, k = f"{base}_{k}", k + 1
            seen.add(cid)
            out.append(pytest.param(d, c, id=cid))
    return out


def prepare(case):
    """Generates a model case on first use; called before each test."""
    if case not in _live or os.path.exists(os.path.join(case, "PASS")):
        return
    path, key, part, test_file, error = _live[case]
    if error is None:
        shutil.rmtree(case, ignore_errors=True)
        os.makedirs(case)
        cmd = [os.path.join(MODEL, LAYOUT["generator"])]
        cmd += [x for pair in zip(LAYOUT["arguments"], (path, key, part)) for x in pair]
        if test_file:
            cmd = [os.path.abspath(__file__), test_file] + cmd
        with open(os.path.join(case, "generate.log"), "w") as log:
            rc = subprocess.run([PYTHON, "-B"] + cmd, cwd=case, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        made = LAYOUT["tensors"] if test_file else os.path.join("data", LAYOUT["ofm"])
        if rc == 0 and os.path.exists(os.path.join(case, made)):
            open(os.path.join(case, "PASS"), "w").close()
            return
        with open(os.path.join(case, "generate.log"), errors="replace") as log:
            lines = [line.rstrip() for line in log if line.strip()]
        error = _live[case][4] = f"reference model failed ({rc}):\n" + "\n".join(lines[-20:])
    pytest.fail(error, pytrace=False)


def load(case):
    """The tensors of a case of cases(dirs, test_file); skips a case extract() raised Skip for."""
    z = np.load(os.path.join(case, LAYOUT["tensors"]), allow_pickle=False)
    if "skip" in z:
        pytest.skip(str(z["skip"]))
    return z


def opt(z, k):
    return z[k] if k in z else None


def stream(path, dtype=np.uint8):
    with open(path) as f:
        words = [int(line, 16) for line in f if line.strip()]
    return np.array(words, np.uint32).view(np.uint8).view(dtype)


def data(case, name, dtype=np.uint8):
    return stream(os.path.join(case, "data", name), dtype)


def words(case, name="coeff32.txt"):
    return data(case, name, np.uint32)


def fword(case, i, name="coeff32.txt"):
    return float(words(case, name)[i:i + 1].view(np.float32)[0])


def ty(t):
    return TYPES[t]


def testcase(case):
    """The parameters of a case (its testcase.json)."""
    with open(os.path.join(case, "testcase.json")) as f:
        return json.load(f)


def rtp_words(case):
    """The generated_rtps[] words of a case's kernel_rtp.h as uint32, or None without the file."""
    p = os.path.join(case, "kernel_rtp.h")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        body = re.search(r"generated_rtps[^{]*\{([^}]*)\}", f.read()).group(1)
    return np.array([int(w, 16) for w in re.findall(r"0x[0-9A-Fa-f]+", body)], np.uint32)


def bf16_bits(x):
    """bf16 codes of bf16 values held as float32."""
    return (np.ascontiguousarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def col(v, n):
    """A scalar or a row of float32 coefficients as n columns."""
    return np.ascontiguousarray(np.broadcast_to(np.asarray(v, np.float32).reshape(-1), (n,)))


def run_unary(case, t, op, bits, k):
    """(got, want) of npu_elementwise_<op>_a<bits> on the case's input, with the first k
    coefficient words as float arguments."""
    it, idt = TYPES[t["ifm_dtype"]]
    ot, odt = TYPES[t["ofm_dtype"]]
    x = data(case, "ifm32.txt", idt)
    c = [float(v) for v in words(case)[:k].view(np.float32)]
    got = call(f"npu_elementwise_{op}_a{bits}", x, it, OUT, ot, I64(x.size), *c,
               out_nbytes=x.size * np.dtype(odt).itemsize, out_dtype=odt)
    return got, data(case, ofm(case), odt)[:x.size]


def tensor_descr(case):
    """(valid_shape, physical_shape, dtype) from a case's tensor_descr.h, or None."""
    p = os.path.join(case, "tensor_descr.h")
    if not os.path.exists(p):
        return None
    valid = phys = dt = None
    for line in open(p):
        line = line.strip()
        if line.startswith("#define OFM_VALID_SHAPE"):
            valid = tuple(int(x) for x in line.split("{", 1)[1].split("}", 1)[0].split(","))
        elif line.startswith("#define OFM_PHYSICAL_SHAPE"):
            phys = tuple(int(x) for x in line.split("{", 1)[1].split("}", 1)[0].split(","))
        elif line.startswith("#define DTYPE"):
            dt = line.split()[-1]
    return valid, phys, dt


def assert_bytes(got, want):
    got = np.asarray(got).view(np.uint8).reshape(-1)
    want = np.asarray(want).view(np.uint8).reshape(-1)
    assert got.size == want.size, f"size {got.size} != {want.size}"
    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (f"{bad.size}/{want.size} bytes differ, first at "
                           f"{bad[:8].tolist()}: got {got[bad[:8]].tolist()} "
                           f"want {want[bad[:8]].tolist()}")


def check(got, want, ref=None):
    """got, and the reference model output ref (cast to the type of got), equal want bytewise."""
    assert_bytes(got, want)
    if ref is not None:
        assert_bytes(np.asarray(ref).astype(np.asarray(got).dtype), want)


# ---- the generating side (runs under PYTHON, in the case directory) ----

def last(seen, name="compute_bit_exact"):
    """The model object of the last recorded call of a method."""
    return seen[name][-1][0][0]


def args(seen, name, i=-1):
    """Positional arguments of a recorded call."""
    return seen[name][i][0]


def _record(owner, name, seen):
    f = owner.__dict__.get(name)
    if not isinstance(f, types.FunctionType) or getattr(f, "_recorded", False):
        return

    def w(*a, **k):
        out = f(*a, **k)
        seen.setdefault(name, []).append((a, k, out))
        return out
    w._recorded = True
    setattr(owner, name, w)


class _Recorder(importlib.abc.MetaPathFinder):
    """Records calls of the named functions / methods of matching modules as they are imported."""

    def __init__(self, hooks, seen):
        self.hooks, self.seen = hooks, seen

    def find_spec(self, name, path, target=None):
        short = name.rsplit(".", 1)[-1]
        names = [n for pat, ns in self.hooks.items() if fnmatch.fnmatchcase(short, pat) for n in ns]
        if not names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            owners = [module] + [v for v in vars(module).values() if isinstance(v, type)]
            for owner in owners:
                for n in names:
                    _record(owner, n, self.seen)
        spec.loader.exec_module = exec_module
        return spec


def _main(test_file, generator, *arguments):
    """Runs the generator and saves what extract() of the test file makes of its model objects."""
    top = os.path.dirname(HERE)
    sys.path += [os.path.join(top, n) for n in sorted(os.listdir(top))
                 if os.path.isdir(os.path.join(top, n))]
    spec = importlib.util.spec_from_file_location("fixture_case", test_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen, case = {}, os.getcwd()
    sys.meta_path.insert(0, _Recorder(mod.HOOKS, seen))
    sys.argv = [generator, *arguments]
    try:
        runpy.run_path(generator, run_name="__main__")
        fx = mod.extract(os.path.basename(os.path.dirname(case)), seen)
    except Skip as e:
        fx = {"skip": str(e)}
    except AttributeError as e:
        if "compute_bit_exact" not in seen:
            raise
        fx = {"skip": f"reference model revision lacks {getattr(e, 'name', None) or e}"}
    np.savez_compressed(os.path.join(case, LAYOUT["tensors"]),
                        **{k: np.asarray(v) for k, v in fx.items() if v is not None})


if __name__ == "__main__":
    _main(*sys.argv[1:])
