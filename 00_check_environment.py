#!/usr/bin/env python3
"""Check the Windows research environment without PowerShell python -c quoting.

This script does not generate, delete, or change any existing study data.
Exit code 0 means the basic interpreter/dependency checks passed.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import platform
import sys
import traceback


def main() -> int:
    print("=" * 68, flush=True)
    print("GIORNO-TSALLIS HMM-6 / APARCH - PYTHON ENVIRONMENT CHECK", flush=True)
    print("=" * 68, flush=True)
    print("Python executable:", sys.executable, flush=True)
    print("Python version:   ", sys.version.splitlines()[0], flush=True)
    print("Platform:         ", platform.platform(), flush=True)
    print("Package folder:   ", Path(__file__).resolve().parent, flush=True)
    failures = []
    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or later is required")

    dependencies = ("numpy", "pandas", "scipy", "numba")
    for name in dependencies:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {name:8} version {version}", flush=True)
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            failures.append(f"{name}: {exc}")

    if not failures:
        try:
            import numpy as np
            from scipy.special import logsumexp
            from numba import njit
            @njit(cache=False)
            def _test_sum(a):
                result = 0.0
                for x in a:
                    result += x
                return result
            assert abs(float(_test_sum(np.array([1.0, 2.0]))) - 3.0) < 1e-12
            assert abs(float(logsumexp(np.array([0.0, 0.0]))) - float(np.log(2.0))) < 1e-12
            print("[OK] SciPy numerical test and Numba JIT compilation", flush=True)
        except Exception as exc:
            print(f"[FAIL] Numerical/JIT smoke test: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            failures.append(f"Numba/SciPy numerical smoke test: {exc}")

    if failures:
        print("\nENVIRONMENT CHECK FAILED. The research stages have NOT started.", flush=True)
        print("Install/repair dependencies in THE SAME interpreter shown above:", flush=True)
        print(f'  "{sys.executable}" -m pip install --upgrade -r "{Path(__file__).resolve().parent / "requirements.txt"}"', flush=True)
        print("Then rerun RUN_RESEARCH_WINDOWS.ps1. Existing saved results remain untouched.", flush=True)
        return 2

    print("\nENVIRONMENT CHECK PASSED. Safe to proceed with research stages.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
