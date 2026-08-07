#!/usr/bin/env python
"""One command that dumps everything needed to diagnose a broken setup.

    python scripts/doctor.py                 # from inside the QFormerRec checkout
    python scripts/doctor.py --paths ...      # also check data/ckpt paths

Prints, in order: which QFormerRec revision is checked out and whether the
post-fix files are present, the live library versions, the known-bad-combination
check, whether `minigpt4` imports at all, and whether the registry got populated.
Paste the whole output when reporting a problem.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:                                     # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


def section(t):
    print(f"\n=== {t} " + "=" * max(0, 60 - len(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--mf_ckpt", default=None)
    ap.add_argument("--memory_index", default=None)
    args = ap.parse_args()

    section("this checkout")
    print("QFormerRec root :", ROOT)
    print("revision        :", sh("git log --oneline -1", cwd=ROOT) or "<not a git checkout>")
    print("dirty files     :", sh("git status --porcelain", cwd=ROOT) or "(clean)")
    print("remote          :", sh("git remote get-url origin", cwd=ROOT))
    # the files that only exist after the Python-3.12 / point-in-time fixes
    expected = {
        "qformerrec/compat.py": "decord shim + env check",
        "qformerrec/metrics.py": "sklearn-independent UAUC",
        "scripts/check_pit_history.py": "point-in-time history gate",
    }
    for rel, what in expected.items():
        ok = os.path.exists(os.path.join(ROOT, rel))
        print(f"  {'OK ' if ok else 'MISSING'} {rel:32s} ({what})")
    if not all(os.path.exists(os.path.join(ROOT, r)) for r in expected):
        print("  --> this checkout PREDATES the fixes. `git pull` in the checkout you are"
              "\n      actually running, and note that `git pull` does NOT update a"
              "\n      notebook you already have open in Colab.")

    section("pinned requirements as checked out")
    req = os.path.join(ROOT, "requirements.txt")
    if os.path.exists(req):
        for line in open(req):
            line = line.strip()
            if line and not line.startswith("#"):
                print("  " + line)

    section("live environment")
    print("python          :", sys.version.replace("\n", " "))
    print("executable      :", sys.executable)
    print("COLLM_ROOT      :", os.environ.get("COLLM_ROOT", "<unset>"))
    for mod in ["torch", "transformers", "tokenizers", "peft", "sklearn", "numpy",
                "pandas", "scipy", "omegaconf", "decord"]:
        try:
            m = __import__(mod)
            extra = " (SHIM)" if getattr(m, "__qformerrec_shim__", False) else ""
            print(f"  {mod:14s} {getattr(m, '__version__', '?')}{extra}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  {mod:14s} NOT IMPORTABLE ({type(e).__name__})")
    try:
        import torch
        print("  cuda available :", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception:                                          # noqa: BLE001
        pass

    section("known-bad-combination check")
    try:
        from qformerrec.compat import check_environment, install_import_shims
        install_import_shims()
        problems = check_environment()
        print("  ->", "no known problems" if not problems else f"{len(problems)} PROBLEM(S) above")
        if problems:
            print("  (this is the check that catches transformers 5.x, which imports and")
            print("   loads fine and then silently produces NaN)")
    except Exception as e:                                      # noqa: BLE001
        print(f"  cannot import qformerrec.compat: {type(e).__name__}: {e}")
        print("  --> run this from inside the QFormerRec checkout")
        return

    section("does minigpt4 import?")
    collm = os.path.abspath(os.environ.get("COLLM_ROOT", os.path.join(ROOT, "..", "CoLLM")))
    print("  trying COLLM_ROOT =", collm, "(exists:", os.path.isdir(collm), ")")
    sys.path.insert(0, collm)
    try:
        import minigpt4  # noqa: F401
        print("  OK  minigpt4 imported")
    except Exception as e:                                      # noqa: BLE001
        import traceback
        print(f"  FAIL {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
        return

    section("registry after importing qformerrec")
    try:
        from minigpt4.common.registry import registry
        import qformerrec.datasets.rec_datasets_qformer  # noqa: F401
        import qformerrec.models.minigpt4rec_qformer  # noqa: F401
        import qformerrec.runners.runner_qformer  # noqa: F401
        import qformerrec.tasks.rec_qformer_task  # noqa: F401
        for kind, name in [("model", "mini_gpt4rec_qformer"), ("task", "rec_qformer"),
                           ("runner", "rec_runner_qformer"), ("builder", "movie_ood_qf"),
                           ("builder", "amazon_ood_qf"),
                           ("lr_scheduler", "linear_warmup_cosine_lr_scaled")]:
            got = getattr(registry, f"get_{kind}_class")(name)
            print(f"  {'OK ' if got else 'MISSING'} {kind}: {name}")
    except Exception as e:                                      # noqa: BLE001
        import traceback
        print(f"  FAIL {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
        return

    section("paths")
    for label, path in [("data_dir", args.data_dir), ("mf_ckpt", args.mf_ckpt),
                        ("memory_index", args.memory_index)]:
        if not path:
            continue
        print(f"  {label:13s} {path}  exists={os.path.exists(path)}")
        if label == "data_dir" and os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.endswith(".pkl"):
                    print(f"      {f}  {os.path.getsize(os.path.join(path, f)) / 1e6:.1f} MB")
    if args.memory_index and os.path.exists(args.memory_index):
        import pickle
        with open(args.memory_index, "rb") as f:
            mi = pickle.load(f)
        print("  memory index keys:", sorted(mi))
        print("  has item_in_train:", "item_in_train" in mi,
              "(required since the point-in-time change)")
        print("  meta:", mi.get("meta", {}))

    print("\nall checks completed -- paste this whole output when reporting a problem")


if __name__ == "__main__":
    main()
