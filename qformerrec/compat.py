"""Import shims for running CoLLM on a modern Python.

CoLLM's ``minigpt4/__init__.py`` eagerly imports the whole package, including
video-dataset plumbing we never touch. One of those imports, ``decord``, has no
wheel for Python >= 3.11 (last release 0.6.0, 2021, cp36-cp310 only), so on a
current Colab runtime ``pip install decord`` fails outright.

Rather than pin an ancient Python, install a stub for the unused modules before
``minigpt4`` is imported. Call :func:`install_import_shims` at the top of every
entry point, ahead of any ``minigpt4`` import.

Nothing here changes behaviour for code that has the real package installed: the
shim is only inserted when the genuine import fails.
"""

import importlib.machinery
import importlib.util
import logging
import sys
import types

# modules CoLLM imports at package-import time but this project never calls into
_SHIMMABLE = ("decord",)


def _make_decord_stub():
    mod = types.ModuleType("decord")
    # transformers' import_utils calls importlib.util.find_spec("decord"), which
    # raises ValueError if __spec__ is None -- so give the stub a real spec.
    mod.__spec__ = importlib.machinery.ModuleSpec("decord", None)
    mod.__version__ = "0.0.0-shim"
    mod.__qformerrec_shim__ = True

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "decord is not installed; this is a stub from qformerrec.compat. "
            "Video datasets are not used by CoLLM-QFormer."
        )

    mod.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
    mod.VideoReader = _unavailable
    mod.VideoLoader = _unavailable
    mod.cpu = mod.gpu = _unavailable
    return mod


_FACTORIES = {"decord": _make_decord_stub}


def install_import_shims(verbose=True):
    """Stub the unused-but-imported modules that are unavailable on this Python.

    Returns the list of module names that were stubbed.
    """
    shimmed = []
    for name in _SHIMMABLE:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue                      # the real thing is installed
        except (ImportError, ValueError):
            pass
        sys.modules[name] = _FACTORIES[name]()
        shimmed.append(name)
    if shimmed and verbose:
        msg = (f"[compat] stubbed unused modules: {', '.join(shimmed)} "
               "(imported by CoLLM, never called here)")
        print(msg)
        logging.info(msg)
    return shimmed


def check_environment(strict=False):
    """Report the versions that actually matter, and flag known-bad combinations.

    Cheap to call and worth calling: the failure modes it catches (a peft without
    ``prepare_model_for_int8_training``, a transformers whose ``utils`` no longer
    exports the docstring decorators CoLLM's vendored ``modeling_llama`` imports)
    otherwise surface as confusing ImportErrors deep inside CoLLM.
    """
    import platform

    problems, notes = [], []

    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:                                  # noqa: BLE001
            return None

    py = platform.python_version()
    tf, pf, tk = _v("transformers"), _v("peft"), _v("tokenizers")
    th, sk = _v("torch"), _v("sklearn")
    notes.append(f"python={py} torch={th} transformers={tf} peft={pf} "
                 f"tokenizers={tk} scikit-learn={sk}")

    # CoLLM's vendored modeling_llama needs these from transformers.utils
    try:
        from transformers.utils import (  # noqa: F401
            add_start_docstrings,
            add_start_docstrings_to_model_forward,
            replace_return_docstrings,
        )
    except ImportError as e:
        problems.append(
            f"transformers {tf} no longer exports the docstring helpers CoLLM's "
            f"vendored modeling_llama imports ({e}). Use transformers 4.3x."
        )

    # minigpt4rec_v2 imports this by name; peft removed it in 0.10.0
    try:
        from peft import prepare_model_for_int8_training  # noqa: F401
    except ImportError:
        problems.append(
            f"peft {pf} does not export `prepare_model_for_int8_training`, which "
            "CoLLM's minigpt4rec_v2 imports by name (removed in peft 0.10.0). "
            "Pin peft<=0.9.0."
        )

    for line in notes:
        print(f"[compat] {line}")
    for p in problems:
        print(f"[compat] PROBLEM: {p}")
        logging.warning("compat: %s", p)
    if problems and strict:
        raise RuntimeError("incompatible environment; see the [compat] lines above")
    return problems
