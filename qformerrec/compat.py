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
import os
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


def enable_live_output():
    """Make stdout/stderr line-buffered so training progress appears as it happens.

    Python block-buffers stdout (8 KB) whenever it is not a TTY, which is exactly
    what Colab's ``!python ...`` and any ``> file`` redirect create. CoLLM's
    ``MetricLogger.log_every`` reports progress with ``print()``, so on a long run
    the output sits in that buffer and the cell looks frozen -- measured here:
    a run's output stalled for 20 s and then dumped 5 KB at process exit.

    ``line_buffering=True`` fixes it globally, for CoLLM's prints as well as ours,
    with no need for ``python -u``.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)             # py3.7+
        except (AttributeError, ValueError):                    # already wrapped
            pass


def attach_file_log(output_dir, filename="train.log"):
    """Also write every ``logging`` record to ``<output_dir>/<filename>``.

    Worth having on top of line buffering: CoLLM's ``setup_logger`` installs only a
    ``StreamHandler``, so nothing survives a disconnected Colab session. A file in
    the run directory gets picked up by the S3/Drive sync along with the
    checkpoint and the diagnostics.
    """
    path = os.path.join(str(output_dir), filename)
    root = logging.getLogger()
    for h in root.handlers:                                     # idempotent
        if isinstance(h, logging.FileHandler) and \
                os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(path):
            return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    logging.info("logging to %s", path)
    print(f"[compat] logging to {path}")
    return path


def check_vendored_llama():
    """Numerically verify CoLLM's vendored LLaMA on the installed transformers.

    This is the check that matters, and neither a version test nor an import test
    can replace it.

    The concrete failure on transformers 5.x: its loader reports
    ``self_attn.rotary_emb.inv_freq | MISSING`` and then leaves the vendored
    ``LlamaRotaryEmbedding``'s ``cos_cached`` / ``sin_cached`` **buffers** as
    uninitialised memory (values like 1e+24 and 6e-38). transformers 4.x instead
    re-runs the module init and fills them correctly. Because those are buffers
    rather than parameters, ``named_parameters()`` reports everything finite and
    the corruption only surfaces as ``loss=nan`` after a full model build --
    an expensive and very confusing way to find out.

    Reproducing it needs the *realistic* load: a checkpoint written by
    **transformers' own** ``LlamaForCausalLM`` (which, like every real Vicuna
    checkpoint, contains no ``rotary_emb.inv_freq`` key) loaded by the **vendored**
    class. Saving with the vendored class instead would write ``inv_freq`` into the
    checkpoint, nothing would be MISSING, and the bug would stay hidden -- as would
    building straight from a config, since ``__init__`` computes the caches
    properly. The probe also has to inspect buffers, not just parameters. A
    2-layer, 32-dim model keeps this well under a second, with no download.

    Returns a problem string, or None if healthy / not checkable here.
    """
    try:
        import tempfile

        import torch
        import transformers
        from transformers import LlamaConfig
        from transformers import LlamaForCausalLM as HFLlamaForCausalLM

        from minigpt4.models.modeling_llama import LlamaForCausalLM
    except (ImportError, ModuleNotFoundError):
        return None            # minigpt4 not on sys.path yet -- nothing to check
    try:
        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4,
                          max_position_embeddings=64)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            # written by transformers' class == what a real Vicuna dir looks like
            HFLlamaForCausalLM(cfg).save_pretrained(tmp)
            m = LlamaForCausalLM.from_pretrained(tmp, torch_dtype=torch.float16)
            m = m.float().eval()

            bad_buf = [n for n, b in m.named_buffers()
                       if b.is_floating_point() and not torch.isfinite(b).all()]
            ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
            mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])   # left padding, as we use
            with torch.no_grad():
                out = m(input_ids=ids, attention_mask=mask)
            ok = torch.isfinite(out.logits).all()

        if bad_buf or not ok:
            detail = (f"non-finite buffers after from_pretrained: {bad_buf[:4]}"
                      if bad_buf else "forward produced non-finite logits")
            return (
                f"transformers {transformers.__version__} breaks CoLLM's vendored "
                f"modeling_llama -- {detail}. Every *parameter* is finite, so this "
                "shows up only as `loss=nan` during training. Fix:\n"
                '            pip install "transformers==4.36.2" "peft==0.9.0"\n'
                "        then RESTART the runtime (Colab pre-imports transformers 5.x, "
                "so without a restart the downgrade has no effect)."
            )
    except Exception as e:                                      # noqa: BLE001
        import transformers
        return (f"CoLLM's vendored modeling_llama cannot run on transformers "
                f"{transformers.__version__}: {type(e).__name__}: {e}")
    return None


def check_environment(strict=False, numeric=True):
    """Report the versions that actually matter, and flag known-bad combinations.

    Cheap to call and worth calling: the failure modes it catches (a peft without
    ``prepare_model_for_int8_training``, a transformers whose ``utils`` no longer
    exports the docstring decorators CoLLM's vendored ``modeling_llama`` imports,
    and a transformers that silently NaNs that model) otherwise surface as
    confusing ImportErrors or a bare ``loss=nan`` deep inside CoLLM.
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

    # transformers 5.x is the one that bites hardest: everything imports, every
    # weight loads finite, and then the vendored LLaMA's forward returns NaN.
    if tf:
        try:
            major = int(str(tf).split(".")[0])
        except ValueError:
            major = None
        if major is not None and major >= 5:
            problems.append(
                f"transformers {tf} is a 5.x release. CoLLM's vendored "
                "modeling_llama produces NaN hidden states on 5.x -- you will see "
                "`loss=nan` with all parameters finite. Install transformers<5:\n"
                '            pip install "transformers==4.36.2" "peft==0.9.0"\n'
                "        and RESTART the runtime afterwards (Colab pre-imports "
                "transformers 5.x, so without a restart the downgrade has no effect)."
            )

    # CoLLM's vendored modeling_llama needs these from transformers.utils.
    # NOTE: 5.x still exports all three, so this check alone is not sufficient --
    # that is what check_vendored_llama() is for.
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

    # the decisive test: does the vendored model actually compute finite numbers?
    if numeric:
        bad = check_vendored_llama()
        if bad:
            problems.append(bad)

    for line in notes:
        print(f"[compat] {line}")
    for p in problems:
        print(f"[compat] PROBLEM: {p}")
        logging.warning("compat: %s", p)
    if problems and strict:
        raise RuntimeError("incompatible environment; see the [compat] lines above")
    return problems
