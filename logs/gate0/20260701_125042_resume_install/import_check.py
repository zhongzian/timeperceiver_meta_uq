import json, importlib, sys
mods = [
    "torch", "numpy", "pandas", "scipy", "sklearn", "einops",
    "matplotlib", "sktime", "sympy", "pywt",
    "reformer_pytorch", "local_attention"
]
out = {"python": sys.version, "imports": {}}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out["imports"][m] = {"ok": True, "version": getattr(mod, "__version__", "NA")}
    except Exception as e:
        out["imports"][m] = {"ok": False, "error": repr(e)}
import torch
out["torch"] = {
    "version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
print(json.dumps(out, indent=2, ensure_ascii=False))
failed = [k for k, v in out["imports"].items() if not v["ok"]]
if failed:
    raise SystemExit("IMPORT_FAILED: " + ",".join(failed))
if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE")
