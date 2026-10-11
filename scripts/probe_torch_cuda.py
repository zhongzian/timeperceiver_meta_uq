#!/usr/bin/env python
import json
import os
import platform
import sys
from datetime import datetime


def main() -> int:
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_import_ok": False,
        "torch_version": None,
        "torch_cuda_runtime": None,
        "cudnn_version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "tensor_test": {"attempted": False, "ok": False, "device": None, "error": None},
    }

    try:
        import torch

        result["torch_import_ok"] = True
        result["torch_version"] = torch.__version__
        result["torch_cuda_runtime"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["device_count"] = int(torch.cuda.device_count())

        for idx in range(result["device_count"]):
            props = torch.cuda.get_device_properties(idx)
            result["devices"].append(
                {
                    "logical_index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "total_memory_mib": int(props.total_memory // (1024 * 1024)),
                    "capability": [int(props.major), int(props.minor)],
                }
            )

        if result["cuda_available"] and result["device_count"] > 0:
            result["tensor_test"]["attempted"] = True
            try:
                device = torch.device("cuda:0")
                a = torch.tensor([1.0], device=device)
                b = a * 2
                torch.cuda.synchronize(device)
                result["tensor_test"]["ok"] = bool(float(b.item()) == 2.0)
                result["tensor_test"]["device"] = "cuda:0"
            except Exception as exc:  # noqa: BLE001 - probe records exact failure.
                result["tensor_test"]["error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001 - probe records exact failure.
        result["torch_error"] = repr(exc)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["torch_import_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
