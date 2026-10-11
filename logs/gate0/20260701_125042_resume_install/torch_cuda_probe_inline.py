import json, sys, torch
out = {
    "python": sys.version,
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": []
}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out["devices"].append({
            "logical_index": i,
            "name": p.name,
            "total_memory": p.total_memory,
            "capability": f"{p.major}.{p.minor}"
        })
    x = torch.tensor([1.0], device="cuda")
    y = x * 2
    torch.cuda.synchronize()
    out["tiny_cuda_result"] = float(y.item())
print(json.dumps(out, indent=2, ensure_ascii=False))
