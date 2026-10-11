import json, torch
out = {
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append({
            "logical_index": i,
            "name": p.name,
            "total_memory": p.total_memory,
            "capability": f"{p.major}.{p.minor}",
        })
    out["devices"] = devices
    x = torch.tensor([1.0], device="cuda")
    y = x * 2
    torch.cuda.synchronize()
    out["tiny_cuda_result"] = float(y.item())
print(json.dumps(out, indent=2, ensure_ascii=False))
if not out["cuda_available"]:
    raise SystemExit("torch installed but CUDA is not available")
