#!/usr/bin/env python
import argparse
import csv
import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path("/mnt/data/zza/code/timeperceiver_meta_uq")
UPSTREAM = PROJECT_ROOT / "upstream" / "TimePerceiver"
sys.path.insert(0, str(UPSTREAM))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimePerceiver import Model  # noqa: E402
from utils.metrics import metric  # noqa: E402


@dataclass
class DatasetSpec:
    key: str
    root_path: str
    data_path: str
    data: str
    features: str
    target: str
    freq: str
    enc_in: int
    official_script: str
    url: str


GOOGLE_DRIVE_FILE_ID = "1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot"
DATA_ARCHIVE = PROJECT_ROOT / "data" / "_downloads" / "iTransformer_datasets.zip"

DATASETS = [
    DatasetSpec(
        key="etth1",
        root_path="data/ETT-small",
        data_path="ETTh1.csv",
        data="ETTh1",
        features="M",
        target="OT",
        freq="h",
        enc_in=7,
        official_script="upstream/TimePerceiver/scripts/ETT/TimePerceiver_ETTh1.sh",
        url="gdrive://iTransformer_datasets.zip/dataset/ETT-small/ETTh1.csv",
    ),
    DatasetSpec(
        key="weather",
        root_path="data/weather",
        data_path="weather.csv",
        data="custom",
        features="M",
        target="OT",
        freq="h",
        enc_in=21,
        official_script="upstream/TimePerceiver/scripts/Weather/TimePerceiver.sh",
        url="gdrive://iTransformer_datasets.zip/dataset/weather/weather.csv",
    ),
    DatasetSpec(
        key="electricity",
        root_path="data/electricity",
        data_path="electricity.csv",
        data="custom",
        features="M",
        target="OT",
        freq="h",
        enc_in=321,
        official_script="upstream/TimePerceiver/scripts/ECL/TimePerceiver.sh",
        url="gdrive://iTransformer_datasets.zip/dataset/electricity/electricity.csv",
    ),
]


def run_cmd(cmd, cwd=PROJECT_ROOT):
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml_like(path: Path, obj) -> None:
    def render(value, indent=0):
        pad = " " * indent
        if isinstance(value, dict):
            lines = []
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.extend(render(v, indent + 2))
                else:
                    lines.append(f"{pad}{k}: {json.dumps(v, ensure_ascii=False)}")
            return lines
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(render(item, indent + 2))
                else:
                    lines.append(f"{pad}- {json.dumps(item, ensure_ascii=False)}")
            return lines
        return [f"{pad}{json.dumps(value, ensure_ascii=False)}"]

    write_text(path, "\n".join(render(obj)) + "\n")


def download_google_drive(file_id: str, dst: Path, retries: int = 3, timeout: int = 300) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and zipfile.is_zipfile(dst):
        return {"status": "reused", "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
    last_error = None
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            cookiejar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookiejar))
            base = f"https://docs.google.com/uc?export=download&id={file_id}"
            with opener.open(base, timeout=timeout) as response:
                content = response.read()
                headers = dict(response.headers)
                final_url = response.geturl()
            confirm = None
            for cookie in cookiejar:
                if cookie.name.startswith("download_warning"):
                    confirm = cookie.value
                    break
            if confirm is None:
                text = content[:20000].decode("utf-8", errors="ignore")
                marker = "confirm="
                if marker in text:
                    confirm = text.split(marker, 1)[1].split("&", 1)[0].split('"', 1)[0].split("'", 1)[0]
            if confirm:
                text = content[:30000].decode("utf-8", errors="ignore")
                action_match = re.search(r'<form[^>]+action="([^"]+)"', text)
                action = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
                params = {"id": file_id, "export": "download", "confirm": confirm}
                for name in ("uuid",):
                    m = re.search(rf'name="{name}" value="([^"]+)"', text)
                    if m:
                        params[name] = m.group(1)
                query = urllib.parse.urlencode(params)
                url = f"{action}?{query}"
                with opener.open(url, timeout=timeout) as response, tmp.open("wb") as f:
                    shutil.copyfileobj(response, f)
            else:
                ctype = headers.get("Content-Type", "")
                if "text/html" in ctype.lower() and "drive.google.com" in final_url:
                    raise RuntimeError("Google Drive confirmation token not found")
                tmp.write_bytes(content)
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise RuntimeError("Google Drive download produced an empty archive")
            if not zipfile.is_zipfile(tmp):
                sample = tmp.read_bytes()[:200].decode("utf-8", errors="ignore")
                raise RuntimeError(f"Google Drive response is not a zip file: {sample!r}")
            tmp.replace(dst)
            return {"status": "downloaded", "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(min(10 * attempt, 30))
    return {"status": "failed", "path": str(dst), "source": "google_drive", "error": last_error}


def ensure_from_archive(member: str, dst: Path, retries: int = 3) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return {"status": "reused", "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst), "source_member": member}
    archive = download_google_drive(GOOGLE_DRIVE_FILE_ID, DATA_ARCHIVE, retries=retries)
    if archive["status"] == "failed":
        return {"status": "failed", "path": str(dst), "url": f"gdrive://{GOOGLE_DRIVE_FILE_ID}/{member}", "error": archive.get("error")}
    last_error = None
    try:
        with zipfile.ZipFile(DATA_ARCHIVE) as zf:
            names = zf.namelist()
            candidates = [member, member.removeprefix("dataset/")]
            match = next((n for n in names if n in candidates or n.endswith("/" + member) or n.endswith("/" + member.removeprefix("dataset/"))), None)
            if match is None:
                raise FileNotFoundError(f"{member} not found in archive")
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            with zf.open(match) as src, tmp.open("wb") as out:
                shutil.copyfileobj(src, out)
            tmp.replace(dst)
            return {"status": "extracted", "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst), "archive": str(DATA_ARCHIVE), "source_member": match}
    except Exception as exc:  # noqa: BLE001
        last_error = repr(exc)
    return {"status": "failed", "path": str(dst), "url": f"gdrive://{GOOGLE_DRIVE_FILE_ID}/{member}", "error": last_error}


def download_file(url: str, dst: Path, retries: int = 3, timeout: int = 120) -> dict:
    if url.startswith("gdrive://iTransformer_datasets.zip/"):
        member = url.split("gdrive://iTransformer_datasets.zip/", 1)[1]
        return ensure_from_archive(member, dst, retries=retries)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return {"status": "reused", "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "timeperceiver-meta-uq-gate1"})
            with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
            if tmp.stat().st_size == 0:
                raise RuntimeError("downloaded zero bytes")
            tmp.replace(dst)
            return {"status": "downloaded", "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
        except Exception as exc:  # noqa: BLE001 - exact download failure is reported.
            last_error = repr(exc)
            time.sleep(min(5 * attempt, 15))
    return {"status": "failed", "path": str(dst), "url": url, "error": last_error}


def validate_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return {
        "path": str(path),
        "rows": rows,
        "columns": len(header),
        "header_first": header[:5],
        "has_date": "date" in header,
        "has_ot": "OT" in header,
    }


def make_args(spec: DatasetSpec, run_dir: Path, gpu: int, use_gpu: bool) -> SimpleNamespace:
    return SimpleNamespace(
        is_training=1,
        model_id=f"{spec.key}__smoke",
        model="TimePerceiver",
        data=spec.data,
        root_path=str(PROJECT_ROOT / spec.root_path) + "/",
        data_path=spec.data_path,
        features=spec.features,
        target=spec.target,
        checkpoints=str(run_dir / "checkpoints"),
        freq=spec.freq,
        augmentation_ratio=0,
        seq_len=24,
        label_len=0,
        pred_len=24,
        inverse=False,
        seasonal_patterns="Monthly",
        enc_in=spec.enc_in,
        d_model=16,
        n_heads=2,
        d_ff=32,
        dropout=0.1,
        embed="timeF",
        num_workers=0,
        itr=1,
        train_epochs=1,
        warmup_epochs=0,
        batch_size=2,
        patience=1,
        learning_rate=0.0005,
        weight_decay=0.0,
        des="gate1_smoke",
        loss="MSE",
        lradj="type3",
        use_amp=False,
        use_gpu=use_gpu,
        gpu=gpu,
        gpu_type="cuda" if use_gpu else "cpu",
        use_multi_gpu=False,
        devices=str(gpu),
        num_latents=4,
        latent_dim=16,
        latent_d_ff=32,
        use_latent=1,
        generalized=1,
        standard=0,
        num_latent_blocks=1,
        query_share=1,
        patch_len=12,
        separate_ratio=1.0,
        device=torch.device(f"cuda:{gpu}" if use_gpu else "cpu"),
    )


def save_checkpoint(path: Path, model, optimizer, args, train_loss: float, global_step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "train_loss": train_loss,
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        tmp,
    )
    tmp.replace(path)


def run_smoke(spec: DatasetSpec, args_ns, data_info: dict) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"timeperceiver__{spec.key}__official__smoke__seed2025__{ts}"
    run_dir = PROJECT_ROOT / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)

    args = make_args(spec, run_dir, args_ns.gpu, args_ns.use_gpu and torch.cuda.is_available())
    command = f"bash scripts/gate1_smoke.sh --datasets {spec.key} --gpu {args_ns.gpu}"
    write_text(run_dir / "command.sh", "#!/usr/bin/env bash\n" + command + "\n")

    resolved = {
        "experiment_id": experiment_id,
        "smoke": True,
        "dataset": asdict(spec),
        "data_info": data_info,
            "args": {k: (str(v) if k == "device" else v) for k, v in vars(args).items()},
        "notes": [
            "Gate 1 smoke-only configuration.",
            "Uses official data loader, official TimePerceiver model, and official split/scaling rules.",
            "Tiny seq_len/pred_len/model/batch and one optimizer step are not formal reproduction settings.",
        ],
    }
    write_yaml_like(run_dir / "resolved_config.yaml", resolved)

    train_log = run_dir / "train.log"
    started = time.time()
    status = {
        "experiment_id": experiment_id,
        "dataset": spec.key,
        "status": "RUNNING",
        "smoke": True,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "status.json", status)

    try:
        with train_log.open("w", encoding="utf-8") as log:
            log.write(f"experiment_id={experiment_id}\n")
            log.write(f"dataset={spec.key}\n")
            log.write(f"device={args.device}\n")

            train_data, train_loader = data_provider(args, "train")
            val_data, val_loader = data_provider(args, "val")
            test_data, test_loader = data_provider(args, "test")
            log.write(f"loader_lengths train={len(train_data)} val={len(val_data)} test={len(test_data)}\n")

            model = Model(args).float().to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            criterion = torch.nn.MSELoss()

            model.train()
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
            tensor = torch.cat([batch_x, batch_y[:, -args.pred_len:, 0:]], dim=1)

            from utils.random_split import random_split_patches

            batch_x2, batch_y2, indices = random_split_patches(
                tensor, args.pred_len // args.patch_len, args.patch_len, args.separate_ratio
            )
            optimizer.zero_grad()
            outputs = model(batch_x2, batch_x_mark, dec_inp, batch_y_mark, indices)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            target = batch_y2[:, -args.pred_len:, f_dim:].to(args.device)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            log.write(f"train_one_step_loss={loss.item():.8f}\n")

            checkpoint = run_dir / "checkpoints" / "last.pt"
            save_checkpoint(checkpoint, model, optimizer, args, float(loss.item()), 1)
            loaded = torch.load(checkpoint, map_location=args.device)
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            log.write(f"checkpoint_saved={checkpoint}\ncheckpoint_loaded=true\n")

            model.eval()
            preds, trues = [], []
            eval_losses = []
            with torch.no_grad():
                for split_name, loader in [("val", val_loader), ("test", test_loader)]:
                    for i, (bx, by, bxm, bym) in enumerate(loader):
                        bx = bx.float().to(args.device)
                        by = by.float().to(args.device)
                        bxm = bxm.float().to(args.device)
                        bym = bym.float().to(args.device)
                        din = torch.zeros_like(by[:, -args.pred_len:, :]).float()
                        din = torch.cat([by[:, :args.label_len, :], din], dim=1).float().to(args.device)
                        indices = (
                            list(range(0, args.seq_len // args.patch_len)),
                            list(range(args.seq_len // args.patch_len, (args.seq_len + args.pred_len) // args.patch_len)),
                        )
                        out = model(bx, bxm, din, bym, indices)
                        out = out[:, -args.pred_len:, f_dim:]
                        true = by[:, -args.pred_len:, f_dim:]
                        eval_loss = criterion(out, true).item()
                        eval_losses.append({"split": split_name, "loss": eval_loss})
                        if split_name == "test":
                            preds.append(out.detach().cpu().numpy())
                            trues.append(true.detach().cpu().numpy())
                        log.write(f"{split_name}_batch{i}_loss={eval_loss:.8f}\n")
                        break

            pred = np.concatenate(preds, axis=0)
            true = np.concatenate(trues, axis=0)
            mae, mse, rmse, mape, mspe = metric(pred, true)
            metrics = {
                "experiment_id": experiment_id,
                "dataset": spec.key,
                "smoke": True,
                "train_one_step_loss": float(loss.item()),
                "test_mse": float(mse),
                "test_mae": float(mae),
                "test_rmse": float(rmse),
                "test_mape": float(mape),
                "test_mspe": float(mspe),
                "eval_losses": eval_losses,
                "num_parameters": int(sum(p.numel() for p in model.parameters())),
                "duration_seconds": round(time.time() - started, 3),
                "checkpoint": str(checkpoint),
            }
            write_json(run_dir / "metrics.json", metrics)
            status.update(
                {
                    "status": "SUCCESS",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": metrics["duration_seconds"],
                    "metrics_path": str(run_dir / "metrics.json"),
                    "checkpoint": str(checkpoint),
                }
            )
            write_json(run_dir / "status.json", status)
            return status
    except Exception as exc:  # noqa: BLE001 - status file records exact smoke failure.
        status.update(
            {
                "status": "FAILED",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": repr(exc),
            }
        )
        write_json(run_dir / "status.json", status)
        with train_log.open("a", encoding="utf-8") as log:
            log.write(f"ERROR={repr(exc)}\n")
        return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="etth1,weather,electricity")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--use-gpu", action="store_true", default=True)
    args = parser.parse_args()

    requested = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    specs = [s for s in DATASETS if s.key in requested]
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "logs" / "gate1").mkdir(parents=True, exist_ok=True)

    readme_summary = {
        "official_readme": str(UPSTREAM / "README.md"),
        "official_data_source": "https://github.com/thuml/iTransformer",
        "official_scripts": [s.official_script for s in specs],
    }

    data_results = {}
    smoke_results = []
    blockers = []
    for spec in specs:
        dst = PROJECT_ROOT / spec.root_path / spec.data_path
        data_result = download_file(spec.url, dst, retries=args.download_retries)
        if data_result["status"] != "failed":
            try:
                data_result["csv"] = validate_csv(dst)
            except Exception as exc:  # noqa: BLE001
                data_result["status"] = "failed"
                data_result["error"] = repr(exc)
        data_results[spec.key] = data_result
        if data_result["status"] == "failed":
            blockers.append(f"DATA_PREP_FAILED:{spec.key}:{data_result.get('error')}")
            continue
        smoke_results.append(run_smoke(spec, args, data_result))

    passed = {r["dataset"] for r in smoke_results if r.get("status") == "SUCCESS"}
    failed = [r for r in smoke_results if r.get("status") != "SUCCESS"]
    for r in failed:
        blockers.append(f"SMOKE_FAILED:{r.get('dataset')}:{r.get('error')}")

    gate_status = "PASS" if len(passed) == len(specs) and not blockers else "PARTIAL"
    status = {
        "gate": "Gate 1",
        "status": gate_status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "conda_env": "timeperceiver_meta_uq",
        "official_repo": str(UPSTREAM),
        "official_commit": run_cmd(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]),
        "datasets_requested": requested,
        "datasets_passed": sorted(passed),
        "data_results": data_results,
        "smoke_results": smoke_results,
        "blockers": blockers,
        "next_gate": "Gate 2",
        "notes": [
            "Smoke tests use tiny smoke-only configs and are not formal reproduction results.",
            "Official model files were not modified.",
        ],
    }
    write_json(PROJECT_ROOT / "reports" / "gate1_status.json", status)

    lines = [
        "# Gate 1 Smoke Report",
        "",
        f"- Status: **{gate_status}**",
        f"- Timestamp: `{status['timestamp']}`",
        f"- Official commit: `{status['official_commit']}`",
        f"- Official data source: `{readme_summary['official_data_source']}`",
        "- Scope: ETTh1, Weather, Electricity smoke only; no formal training, no Gate 2.",
        "",
        "## Official Inputs Read",
        "",
        f"- README: `{readme_summary['official_readme']}`",
    ]
    lines.extend(f"- Script: `{p}`" for p in readme_summary["official_scripts"])
    lines.extend(["", "## Dataset Preparation", ""])
    for key, item in data_results.items():
        lines.append(f"- {key}: `{item.get('status')}` path=`{item.get('path')}` bytes=`{item.get('bytes', 'NA')}`")
    lines.extend(["", "## Smoke Results", ""])
    for item in smoke_results:
        lines.append(f"- {item.get('dataset')}: `{item.get('status')}` experiment_id=`{item.get('experiment_id')}` run_dir=`{item.get('run_dir')}`")
        if item.get("error"):
            lines.append(f"  error: `{item.get('error')}`")
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- `{b}`" for b in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Gate 1 is PASS only if all three smoke tests completed data loading, model construction, forward, backward, optimizer step, checkpoint save/load, evaluation, and metrics.json writing.",
        ]
    )
    write_text(PROJECT_ROOT / "reports" / "gate1_smoke_report.md", "\n".join(lines) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if gate_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
