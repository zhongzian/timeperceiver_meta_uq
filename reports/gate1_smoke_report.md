# Gate 1 Smoke Report

- Status: **PASS**
- Timestamp: `2026-07-01T15:02:49`
- Official commit: `7e30cc07b51c709f408409fd60a34c81ae8990be`
- Official data source: `https://github.com/thuml/iTransformer`
- Scope: ETTh1, Weather, Electricity smoke only; no formal training, no Gate 2.

## Official Inputs Read

- README: `/mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver/README.md`
- Script: `upstream/TimePerceiver/scripts/ETT/TimePerceiver_ETTh1.sh`
- Script: `upstream/TimePerceiver/scripts/Weather/TimePerceiver.sh`
- Script: `upstream/TimePerceiver/scripts/ECL/TimePerceiver.sh`

## Dataset Preparation

- etth1: `reused` path=`/mnt/data/zza/code/timeperceiver_meta_uq/data/ETT-small/ETTh1.csv` bytes=`2589657`
- weather: `reused` path=`/mnt/data/zza/code/timeperceiver_meta_uq/data/weather/weather.csv` bytes=`7235425`
- electricity: `reused` path=`/mnt/data/zza/code/timeperceiver_meta_uq/data/electricity/electricity.csv` bytes=`95581762`

## Smoke Results

- etth1: `SUCCESS` experiment_id=`timeperceiver__etth1__official__smoke__seed2025__20260701_150238` run_dir=`/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official__smoke__seed2025__20260701_150238`
- weather: `SUCCESS` experiment_id=`timeperceiver__weather__official__smoke__seed2025__20260701_150239` run_dir=`/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__official__smoke__seed2025__20260701_150239`
- electricity: `SUCCESS` experiment_id=`timeperceiver__electricity__official__smoke__seed2025__20260701_150241` run_dir=`/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__official__smoke__seed2025__20260701_150241`

## Blockers

- None

## Conclusion

Gate 1 is PASS only if all three smoke tests completed data loading, model construction, forward, backward, optimizer step, checkpoint save/load, evaluation, and metrics.json writing.
