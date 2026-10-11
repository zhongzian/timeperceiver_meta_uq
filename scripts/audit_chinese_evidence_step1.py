"""Read original runs; write a separate, reproducible paper evidence inventory."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

try:
    import yaml
except ImportError:
    sys.path.insert(0, '/mnt/data/zza/anaconda3/lib/python3.11/site-packages')
    import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/chinese_evidence_step1'
SOURCES = {
    'timeperceiver_multiseed': 'reports/gate4b_multiseed_results.csv',
    'timeperceiver_distribution': 'reports/gate5a_distribution_results.csv',
    'patchtst_distribution': 'reports/gate5b_patchtst_probability_results.csv',
}
METRICS = {'nll': 'test_mean_nll', 'crps': 'test_crps',
           'mse': 'test_mse', 'mae': 'test_mae'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def classify(config, metrics):
    variant = config.get('variant', metrics.get('variant', ''))
    if 'gaussian' in variant:
        return 'gaussian'
    if 'laplace' in variant:
        return 'laplace'
    names = metrics.get('trainable_parameter_names', [])
    if any('raw_df' in name for name in names):
        return 'learned_student_t'
    if config.get('learnable_df') is False and config.get('fixed_t_df') == 8:
        return 'fixed_student_t_df8'
    return 'UNRESOLVED'


def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes, issues, records = {}, [], []

    def track(path):
        path = path.resolve()
        if path not in hashes:
            hashes[path] = digest(path)
        return hashes[path]

    def issue(run, kind, detail):
        issues.append({'experiment_id': run, 'kind': kind, 'detail': detail})

    for cohort, source in SOURCES.items():
        track(ROOT / source)
        for row in csv.DictReader((ROOT / source).open()):
            if row.get('variant') == 'official':
                continue
            directory = ROOT / row['run_dir']
            run = directory.name
            loaded = {}
            for filename in ('metrics.json', 'status.json', 'resolved_config.yaml'):
                path = directory / filename
                track(path)
                loaded[filename] = (yaml.safe_load(path.read_text()) if filename.endswith('yaml')
                                    else json.loads(path.read_text()))
            m, c, s = (loaded[k] for k in ('metrics.json', 'resolved_config.yaml', 'status.json'))
            head = classify(c, m)
            if head == 'UNRESOLVED':
                issue(run, 'ERROR', 'Unresolved distribution identity')
            if c.get('experiment_id', run) != run:
                issue(run, 'CONFIG_ID_INHERITED', str(c['experiment_id']))
            if c.get('run_dir') and Path(c['run_dir']).name != run:
                issue(run, 'CONFIG_RUN_DIR_INHERITED', str(c['run_dir']))
            record = {'cohort': cohort, 'experiment_id': run, 'source_csv': source,
                      'run_dir': str(directory.relative_to(ROOT)), 'head': head,
                      'dataset': row['dataset'], 'head_seed': row['experiment_seed'],
                      'status': s.get('status'), 'seq_len': c.get('seq_len'),
                      'pred_len': c.get('pred_len'), 'batch_size': c.get('batch_size'),
                      'missing_type': c.get('missing_type', row.get('missing_type')),
                      'missing_rate': c.get('missing_rate', row.get('missing_rate')),
                      'metric_space': m.get('metric_space', 'NOT_EXPLICIT_IN_METRICS'),
                      'crps_samples': m.get('crps_samples', c.get('crps_samples')),
                      'min_df': m.get('min_df'), 'max_df': m.get('max_df'),
                      'trainable_parameter_count': m.get('trainable_parameter_count'),
                      'best_epoch': m.get('best_epoch'),
                      'metric_protocol_hash': m.get('metric_protocol_hash', c.get('metric_protocol_hash')),
                      'metrics_sha256': track(directory / 'metrics.json'),
                      'config_sha256': track(directory / 'resolved_config.yaml')}
            for metric, key in METRICS.items():
                value = float(m[key])
                record[metric] = value
                if not math.isfinite(value) or abs(value - float(row[key])) > 1e-10:
                    issue(run, 'ERROR', 'CSV/raw metric mismatch: ' + key)
            for flag in ('input_mask_applied', 'mask_application_verified'):
                if flag not in m:
                    issue(run, 'AUDIT_FIELD_ABSENT', flag + '; not inferred as PASS')
                elif m.get(flag) is not True:
                    issue(run, 'ERROR', flag + ' is not true')
            if s.get('status') not in ('SUCCESS', 'SUCCEEDED'):
                issue(run, 'ERROR', 'Run not successful: ' + str(s.get('status')))
            for flag in ('legacy_unmasked_invalid', 'batch_y_modified', 'test_used_for_model_selection'):
                if m.get(flag) is True:
                    issue(run, 'ERROR', flag + ' is true')
            for asset in ('source_checkpoint', 'test_mask', 'validation_mask'):
                value = c.get(asset) or c.get('source_backbone_checkpoint' if asset == 'source_checkpoint' else asset) or s.get(asset) or row.get(asset)
                expected = m.get(asset + '_sha256') or c.get(asset + '_sha256') or row.get(asset + '_sha256')
                record[asset] = value
                record[asset + '_sha256'] = expected
                if not value or not expected:
                    issue(run, 'ERROR', 'Missing asset provenance: ' + asset)
                elif track(ROOT / value) != expected:
                    issue(run, 'ERROR', 'Asset hash mismatch: ' + asset)
            records.append(record)

    groups = {}
    for record in records:
        key = (record['cohort'], record['dataset'], record['head'])
        groups.setdefault(key, []).append(record)
    aggregates = []
    for (cohort, dataset, head), rows in sorted(groups.items()):
        result = {'cohort': cohort, 'dataset': dataset, 'head': head, 'n_head_seeds': len(rows),
                  'head_seeds': '|'.join(sorted(r['head_seed'] for r in rows)),
                  'n_backbone_checkpoints': len(set(r['source_checkpoint_sha256'] for r in rows)),
                  'experiment_ids': '|'.join(r['experiment_id'] for r in rows)}
        for metric in METRICS:
            values = [r[metric] for r in rows]
            result[metric + '_mean'] = statistics.mean(values)
            result[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else 'NA'
        aggregates.append(result)
    pairs = []
    for r in records:
        if r['head'] == 'gaussian':
            continue
        base = next(g for g in records if (g['cohort'], g['dataset'], g['head_seed'], g['head']) ==
                    (r['cohort'], r['dataset'], r['head_seed'], 'gaussian'))
        pair = {'cohort': r['cohort'], 'dataset': r['dataset'], 'head_seed': r['head_seed'],
                'head': r['head'], 'experiment_id': r['experiment_id'], 'gaussian_id': base['experiment_id']}
        for field in ('source_checkpoint_sha256', 'test_mask_sha256', 'validation_mask_sha256',
                      'metric_protocol_hash', 'seq_len', 'pred_len', 'batch_size'):
            if not r[field] or r[field] != base[field]:
                issue(r['experiment_id'], 'ERROR', 'Pair mismatch: ' + field)
        for metric in METRICS:
            pair['delta_' + metric] = r[metric] - base[metric]
        pairs.append(pair)

    table = ROOT / 'paper/venues/journal_of_forecasting/manuscript/tables/table4_timeperceiver_multiseed.tex'
    track(table)
    table_checks = []
    for line in table.read_text().splitlines():
        cells = [x.strip() for x in line.split('&')]
        if cells[0].lower() not in ('etth1', 'weather', 'electricity'):
            continue
        dataset = cells[0].lower()
        for head, positions in [('gaussian', [1, 5, 8]), ('learned_student_t', [2, 6, 9])]:
            a = next(a for a in aggregates if a['cohort'] == 'timeperceiver_multiseed' and a['dataset'] == dataset and a['head'] == head)
            for metric, pos in zip(('nll', 'crps', 'mse'), positions):
                match = abs(float(cells[pos]) - a[metric + '_mean']) <= 0.00000051
                table_checks.append({'table': str(table.relative_to(ROOT)), 'dataset': dataset,
                                     'actual_head': head, 'metric': metric, 'numeric_match': match,
                                     'semantic_correction': 'Student-t column is learned-df, not fixed-df8'})
                if not match:
                    issue('table4', 'ERROR', dataset + '/' + metric + ' mismatch')

    for filename in ('table8_distribution_ablation.tex', 'table10_second_backbone.tex'):
        path = table.parent / filename
        track(path)
        for line in path.read_text().splitlines():
            cells = [x.strip() for x in line.split('&')]
            if cells[0].lower() not in ('etth1', 'weather', 'electricity'):
                continue
            dataset = cells[0].lower()
            if filename.startswith('table8'):
                head = {'Gaussian': 'gaussian', 'Laplace': 'laplace',
                        'Fixed Student-t (df=8)': 'fixed_student_t_df8',
                        'Learned Student-t': 'learned_student_t'}[cells[1]]
                a = next(a for a in aggregates if a['cohort'] == 'timeperceiver_distribution'
                         and a['dataset'] == dataset and a['head'] == head)
                values = [a[k + '_mean'] for k in ('nll', 'crps', 'mse')]
            else:
                cohort = ('timeperceiver_distribution' if cells[1] == 'TimePerceiver'
                          else 'patchtst_distribution')
                p = next(p for p in pairs if p['cohort'] == cohort and p['dataset'] == dataset
                         and p['head'] == 'fixed_student_t_df8')
                head = 'fixed_student_t_df8_minus_gaussian'
                values = [p['delta_' + k] for k in ('nll', 'crps', 'mse')]
            for pos, metric, value in zip((2, 3, 4), ('nll', 'crps', 'mse'), values):
                match = abs(float(cells[pos]) - value) <= 0.00000051
                table_checks.append({'table': str(path.relative_to(ROOT)), 'dataset': dataset,
                                     'actual_head': head, 'metric': metric, 'numeric_match': match,
                                     'semantic_correction': 'single head seed2025'})
                if not match:
                    issue(filename, 'ERROR', dataset + '/' + metric + ' mismatch')

    paired_summary = []
    for cohort, dataset, head in sorted(set((p['cohort'], p['dataset'], p['head']) for p in pairs)):
        rows = [p for p in pairs if (p['cohort'], p['dataset'], p['head']) == (cohort, dataset, head)]
        summary = {'cohort': cohort, 'dataset': dataset, 'head': head, 'n_pairs': len(rows)}
        for metric in METRICS:
            values = [p['delta_' + metric] for p in rows]
            summary['delta_' + metric + '_mean'] = statistics.mean(values)
            summary['delta_' + metric + '_std'] = statistics.stdev(values) if len(values) > 1 else 'NA'
        paired_summary.append(summary)

    write_csv('run_inventory.csv', records)
    write_csv('mean_std.csv', aggregates)
    write_csv('paired_deltas.csv', pairs)
    write_csv('table_numeric_checks.csv', table_checks)
    write_csv('paired_mean_std.csv', paired_summary)
    write_csv('provenance_findings.csv', issues)
    unchanged = all(digest(path) == expected for path, expected in hashes.items())
    write_csv('input_hashes.csv', [{'path': str(p), 'sha256': h} for p, h in sorted(hashes.items())])
    errors = [i for i in issues if i['kind'] == 'ERROR']
    status = {'stage': 'Chinese revision step 1', 'status': 'PASS_WITH_DOCUMENT_CORRECTIONS' if not errors and unchanged else 'FAILED',
              'scope': 'Gate4b multiseed, Gate5a distribution, Gate5b probability; Table4/8/10 NLL, CRPS, MSE and head identity audit',
              'run_appearances': len(records), 'unique_runs': len(set(r['experiment_id'] for r in records)),
              'asset_and_source_files_hashed': len(hashes), 'inputs_unchanged': unchanged,
              'errors': errors, 'provenance_warning_count': len(issues) - len(errors),
              'table_numeric_checks': len(table_checks),
              'training_started': False, 'frozen_manuscript_modified': False,
              'whole_manuscript_audit_complete': False}
    (ROOT / 'results/chinese_evidence_step1_status.json').write_text(json.dumps(status, indent=2) + '\n')
    lines = ['# 中文初稿证据核对：第一步', '',
             '## 范围与结论', '',
             '核对 Gate4b 四 head seed、Gate5a 分布消融、Gate5b PatchTST 概率结果。逐运行读取 metrics/config/status，复算统计并核验源 checkpoint 与固定 mask 的 SHA256。',
             '本报告不代表全稿全部表格已审计；未修改历史结果、正式 decision 或冻结稿件。', '',
             '## 必须修正的证据归属', '',
             '- 多 seed 表的 Student-t 是 learned-df，不是 fixed df=8。初始 df=8 不等于固定自由度。',
             '- 四个 seed 为概率头训练 seed；每个数据集共享一个 backbone checkpoint，不能写成四个 backbone seed。',
             '- 固定 df=8 的 TimePerceiver/PatchTST 当前核对证据为 seed2025；不得把 learned-df 的 12 对结果移用为 fixed-df 的多 seed 结论。',
             '- 表格差异须区分单 seed、四 seed 均值、learned/fixed 和 backbone；下表是同一指标原始文件重算值，标准差使用 ddof=1，单 seed 不估算标准差。',
             '- 部分 resolved_config 继承 baseline experiment_id/run_dir。保留原件，以实际目录和 status 交叉追溯，详见 provenance_findings.csv。', '',
             '- 两个 corrected_minimal 运行未记录 mask_application_verified 字段；input_mask_applied 为 true，资产哈希匹配，但不能把缺失字段写成已核验通过。',
             '- 表4、表8、表10的 NLL/CRPS/MSE 已与逐运行指标核对；表4数值匹配不代表 fixed-df 标签正确。', '',
             '## 中文初稿可用的分组结果', '',
             '| 证据组 | 数据集 | 分布头 | n | NLL mean ± std | CRPS mean ± std | MSE mean ± std |',
             '|---|---|---|---:|---|---|---|']
    for a in aggregates:
        def cell(metric):
            sd = a[metric + '_std']
            return f"{a[metric + '_mean']:.6f}" + (f' ± {sd:.6f}' if sd != 'NA' else ' (单 seed)')
        lines.append('| ' + ' | '.join([a['cohort'], a['dataset'], a['head'], str(a['n_head_seeds']), cell('nll'), cell('crps'), cell('mse')]) + ' |')
    lines += ['', '## 使用边界与下一步', '',
              'mean_std.csv 与 paired_deltas.csv 来自逐运行指标，不从论文表格抄数。配对差值定义为目标头减 Gaussian。',
              '不能将同一运行在不同阶段的复用当成新增独立样本。当前未计算显著性，不能把计数或均值优势表述为统计显著。',
              'CRPS 延续既有实现；本轮未提升采样数或重评估。跨 backbone 的指标协议 hash 不同，不能宣称训练器完全相同。',
              '下一步：用上述分组证据创建独立中文双栏初稿；补查其他表格与正文引用，不覆盖冻结投稿稿。', '',
              '```json', json.dumps(status, ensure_ascii=False, indent=2), '```']
    (OUT / 'audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(status, indent=2))
    return 1 if errors or not unchanged else 0


if __name__ == '__main__':
    sys.exit(main())
