"""Generate draft tables from audited CSVs and compile without network access."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'paper/chinese_empirical_draft'
AUDIT = ROOT / 'reports/chinese_evidence_step1'


def main():
    rows = list(csv.DictReader((AUDIT / 'mean_std.csv').open()))
    pairs = list(csv.DictReader((AUDIT / 'paired_deltas.csv').open()))
    if json.loads((ROOT / 'results/chinese_evidence_step1_status.json').read_text())['errors']:
        raise ValueError('Evidence audit contains errors')
    tables = DRAFT / 'tables'
    tables.mkdir(exist_ok=True)
    names = {'etth1': 'ETTh1', 'weather': 'Weather', 'electricity': 'Electricity'}
    heads = {'gaussian': 'Gaussian', 'laplace': 'Laplace',
             'fixed_student_t_df8': 'Student-t (fixed 8)',
             'learned_student_t': 'Student-t (learned)'}

    def table(filename, label, caption, data):
        text = ['\\begin{table*}[t]', '\\centering\\small',
                '\\caption{' + caption + '}\\label{tab:' + label + '}',
                '\\begin{tabular}{llrrr}', '\\toprule',
                '数据集 & 分布头/主干 & NLL & CRPS & MSE \\\\', '\\midrule']
        text.extend(' & '.join(row) + ' \\\\' for row in data)
        text.extend(['\\bottomrule', '\\end{tabular}', '\\end{table*}'])
        (tables / filename).write_text('\n'.join(text) + '\n')

    for cohort, filename, label, caption in [
        ('timeperceiver_multiseed', 'multiseed.tex', 'multiseed',
         'TimePerceiver 四概率头种子结果：均值 $\\pm$ 样本标准差。Student-t 为可学习自由度；每个数据集共享一个主干检查点。'),
        ('timeperceiver_distribution', 'distribution.tex', 'distribution',
         'TimePerceiver 分布消融，种子2025。所有结果为 random 0.3、$L=384,H=96$；单种子不报告标准差。')]:
        data = []
        for dataset in names:
            for head in heads:
                matching = [r for r in rows if (r['cohort'], r['dataset'], r['head']) == (cohort, dataset, head)]
                if not matching:
                    continue
                r = matching[0]
                values = []
                for metric in ('nll', 'crps', 'mse'):
                    value = f"{float(r[metric + '_mean']):.6f}"
                    if r[metric + '_std'] != 'NA':
                        value = '$' + value + r'\pm' + f"{float(r[metric + '_std']):.6f}" + '$'
                    values.append(value)
                data.append([names[dataset], heads[head]] + values)
        table(filename, label, caption, data)
    data = []
    for dataset in names:
        for cohort, backbone in [('timeperceiver_distribution', 'TimePerceiver'),
                                 ('patchtst_distribution', 'PatchTST')]:
            r = next(r for r in pairs if (r['cohort'], r['dataset'], r['head']) ==
                     (cohort, dataset, 'fixed_student_t_df8'))
            data.append([names[dataset], backbone] + [f"{float(r['delta_' + m]):+.6f}" for m in ('nll', 'crps', 'mse')])
    table('backbone.tex', 'backbone', '固定 Student-t 减 Gaussian 的配对差值（表中三项均为差值）。种子2025，random 0.3，$H=96$。', data)

    build = DRAFT / 'build'
    build.mkdir(exist_ok=True)
    engine = '/mnt/data/zza/anaconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic'
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/mnt/data/zza/anaconda3/pkgs/icu-78.3-h84d19a5_0/lib:' + env.get('LD_LIBRARY_PATH', '')
    command = [engine, '--only-cached', '--keep-logs', '--keep-intermediates', '--outdir', str(build), 'main.tex']
    with (build / 'compile.log').open('w') as log:
        result = subprocess.run(command, cwd=DRAFT, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [AUDIT / 'mean_std.csv', AUDIT / 'paired_deltas.csv', DRAFT / 'main.tex'] + list(tables.glob('*.tex'))}
    status = {'stage': 'Chinese empirical draft', 'compile_exit_code': result.returncode,
              'pdf': str((build / 'main.pdf').relative_to(ROOT)), 'source_hashes': manifest,
              'training_started': False, 'submission_ready': False,
              'command': command}
    (ROOT / 'results/chinese_draft_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
