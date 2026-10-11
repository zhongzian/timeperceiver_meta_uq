"""Build a five-minute Chinese talk with evidence tables and a code-grounded diagram."""
import csv
import hashlib
import json
import os
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/group_meeting_20261010'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/timeperceiver_meeting_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
plt.rcParams['axes.unicode_minus'] = False
SOURCES = {}


def read(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p.read_text()


def wrap(text, width=72):
    return '\n'.join('\n'.join(textwrap.wrap(s, width=width, break_long_words=True)) for s in text.split('\n'))


def txt(ax, x, y, text, size=17, color='#20282d', **kw):
    return ax.text(x, y, text, fontproperties=FONT, fontsize=size, color=color,
                   va='top', transform=ax.transAxes, **kw)


def page(title, number, subtitle=''):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    txt(ax, .05, .955, title, 27, '#145b63')
    txt(ax, .05, .88, subtitle, 13, '#54636b')
    txt(ax, .05, .035, 'TimePerceiver Meta-UQ  |  组会汇报  |  2026-10-10  |  结果来自已保存实验；本次未启动训练', 10, '#667780')
    txt(ax, .94, .035, str(number), 10)
    return fig, ax


def notes(ax, text):
    ax.axhline(.25, xmin=.05, xmax=.95, color='#cbd8da', linewidth=1)
    txt(ax, .05, .225, '口头讲稿', 14, '#145b63')
    txt(ax, .05, .18, wrap(text, 68), 14, linespacing=1.4)


def table(ax, columns, rows, box, size=14):
    t = ax.table(cellText=rows, colLabels=columns, bbox=box, cellLoc='center')
    t.auto_set_font_size(False)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#d7e0e2'); cell.set_linewidth(.5)
        cell.get_text().set_fontproperties(FONT); cell.get_text().set_fontsize(size)
        cell.set_facecolor('#e2eff0' if r == 0 else ('#f4f7f7' if r % 2 else 'white'))
    return t


def box(ax, x, y, w, h, label, color='#e6edef', size=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.006',
                              facecolor=color, edgecolor='#52737b', transform=ax.transAxes))
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontproperties=FONT, fontsize=size, transform=ax.transAxes)


def arrow(ax, a, b, color='#52737b'):
    ax.annotate('', xy=b, xytext=a, xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(read('reports/chinese_evidence_step1/mean_std.csv').splitlines()))
    pairs = list(csv.DictReader(read('reports/chinese_evidence_step1/paired_mean_std.csv').splitlines()))
    for path in ['results/gate4a_formal_decision.json', 'results/gate4b_expanded_formal_decision.json',
                 'results/gate4b_multiseed_formal_decision.json', 'results/gate5a_formal_decision.json',
                 'results/gate5b_formal_decision.json', 'results/gate4c_formal_decision.json',
                 'reports/gate4c_interval_metrics.csv', 'reports/gate3a_protocol.md',
                 'models/probabilistic_timeperceiver.py', 'models/probabilistic_head.py',
                 'models/missingness_aware_timeperceiver.py']:
        read(path)
    talks = [
        '老师、同学们好。我这次汇报的是：历史观测缺失以后，怎样让预测结果不仅有一个数，还能反映不确定性。我们最开始尝试了缺失感知适配器和概率预测两条路线。现在实验的主线已经明确：真正持续出现正向结果的是重尾概率头，而不是所有模块叠加。接下来我用一张结构图、两组结果和两个负结果，把目前结论讲清楚。',
        '看这张图，灰色部分都是原来的 TimePerceiver。输入先应用固定缺失掩码，再经过分块和编码器，最后由未来查询读出表示。我们的主要改动就在 query cross-attention 后面：把原来的点预测投影换成位置和尺度两个投影，用它们构造预测分布。概率头训练时，主干冻结。固定 Student-t 的自由度就是8；可学习版本每个通道多一个参数。适配器是在同一个表示位置做另一组实验，conformal 则在区间输出后校准，它们不是当前主模型里已经验证有效的串联模块。',
        '实验过程可以概括成三步：先跑通官方模型，再固定数据划分和缺失掩码，最后保持主干不变比较不同分布头。现在已经做了三个数据集、第二个主干，以及 TimePerceiver 的四个概率头种子。表里最重要的一点是：这四个种子用的是可学习自由度，不是固定自由度，也不是四次端到端训练。三个数据集的平均 NLL 和 CRPS 都改善了；但 Electricity 的点误差略升，所以我们同时保留概率收益和点预测代价。',
        '接下来把自由度固定为8，看收益能不能保留下来。这里比较的都是同一个种子。三个数据集、两个主干一共六组配对，NLL 和 CRPS 都下降，说明这个方向并不只出现在 TimePerceiver 上。不过，PatchTST 在 Electricity 上的 MSE 上升了百分之一点一九四，超过原来百分之一的容许线。因此准确说法是概率评分方向复现了，而不是第二主干所有指标都通过。',
        '两个负结果也需要交代。适配器本身改善了点预测，但加上缺失感知条件以后，没有超过静态适配器；这更像是容量带来的收益。另一条路线是 conformal 校准，代码和泄漏检查通过了，但当前校准把区间扩得更宽，覆盖率反而偏离目标。我们没有把它们硬塞进最终方法。另外，早期没有真正施加掩码的结果已经作废；后来的表格标签错误也已识别，后续写作只使用修正后的证据。',
        '所以，能不能发论文？我的判断是已经有实证论文的基础，但还不能把工程通过等同于达到录用标准。我们的优势是固定掩码、公平配对、两个主干，以及愿意保留负结果。最需要补的不是马上扩大到所有数据集，而是统计可信度和机制解释：补配对置信区间、检查 CRPS 的采样误差，再做方差匹配与自由度敏感性，并补固定自由度的多种子。论文会定位成缺失历史下重尾概率预测的配对实证研究，不宣称发明了 Student-t，也不保证 CCF-C 录用。'
    ]
    pdf = OUT / '组会汇报_重尾概率预测_五分钟.pdf'
    with PdfPages(pdf) as out:
        fig, ax = page('01  我们现在做的是什么？', 1, '建议用时 35 秒  |  研究定位：缺失历史观测下的重尾概率预测配对实证研究')
        txt(ax, .07, .75, '问题：历史缺了，未来预测还能有多可信？', 26)
        txt(ax, .07, .63, '做法：同一主干、同一固定 mask，只改变输出分布。', 22)
        txt(ax, .07, .50, '目前主结论：重尾头改善概率评分；点预测存在数据集相关权衡。', 21)
        txt(ax, .07, .38, '已做：ETTh1 / Weather / Electricity；TimePerceiver + PatchTST。', 18)
        notes(ax, talks[0]); out.savefig(fig); plt.close(fig)

        fig, ax = page('02  模块具体加在哪里？', 2, '建议用时 65 秒  |  按项目实际 forward 重绘；参考 TimePerceiver 原论文 Figure 2')
        for x, label in [(.04, '历史输入 X\n固定 mask M\n训练集标准化'), (.225, '窗口归一化\nPatch embedding\n时间 + 通道编码'), (.41, 'Latent cross-attn\nLatent self-attn\nWrite cross-attn'), (.595, '未来 Query\nQuery cross-attn\n得到表示 h')]:
            box(ax, x, .67, .16, .15, label, size=13)
        for x in [.20, .385, .57]: arrow(ax, (x,.745), (x+.025,.745))
        box(ax, .80, .68, .16, .14, '原点预测投影\noutput_projection\n仅作基线', '#e6edef', 13)
        arrow(ax,(.755,.745),(.80,.745))
        box(ax, .55, .43, .21, .15, '新增概率头（当前主线）\nloc_projection → μ\nscale_projection → s > 0', '#d6eee5', 13)
        arrow(ax,(.675,.67),(.655,.58),'#278565')
        box(ax, .80, .43, .16, .15, 'Gaussian / Laplace\nStudent-t：ν = 8\n或通道级 learned ν', '#d6eee5', 12)
        arrow(ax,(.76,.505),(.80,.505),'#278565')
        box(ax, .04, .43, .40, .15, '独立 adapter 实验（不是概率头前置步骤）\nh → static adapter 或 mask encoder + FiLM\n→ 原点预测投影；条件化未超过 static', '#fff0d9', 12)
        arrow(ax,(.61,.67),(.40,.58),'#ae823a')
        txt(ax, .05, .365, '冻结：灰色 backbone；训练：绿色概率头。μ 恢复窗口尺度，s 在评价标准化空间学习。', 13)
        txt(ax, .05, .315, '独立后处理：预测区间 → conformal（校准集拟合）；当前为负结果，不纳入主线。', 13)
        notes(ax, talks[1]); out.savefig(fig); fig.savefig(OUT/'模型架构图.png', dpi=180); plt.close(fig)

        fig, ax = page('03  多种子：可学习 Student-t 的证据', 3, '建议用时 55 秒  |  TimePerceiver；random 0.3；L=384，H=96；四个 head seeds，共享固定主干')
        data=[]
        for ds in ['etth1','weather','electricity']:
            r=next(r for r in pairs if r['cohort']=='timeperceiver_multiseed' and r['dataset']==ds)
            data.append([ds, *[f"{float(r['delta_'+m+'_mean']):+.6f} ± {float(r['delta_'+m+'_std']):.6f}" for m in ['nll','crps']], f"{float(r['delta_mse_mean']):+.6f}"])
        table(ax,['数据集','ΔNLL：均值 ± 样本std','ΔCRPS：均值 ± 样本std','ΔMSE：均值'],data,[.05,.52,.90,.28],14)
        txt(ax,.06,.46,'差值 = learned Student-t − Gaussian；负值表示改善。不是显著性检验。',16)
        txt(ax,.06,.38,'12/12 配对 NLL 改善；点非劣不等于点误差全部下降。',17)
        txt(ax,.06,.31,'ETTh1 六种缺失条件扩展：NLL 6/6；CRPS非劣 5/6；点非劣 5/6（单seed）。',14)
        notes(ax,talks[2]);out.savefig(fig);plt.close(fig)

        fig,ax=page('04  固定 df=8：第二主干也有概率收益',4,'建议用时 55 秒  |  random 0.3；H=96；seed2025；全部为固定 Student-t 减 Gaussian')
        data=[]
        for ds in ['etth1','weather','electricity']:
            for cohort,name in [('timeperceiver_distribution','TimePerceiver'),('patchtst_distribution','PatchTST')]:
                r=next(r for r in pairs if r['cohort']==cohort and r['dataset']==ds and r['head']=='fixed_student_t_df8')
                data.append([ds,name,*[f"{float(r['delta_'+m+'_mean']):+.6f}" for m in ['nll','crps','mse']]])
        table(ax,['数据集','主干','ΔNLL','ΔCRPS','ΔMSE'],data,[.05,.40,.90,.40],15)
        txt(ax,.06,.33,'Electricity / PatchTST：MSE +1.194%，超过预设1%容许线；全指标 gate 未通过。',16,'#9b4b32')
        notes(ax,talks[3]);out.savefig(fig);plt.close(fig)

        fig,ax=page('05  哪些尝试没有形成正向结论？',5,'建议用时 45 秒  |  保留负结果，避免把“实现成功”写成“创新有效”')
        table(ax,['分支','实测结果','判断'],[
            ['Static adapter','ETTh1 MSE 0.361377；较official改善15.05%','容量收益'],
            ['Missingness-aware adapter','MSE 0.363156；比static恶化约0.49%','条件化无额外收益'],
            ['Conformal / learned-t','90%覆盖：89.76% → 92.76%（marginal）','更宽但偏离目标'],
            ['分布消融','learned-df未稳定优于fixed-df8','额外自由度不是必要条件']],[.05,.46,.90,.34],14)
        txt(ax,.06,.40,'Conformal 独立严格划分；不能与上一页原生分布表跨协议直接相减。',14)
        txt(ax,.06,.33,'历史无mask结果作废；四seed learned-df曾被误标fixed-df，现已更正证据归属。',14,'#9b4b32')
        notes(ax,talks[4]);out.savefig(fig);plt.close(fig)

        fig,ax=page('06  能发论文吗？下一步补什么？',6,'建议用时 45 秒  |  判断：具备实证研究初稿基础；尚不能保证 CCF-C 录用')
        txt(ax,.06,.76,'优势：可复核的配对协议 + 两个主干 + 多seed证据 + 完整负结果。',20)
        txt(ax,.06,.65,'关键缺口',20,'#145b63')
        txt(ax,.08,.58,'1. 统计：考虑时序相关的配对置信区间；CRPS 64次采样误差。\n2. 机制：方差匹配、df敏感性；证明不只是尺度变化。\n3. 覆盖：固定df多seed、更多预测长度与外部概率参照。',19,linespacing=1.8)
        txt(ax,.06,.34,'优先补可信度，再扩矩阵；不把“三个数据集”写成“六个已完成”。',17)
        notes(ax,talks[5]);out.savefig(fig);plt.close(fig)

        for cohort,title in [('timeperceiver_distribution','附录 A  分布消融完整核心指标'),('timeperceiver_multiseed','附录 B  四概率头种子结果')]:
            fig,ax=page(title,7 if 'distribution' in cohort else 8,'备查，不计入五分钟  |  NLL / CRPS / MSE / MAE 越低越好；所有值来自逐运行审计汇总')
            data=[]
            for r in rows:
                if r['cohort']!=cohort:continue
                head={'gaussian':'Gaussian','laplace':'Laplace','fixed_student_t_df8':'fixed t8','learned_student_t':'learned t'}[r['head']]
                vals=[]
                for m in ['nll','crps','mse','mae']:
                    v=f"{float(r[m+'_mean']):.6f}"
                    if r[m+'_std']!='NA':v+='\n±'+f"{float(r[m+'_std']):.6f}"
                    vals.append(v)
                data.append([r['dataset'],head,*vals])
            table(ax,['数据集','分布头','NLL','CRPS','MSE','MAE'],data,[.04,.20,.92,.61],12)
            txt(ax,.05,.14,'主表不混用：fixed t8为单seed；learned t四head seeds不等于四backbone seeds。',14)
            out.savefig(fig);plt.close(fig)
        fig,ax=page('附录 C  实验全流程与证据来源',9,'备查，不计入五分钟；“全流程”指所有已做方向，不代表所有配置均完成多seed')
        table(ax,['阶段','完成内容','科学结论'],[
            ['Gate 0–3b','环境、官方基线、固定mask与缺失评估','后续比较的基础'],
            ['Gate 4a','静态/缺失条件adapter及机制审计','static有效；条件化未增益'],
            ['Gate 4b','修正mask；条件扩展；跨数据集；四head seeds','learned Student-t概率信号正向'],
            ['Gate 4c','严格划分的原生/边际/条件conformal','实现通过；附加校准信号负向'],
            ['Gate 5a','Gaussian/Laplace/fixed-t8/learned-t','固定重尾已有收益；learned非稳定更优'],
            ['Gate 5b','PatchTST：3个数据集；fixed-t8对Gaussian','概率改善；Electricity点误差权衡'],
            ['Gate 6–7及大修','稿件工程与证据标签审计','不等于同行评审或录用']],[.05,.35,.90,.45],13)
        txt(ax,.05,.28,'主数值：reports/chinese_evidence_step1/{mean_std,paired_mean_std}.csv',13)
        txt(ax,.05,.23,'其他分支：results/gate4a_formal_decision.json；reports/gate4c_interval_metrics.csv',12)
        txt(ax,.05,.18,'参考图：Lee & Lee, TimePerceiver, NeurIPS 2025, Figure 2；本报告按实际代码重绘。',12)
        txt(ax,.05,.13,'未证明：MNAR、联合覆盖、所有数据集泛化、统计显著性。当前主线不是meta-learning。',13)
        out.savefig(fig);plt.close(fig)
    (OUT/'五分钟口语稿.md').write_text('# 五分钟组会口语稿\n\n'+'\n\n'.join(f'## 第{i+1}页\n\n{s}' for i,s in enumerate(talks))+'\n')
    (OUT/'source_manifest.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    (OUT/'README.md').write_text('# 组会汇报\n\n前6页按约5分钟讲述，附录3页备查。口语稿约'+str(sum(len(s) for s in talks))+'字，实际时长请以朗读为准。\n\n架构依据项目实际代码与[TimePerceiver原论文Figure 2](https://proceedings.neurips.cc/paper_files/paper/2025/file/c6c682ba9bd8839104f2a82901da4109-Paper-Conference.pdf)，重绘并区分独立实验分支。\n\n不宣称全稿审计或论文录用；本次未训练、未更改正式指标。\n')
    print(pdf)
    print('Spoken characters:',sum(len(s) for s in talks))


if __name__=='__main__':main()
