"""Figure 3 - item-level agreement follows the AI modality, and within the vision
modality it follows the class of computer-vision task.

Panel a ranks all 30 items coloured by AI modality (speech vs vision). Panels b-c
contrast the two modalities and rule out a difficulty artefact. Panel d shows that six
items carry most of the disagreement. Panel e resolves the vision block into its two
computer-vision task types (static image/OCR vs dynamic pose).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np, matplotlib.pyplot as plt, json
from matplotlib.gridspec import GridSpec
from scipy import stats

prep.style()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(OUT, exist_ok=True)
it = pd.read_csv(os.path.join(DAT, 'item_stats.csv'))
mg = pd.read_pickle(os.path.join(DAT, 'mg.pkl'))
S = json.load(open(os.path.join(DAT, 'summary.json')))
PAL = prep.PAL
PH = prep.VISION
VB = [i for i in range(1, 31) if i not in PH]

fig = plt.figure(figsize=(11.6, 5.3))
gs = GridSpec(2, 4, figure=fig, height_ratios=[1.05, 1], hspace=0.9, wspace=0.4)

# (a) kappa per item, ranked, coloured by AI modality
ax = fig.add_subplot(gs[0, :])
d = it.sort_values('kappa').reset_index(drop=True)
ax.bar(range(len(d)), d.kappa, color=[PAL[c] for c in d.cls], width=.72)
ax.errorbar(range(len(d)), d.kappa, yerr=[d.kappa - d.lo, d.hi - d.kappa],
            fmt='none', ecolor='#33333399', elinewidth=.8, capsize=1.5)
ax.set_xticks(range(len(d)))
ax.set_xticklabels(d.label, rotation=55, ha='right', fontsize=6.2)
ax.set_ylabel("Cohen's \u03ba  (AI vs clinician)"); ax.set_ylim(0, 1)
for y, lb in [(0.4, 'fair'), (0.6, 'moderate'), (0.8, 'substantial')]:
    ax.axhline(y, color='#bbb', lw=.7, ls=':')
    ax.text(9.2, y + .012, lb, fontsize=5.8, color='#888', ha='left')
h = [plt.Rectangle((0, 0), 1, 1, color=PAL[k]) for k in ['Speech AI', 'Vision AI']]
ax.legend(h, ['Speech AI  (24 items, iFlytek ASR)', 'Vision AI  (6 items)'],
          loc='upper left')
ax.set_title('a  Item-level agreement separates by AI modality', loc='left')

# (b) kappa by modality
ax = fig.add_subplot(gs[1, 0])
order = ['Speech AI', 'Vision AI']
rng = np.random.default_rng(1)
for i, c in enumerate(order):
    v = it[it.cls == c].kappa.values
    ax.scatter(np.full(len(v), i) + rng.normal(0, .06, len(v)), v, s=26,
               color=PAL[c], alpha=.85, edgecolors='white', linewidths=.5, zorder=3)
    ax.hlines(np.median(v), i - .26, i + .26, color='#1a1a1a', lw=1.6, zorder=4)
ax.set_xticks(range(2)); ax.set_xticklabels(['Speech AI\n(n=24)', 'Vision AI\n(n=6)'])
ax.set_ylabel("Cohen's \u03ba")
ax.plot([0, 0, 1, 1], [.96, .99, .99, .96], color='#1a1a1a', lw=.9)
ax.text(0.5, 1.0, 'P = %.1e' % S['mwu_p'], ha='center', fontsize=6.5)
ax.set_ylim(0, 1.1); ax.set_xlim(-.5, 1.5)
ax.set_title('b  Agreement by modality', loc='left')

# (c) kappa vs item difficulty - rules out a difficulty artefact
ax = fig.add_subplot(gs[1, 1])
for c in order:
    s = it[it.cls == c]
    ax.scatter(s.doc_mean, s.kappa, s=30, color=PAL[c], edgecolors='white',
               linewidths=.5, zorder=3)
r_v = stats.pearsonr(it[it.cls == 'Speech AI'].doc_mean,
                     it[it.cls == 'Speech AI'].kappa)
ax.set_xlabel('Item pass rate (clinician)'); ax.set_ylabel("Cohen's \u03ba")
ax.set_title('c  Not explained by pass rate', loc='left')
ax.text(0.03, 0.08, 'within speech items:\nr = %.2f, P = %.2f' % (r_v[0], r_v[1]),
        transform=ax.transAxes, fontsize=6.5)
ax.set_ylim(0, 1)

# (d) concentration of total disagreement
ax = fig.add_subplot(gs[1, 2])
contrib = pd.Series({i: (mg['m%d' % i].fillna(0) - mg['d%d' % i]).abs().sum()
                     for i in range(1, 31)})
contrib = 100 * contrib / contrib.sum()
o = contrib.sort_values(ascending=False)
cum = np.cumsum(o.values)
ax.plot(range(1, 31), cum, color='#1a1a1a', lw=1.3, zorder=2)
ax.scatter(range(1, 31), cum, s=16, color=[PAL[prep.item_class(i)] for i in o.index], zorder=3)
ax.plot([1, 30], [100 / 30, 100], color='#999', lw=.9, ls='--')
share = contrib[PH].sum()
ax.set_xlabel('Items ranked by error contribution')
ax.set_ylabel('Cumulative % of disagreement')
ax.set_title('d  Error concentration', loc='left')
ax.text(0.52, 0.10, 'uniform expectation', transform=ax.transAxes, fontsize=6,
        color='#888', rotation=22)
ax.text(0.96, 0.42, '6 vision items carry\n%.0f%% of all\ndisagreement' % share,
        transform=ax.transAxes, ha='right', fontsize=6.5)

# (e) within-vision: static image vs dynamic action
ax = fig.add_subplot(gs[1, 3])
sub = it[it.cls == 'Vision AI'].copy()
sub['vsub'] = sub.item.map(prep.vision_subtype)
subpal = prep.PAL_SUB
groups = ['Static image (OCR / figure)', 'Dynamic action (pose)']
for i, g in enumerate(groups):
    v = sub[sub.vsub == g].sort_values('kappa')
    xs = np.full(len(v), i) + rng.normal(0, .05, len(v))
    ax.scatter(xs, v.kappa, s=42, color=subpal[g], edgecolors='white',
               linewidths=.6, zorder=3)
    for x, (_, row) in zip(xs, v.iterrows()):
        ax.annotate(row.label.split()[0], (x, row.kappa), fontsize=5.6,
                    xytext=(5, 0), textcoords='offset points', va='center')
    ax.hlines(v.kappa.mean(), i - .26, i + .26, color='#1a1a1a', lw=1.6, zorder=4)
ax.set_xticks(range(2))
ax.set_xticklabels(['Static image\n(write, copy)', 'Dynamic action\n(eyes, hands)'], fontsize=6.3)
ax.set_ylabel("Cohen's \u03ba"); ax.set_ylim(0, 0.62); ax.set_xlim(-.5, 1.5)
ax.set_title('e  Within vision AI, by task type', loc='left')

prep.save(fig, 'Fig3_item_atlas', OUT)
print('Fig3: 5 panels; vision share of disagreement %.1f%%' % share)
