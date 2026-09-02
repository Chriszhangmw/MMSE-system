"""Figure 5 - in-silico modality-aware routing recovers most of the lost agreement.

Panels: agreement metrics, screening decision quality, subgroup error gaps
(the subgroup panel contrasts full automation with the in-silico routed total),
deployment drift.
The Bland-Altman overlay was removed: the limits-of-agreement width it displayed
is reported numerically in panel a, and the plot itself duplicated Fig2b.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np, matplotlib.pyplot as plt, json

prep.style()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(OUT, exist_ok=True)
mg = pd.read_pickle(os.path.join(DAT, 'mg.pkl'))
S = json.load(open(os.path.join(DAT, 'summary.json')))
PH = prep.VISION
VB = [i for i in range(1, 31) if i not in PH]
mg['hyb'] = (mg[['m%d' % i for i in VB]].fillna(0).sum(axis=1)
             + mg[['d%d' % i for i in PH]].sum(axis=1))
mg['abserr'] = (mg.sum_m - mg.sum_d).abs()
mg['hyb_abs'] = (mg.hyb - mg.sum_d).abs()

fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.1),
                         gridspec_kw=dict(width_ratios=[1.15, 1, 1.05, 1.15], wspace=0.62))
fig.subplots_adjust(bottom=0.24)

# (a) agreement metrics, full automation vs hybrid
ax = axes[0]
mets = [('ICC(2,1)', S['icc_auto'], S['icc_hyb'], '%.3f'),
        ('Systematic bias\n(points)', abs(S['bias_auto']), abs(S['bias_hyb']), '%.2f'),
        ('SD of error\n(points)', S['sd_auto'], S['sd_hyb'], '%.2f'),
        ('95% LoA width\n(points)', 2 * 1.96 * S['sd_auto'], 2 * 1.96 * S['sd_hyb'], '%.1f')]
ys = np.arange(len(mets))[::-1]
for (lb, a_, h_, fmt), y in zip(mets, ys):
    ref = max(a_, h_)
    xa, xh = a_ / ref, h_ / ref
    ax.plot([xa, xh], [y, y], color='#999', lw=1.4, zorder=1)
    ax.scatter([xa], [y], s=46, color='#c0392b', zorder=3,
               label='Full automation' if y == ys[0] else None)
    ax.scatter([xh], [y], s=46, color='#2c6fbb', zorder=3,
               label='In-silico routed total' if y == ys[0] else None)
    ax.text(xa, y + .24, fmt % a_, ha='center', fontsize=6.3, color='#c0392b')
    ax.text(xh, y - .34, fmt % h_, ha='center', fontsize=6.3, color='#2c6fbb')
ax.set_yticks(ys); ax.set_yticklabels([m[0] for m in mets], fontsize=6.5)
ax.set_xlabel('Value relative to the worse of the two')
ax.set_xlim(0, 1.3); ax.set_ylim(-.9, len(mets) - .25)
ax.legend(fontsize=6, loc='lower left')
ax.set_title('a  In-silico substitution', loc='left')

# (b) screening decision quality
ax = axes[1]
a, h = S['screen_auto'], S['screen_hyb']
keys = [('Positive\nagreement', 'sens'), ('Negative\nagreement', 'spec'), ('PPV', 'ppv'), ('Cohen \u03ba', 'kappa')]
x = np.arange(len(keys))
ax.bar(x - .19, [a[k] for _, k in keys], .36, color='#c0392b')
ax.bar(x + .19, [h[k] for _, k in keys], .36, color='#2c6fbb')
for i, (_, k) in enumerate(keys):
    ax.text(i - .19, a[k] + .015, '%.2f' % a[k], ha='center', fontsize=6.3)
    ax.text(i + .19, h[k] + .015, '%.2f' % h[k], ha='center', fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in keys], fontsize=6.8, rotation=15)
ax.set_ylim(0, 1.18); ax.set_ylabel('Value')
ax.set_title('b  Agreement with clinician', loc='left')
ax.text(0.5, -0.42, 'relative to clinician screening classification', transform=ax.transAxes, ha='center', fontsize=5.6, style='italic')
ax.annotate('', xy=(1.19, h['spec'] + .01), xytext=(1.19, a['spec'] - .01),
            arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1.1))
ax.text(1.5, 1.06, '+%.0f pts negative agreement'
        % (100 * (h['spec'] - a['spec'])), fontsize=5.8, ha='center')

# (c) subgroup error gaps, before and after routing (moved from the old Fig5d)
ax = axes[2]
comp = [('Age > 70 y', mg.age > 70), ('Edu \u2264 9 y', mg.eduy <= 9),
        ('Mandarin', mg.mandarin), ('Female', mg.sex == 'F')]
ys = np.arange(len(comp))[::-1]
for (lb, mask), y in zip(comp, ys):
    for tag, col, off, key in [('Full automation', '#c0392b', .16, 'abserr'),
                               ('In-silico routed total', '#2c6fbb', -.16, 'hyb_abs')]:
        aa = mg.loc[mask.fillna(False), key]; bb = mg.loc[~mask.fillna(True), key]
        dd = aa.mean() - bb.mean()
        se = np.sqrt(aa.var() / len(aa) + bb.var() / len(bb))
        ax.errorbar(dd, y + off, xerr=1.96 * se, fmt='o', ms=4, color=col,
                    capsize=2, lw=1.1)
ax.axvline(0, color='#1a1a1a', lw=.9)
ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in comp], fontsize=7)
ax.set_xlabel('\u0394 mean |error| vs complement (points)')
ax.set_title('c  Subgroup error gaps', loc='left')
ax.set_xlim(-1.5, 2.0); ax.set_ylim(-0.7, len(comp) - 0.3)

# (d) deployment drift and out-of-period validation
ax = axes[3]
mg['q'] = mg.date.dt.to_period('Q').dt.to_timestamp()
g = mg.groupby('q')
ok = g.size() >= 30
bias_a = g.apply(lambda s: (s.sum_m - s.sum_d).mean(), include_groups=False)[ok]
bias_h = g.apply(lambda s: (s.hyb - s.sum_d).mean(), include_groups=False)[ok]
ax.plot(bias_a.index, bias_a.values, 'o-', color='#c0392b', ms=3.2, lw=1.2,
        label='Full automation')
ax.plot(bias_h.index, bias_h.values, 'o-', color='#2c6fbb', ms=3.2, lw=1.2,
        label='In-silico routed total')
ax.axhline(0, color='#1a1a1a', lw=.8, ls='--')
ax.axvspan(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-12-31'), color='#eeeeee')
ax.set_ylabel('Mean AI \u2212 clinician (points)')
ax.set_title('d  Deployment drift', loc='left')
ax.legend(fontsize=6, loc='lower right')
ax.set_ylim(-6.5, 3.4)
ax.tick_params(axis='x', rotation=35, labelsize=6)
ax.text(pd.Timestamp('2021-02-01'), -6.1, 'earlier\nperiod', fontsize=6, color='#666')
T = S['temporal']
ax.text(0.02, 0.97,
        'later period 2023\u201324 (n=%d):\nMAE %.2f \u2192 %.2f points\nrecalibration fitted on\n2021\u201322 gives %.2f'
        % (T['n_test'], T['mae_auto'], T['mae_hyb'], T['mae_cal']),
        transform=ax.transAxes, fontsize=5.9, va='top')

prep.save(fig, 'Fig5_routing_temporal', OUT)
print('Fig5: 4 panels; specificity %.3f -> %.3f' % (a['spec'], h['spec']))
