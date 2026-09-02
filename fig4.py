"""Figure 4 - demographic gradients are a downstream consequence of the modality gap.

Panels: error-decile composition, education gradient by block, test language.
The subgroup forest plot was moved to Fig5: it contrasts full automation with the
modality-aware hybrid, which is not introduced until that figure, so showing it here
was a forward reference.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np, matplotlib.pyplot as plt, json
from scipy import stats

prep.style()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(OUT, exist_ok=True)
mg = pd.read_pickle(os.path.join(DAT, 'mg.pkl'))
S = json.load(open(os.path.join(DAT, 'summary.json')))
PH = prep.VISION
VB = [i for i in range(1, 31) if i not in PH]
# absolute error normalised by available points, so speech (24 pts) and vision
# (6 pts) blocks are on the same scale and match the regression model in the text
mg['ph_abs'] = (mg[['m%d' % i for i in PH]].fillna(0).sum(axis=1)
                - mg[['d%d' % i for i in PH]].sum(axis=1)).abs() / 6
mg['vb_abs'] = (mg[['m%d' % i for i in VB]].fillna(0).sum(axis=1)
                - mg[['d%d' % i for i in VB]].sum(axis=1)).abs() / 24

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0),
                         gridspec_kw=dict(width_ratios=[1.25, 1.15, 0.85], wspace=0.40))

# (a) demographic composition across error deciles - corrects the published direction
ax = axes[0]
dec = pd.read_csv(os.path.join(DAT, 'deciles.csv'))
for col, c, lb, base in [('old', '#c0392b', 'Age > 70 y', S['base_old']),
                         ('lowedu', '#2c6fbb', 'Education \u2264 9 y', S['base_lowedu']),
                         ('dialect', '#5b8c5a', 'Dialect testing', S['base_dialect'])]:
    ax.plot(dec.dec, 100 * dec[col], 'o-', color=c, ms=3.4, lw=1.2, label=lb)
    ax.axhline(100 * base, color=c, ls=':', lw=.9)
ax.set_xlabel('Decile of |AI \u2212 clinician| total'); ax.set_ylabel('% of decile')
ax.set_title('a  Error-decile composition', loc='left')
ax.legend(fontsize=6, loc='lower left')
ax.text(0.97, 0.03, 'dotted = cohort base rate', transform=ax.transAxes,
        ha='right', fontsize=6, style='italic', color='#666')
ax.set_ylim(20, 100)

# (b) education gradient, split by item block
ax = axes[1]
bands = ['0 y', '1-6 y', '7-9 y', '10-12 y', '>12 y']
g = mg.groupby('eduband', observed=True)
v = [g.get_group(b).vb_abs.mean() if b in g.groups else np.nan for b in bands]
p = [g.get_group(b).ph_abs.mean() if b in g.groups else np.nan for b in bands]
x = np.arange(len(bands))
ax.bar(x - .19, v, .36, color='#2c6fbb')
ax.bar(x + .19, p, .36, color='#c0392b')
ax.set_xticks(x); ax.set_xticklabels(bands, rotation=25, ha='right')
ax.set_ylabel('Mean |error| per available point')
ax.set_title('b  Education gradient by modality', loc='left')
e = mg.dropna(subset=['eduy'])
sl_v = stats.linregress(e.eduy, e.vb_abs)
sl_p = stats.linregress(e.eduy, e.ph_abs)
ax.text(0.03, 0.97, 'slope per school year:\nspeech AI %+.4f (P = %.2f)\nvision AI %+.4f (P = %.0e)'
        % (sl_v.slope, sl_v.pvalue, sl_p.slope, sl_p.pvalue),
        transform=ax.transAxes, va='top', fontsize=6)
ax.set_ylim(0, 0.30)

# (c) test language
ax = axes[2]
gg = mg.groupby('langlab')
lbs = ['Dialect', 'Mandarin']
v = [gg.get_group(l).vb_abs.mean() for l in lbs]
p = [gg.get_group(l).ph_abs.mean() for l in lbs]
x = np.arange(2)
ax.bar(x - .19, v, .36, color='#2c6fbb'); ax.bar(x + .19, p, .36, color='#c0392b')
ax.set_xticks(x)
ax.set_xticklabels(['Dialect\n(n=%d)' % len(gg.get_group('Dialect')),
                    'Mandarin\n(n=%d)' % len(gg.get_group('Mandarin'))])
ax.set_ylabel('Mean |error| per available point')
ax.set_title('c  Test language', loc='left')
t = stats.mannwhitneyu(gg.get_group('Mandarin').vb_abs, gg.get_group('Dialect').vb_abs)
ax.text(0.03, 0.97, 'speech AI, Mandarin\nvs dialect: P = %.2f\nno sig. difference' % t.pvalue,
        transform=ax.transAxes, va='top', fontsize=6)
ax.set_ylim(0, 0.32)

h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ['#2c6fbb', '#c0392b']]
fig.legend(h, ['24 Speech-AI items', '6 Vision-AI items'],
           loc='lower center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.58, -0.11))

prep.save(fig, 'Fig4_demographic', OUT)
print('Fig4: 3 panels; edu slope speech %+.3f (P=%.2f), vision %+.3f (P=%.1e)'
      % (sl_v.slope, sl_v.pvalue, sl_p.slope, sl_p.pvalue))
