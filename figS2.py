"""Supplementary Figure 2 - the decision paradox is not a cut-off artefact, and the
education gradient reflects exposure to the vision-item failure mode.

a  Threshold sensitivity: positive and negative agreement, and over-referral, as the
   single MMSE cut-off applied to both administrations is swept from 16 to 26.
b  Clinician vision-item pass rate rises steeply with education, so educated patients
   are far more exposed to the failure mode of the device missing a passed vision item.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np, matplotlib.pyplot as plt

prep.style()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
mg = pd.read_pickle(os.path.join(DAT, 'mg.pkl'))
PH = prep.VISION
VB = [i for i in range(1, 31) if i not in PH]
mg['sum_d'] = mg[['d%d' % i for i in range(1, 31)]].sum(1)
mg['sum_m'] = mg[['m%d' % i for i in range(1, 31)]].fillna(0).sum(1)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0),
                         gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.42))

# (a) threshold sweep
ax = axes[0]
cuts = list(range(16, 27))
posa, nega, over = [], [], []
for c in cuts:
    pd_ = mg.sum_d <= c; pm_ = mg.sum_m <= c
    tp = (pd_ & pm_).sum(); fn = (pd_ & ~pm_).sum()
    fp = (~pd_ & pm_).sum(); tn = (~pd_ & ~pm_).sum()
    posa.append(tp / max(tp + fn, 1)); nega.append(tn / max(tn + fp, 1))
    over.append(100 * (pm_.mean() - pd_.mean()))
ax.plot(cuts, posa, 'o-', color='#2c6fbb', ms=3.4, lw=1.3, label='Positive agreement')
ax.plot(cuts, nega, 'o-', color='#c0392b', ms=3.4, lw=1.3, label='Negative agreement')
ax.set_xlabel('MMSE cut-off (screen-positive if ≤)')
ax.set_ylabel('Agreement with clinician decision')
ax.set_ylim(0, 1.02); ax.legend(fontsize=6.3, loc='center left')
ax.set_title('a  Robust to the screening threshold', loc='left')
ax.axvspan(16.5, 24.5, color='#f4f4f4', zorder=0)
ax.text(0.97, 0.10, 'positive agreement ≥ 0.98\nthroughout; negative\nagreement low at every cut-off',
        transform=ax.transAxes, ha='right', fontsize=5.8)

# (b) composition mechanism
ax = axes[1]
mg['eb'] = pd.cut(mg['eduy'], [-1, 6, 9, 30], labels=['≤6 y', '7–9 y', '>9 y'])
rate = mg.groupby('eb', observed=True).apply(
    lambda g: g[['d%d' % i for i in PH]].values.mean(), include_groups=False)
ax.bar(range(len(rate)), 100 * rate.values, color='#8e5ea2', width=.6)
for i, v in enumerate(rate.values):
    ax.text(i, 100 * v + 1.5, '%.0f%%' % (100 * v), ha='center', fontsize=7)
ax.set_xticks(range(len(rate))); ax.set_xticklabels(rate.index)
ax.set_xlabel('Education'); ax.set_ylabel('Clinician vision-item pass rate (%)')
ax.set_ylim(0, 78)
ax.set_title('b  Exposure to the failure mode', loc='left')
r = np.corrcoef(mg.dropna(subset=['eduy']).eduy,
                mg.dropna(subset=['eduy'])[['d%d' % i for i in PH]].mean(1))[0, 1]
ax.text(0.03, 0.95, 'educated patients pass the\nvision items far more often\n(r = %.2f), so they are more\nexposed to the device missing\na passed item' % r,
        transform=ax.transAxes, va='top', fontsize=5.8)

prep.save(fig, 'FigS2_threshold_composition', OUT)
print('FigS2 done; neg agreement range %.2f-%.2f over cuts 16-26' % (min(nega), max(nega)))
