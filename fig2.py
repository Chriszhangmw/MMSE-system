"""Figure 2 - high correlation of totals conceals a clinically material measurement gap.

Panels: score agreement, Bland-Altman, screening-decision concordance.
The standalone error histogram was removed: the distribution of differences is already
the vertical spread of the Bland-Altman panel, so it added no information.
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

fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.9),
                         gridspec_kw=dict(width_ratios=[1, 1, 1.25], wspace=0.44))

# (a) AI vs clinician total
ax = axes[0]
hb = ax.hexbin(mg.sum_d, mg.sum_m, gridsize=26, cmap='Blues', mincnt=1, linewidths=0)
ax.plot([0, 30], [0, 30], color='#c0392b', lw=1.1, ls='--')
b, a = np.polyfit(mg.sum_d, mg.sum_m, 1)
xs = np.array([0, 30]); ax.plot(xs, a + b * xs, color='#1a1a1a', lw=1.2)
ax.set_xlabel('Clinician MMSE total'); ax.set_ylabel('AI-MMSE total')
ax.set_title('a  Score agreement', loc='left')
ax.text(0.04, 0.96, 'r = %.3f\nICC(2,1) = %.3f\nslope = %.2f' % (S['r_auto'], S['icc_auto'], b),
        transform=ax.transAxes, va='top', fontsize=7)
ax.set_xlim(-1, 31); ax.set_ylim(-1, 31)
cb = plt.colorbar(hb, ax=ax, pad=0.015, fraction=0.038)
cb.set_label('n', fontsize=6.5); cb.ax.tick_params(labelsize=6)

# (b) Bland-Altman
ax = axes[1]
mean = (mg.sum_d + mg.sum_m) / 2; diff = mg.sum_m - mg.sum_d
ax.scatter(mean, diff, s=3, alpha=.16, color='#2c6fbb', edgecolors='none')
m_, sd_ = diff.mean(), diff.std()
ax.axhline(m_, color='#c0392b', lw=1.2)
for y in (m_ - 1.96 * sd_, m_ + 1.96 * sd_):
    ax.axhline(y, color='#c0392b', lw=1, ls='--')
ax.axhline(0, color='#999', lw=.8)
bb, aa = np.polyfit(mean, diff, 1)
ax.plot([0, 30], [aa, aa + 30 * bb], color='#1a1a1a', lw=1.2)
ax.set_xlabel('Mean of the two administrations'); ax.set_ylabel('AI \u2212 clinician (points)')
ax.set_title('b  Bland\u2013Altman', loc='left')
ax.text(0.03, 0.05, 'bias = %.2f\n95%% LoA  %.1f to %.1f\nproportional bias = %.3f/point'
        % (m_, m_ - 1.96 * sd_, m_ + 1.96 * sd_, bb), transform=ax.transAxes, fontsize=6.6)

# (c) screening-decision concordance at identical education-adjusted cut-offs
ax = axes[2]
sc = S['screen_auto']
M = np.array([[sc['tp'], sc['fn']], [sc['fp'], sc['tn']]])
ax.imshow(M, cmap='Oranges', vmin=0)
for i in range(2):
    for j in range(2):
        ax.text(j, i, '%d' % M[i, j], ha='center', va='center', fontsize=11,
                fontweight='bold', color='white' if M[i, j] > M.max() * .55 else '#1a1a1a')
ax.set_xticks([0, 1]); ax.set_xticklabels(['AI\nscreen +', 'AI\nscreen \u2212'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Clinician +', 'Clinician \u2212'])
ax.set_title('c  Agreement with clinician decision', loc='left')
ax.text(0.5, -0.34, 'equivalent to sensitivity / specificity vs clinician classification', transform=ax.transAxes, ha='center', fontsize=5.4, style='italic')
ax.text(1.15, 0.5,
        'pos. agreement  %.3f\nneg. agreement  %.3f\nPPV  %.3f\nCohen \u03ba  %.3f\n\n+%.1f%% absolute\nover-referral'
        % (sc['sens'], sc['spec'], sc['ppv'], sc['kappa'], 100 * sc['overcall']),
        transform=ax.transAxes, va='center', fontsize=7)
for s in ('top', 'right', 'left', 'bottom'):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)

prep.save(fig, 'Fig2_total_agreement', OUT)
print('Fig2: 3 panels; specificity %.3f' % sc['spec'])
