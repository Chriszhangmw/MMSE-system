"""Supplementary Figure S1 - robustness of the modality finding.

a  Prevalence-robust agreement: Cohen's kappa vs Gwet AC1 by modality.
b  Decomposition of vision-item disagreement into abstention vs confident error.
c  Participant-clustered bootstrap of the per-point modality gap.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np, matplotlib.pyplot as plt, json

prep.style()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
ac = pd.read_csv(os.path.join(DAT, 'item_ac1.csv'))
R = json.load(open(os.path.join(DAT, 'robustness.json')))
PAL = prep.PAL

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0),
                         gridspec_kw=dict(width_ratios=[1.05, 1.1, 1], wspace=0.42))
fig.subplots_adjust(bottom=0.22)

# (a) kappa vs AC1, coloured by modality
ax = axes[0]
for c in ['Speech AI', 'Vision AI']:
    s = ac[ac.cls == c]
    ax.scatter(s.kappa, s.ac1, s=30, color=PAL[c], edgecolors='white',
               linewidths=.5, label=c, zorder=3)
ax.plot([0, 1], [0, 1], color='#999', lw=.8, ls='--')
ax.set_xlabel("Cohen's \u03ba"); ax.set_ylabel('Gwet AC1')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('a  Prevalence-robust agreement', loc='left')
ax.legend(fontsize=6.5, loc='lower right')
ax.text(0.03, 0.95, 'speech AC1 %.2f\nvision AC1 %.2f\nP = %.0e'
        % (R['ac1_speech'], R['ac1_vision'], R['ac1_mwu_p']),
        transform=ax.transAxes, va='top', fontsize=6.3)

# (b) abstention vs confident error, per vision item
ax = axes[1]
mg = pd.read_pickle(os.path.join(DAT, 'mg.pkl'))
PH = prep.VISION
labels = {25: 'Q25 eyes', 26: 'Q26 R-hand', 27: 'Q27 fold',
          28: 'Q28 lap', 29: 'Q29 write', 30: 'Q30 copy'}
ab, cw = [], []
for i in PH:
    d = mg['d%d' % i]; m = mg['m%d' % i]; u = mg['u%d' % i]
    ab.append(100 * (u == 1).mean())
    ok = (u == 0) & m.notna() & d.notna()
    cw.append(100 * ((m != d) & ok).sum() / ok.sum())
y = np.arange(len(PH))
ax.barh(y + .2, cw, .38, color='#c0392b', label='Confident disagreement')
ax.barh(y - .2, ab, .38, color='#8e8e8e', label='Abstention')
ax.set_yticks(y); ax.set_yticklabels([labels[i] for i in PH], fontsize=6.3)
ax.set_xlabel('% of assessments'); ax.set_title('b  Vision error is not abstention', loc='left')
ax.legend(fontsize=6, loc='upper right')
ax.set_xlim(0, 58)
ax.text(0.5, -0.30, 'Only %.1f%% of vision-item disagreement is abstention' % R['vision_abstention_share_of_disagreement'],
        transform=ax.transAxes, ha='center', va='top', fontsize=6.3, style='italic')

# (c) clustered bootstrap of the modality gap
ax = axes[2]
gap = R['modality_gap_per_point']; lo, hi = R['modality_gap_ci']
ax.errorbar([gap], [0], xerr=[[gap - lo], [hi - gap]], fmt='o', ms=7,
            color='#2c6fbb', capsize=4, lw=1.6)
ax.axvline(0, color='#1a1a1a', lw=.8, ls='--')
ax.set_yticks([]); ax.set_xlim(-0.02, 0.22)
ax.set_xlabel('Vision \u2212 speech |error| per available point')
ax.set_title('c  Gap holds under patient clustering', loc='left')
ax.text(0.5, 0.7, '%.3f (95%% CI %.3f\u2013%.3f)\nresampling %d unique patients'
        % (gap, lo, hi, R['n_persons']),
        transform=ax.transAxes, ha='center', fontsize=6.3)

prep.save(fig, 'FigS1_robustness', OUT)
print('FigS1 done: AC1 P=%.1e, vision abstention %.1f%%, gap CI [%.3f,%.3f]'
      % (R['ac1_mwu_p'], R['vision_abstention_share_of_disagreement'], lo, hi))
