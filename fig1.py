"""Figure 1 - deployment context of the automated MMSE workstation.

Panels: accrual, reference-score distribution with screening cut-offs, device abstention.
The age/sex and education distributions that previously occupied two panels carry no
argument and are reported in Table 1 instead (written to data/table1.csv).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np, matplotlib.pyplot as plt

prep.style()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(OUT, exist_ok=True)
mg = pd.read_pickle(os.path.join(DAT, 'mg.pkl'))

fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7),
                         gridspec_kw=dict(width_ratios=[1.8, 1, 1], wspace=0.36))

# (a) accrual over the deployment period
ax = axes[0]
mg['q'] = mg.date.dt.to_period('Q').dt.to_timestamp()
t = mg.groupby(['q', 'mandarin']).size().unstack(fill_value=0)
ax.bar(t.index, t[False], width=70, color='#2c6fbb', label='Dialect (Chongqing)')
ax.bar(t.index, t.get(True, 0), width=70, bottom=t[False], color='#f0a13c', label='Mandarin')
ax.set_ylabel('Assessments per quarter')
ax.set_title('a  Outpatient accrual, 2021–2024', loc='left')
ax.legend(loc='upper left')
ax.tick_params(axis='x', rotation=30, labelsize=6.5)
ax.text(0.985, 0.06, 'n = %d paired administrations' % len(mg), transform=ax.transAxes,
        ha='right', fontsize=7, style='italic')

# (b) reference score distribution and the cut-offs used throughout
ax = axes[1]
ax.hist(mg.sum_d, bins=np.arange(-0.5, 31, 1), color='#7d7d7d')
for c, col, lb in [(17, '#c0392b', '0 y: ≤17'), (20, '#d98c1f', '1–6 y: ≤20'),
                   (24, '#2c6fbb', '≥7 y: ≤24')]:
    ax.axvline(c + 0.5, color=col, lw=1.2, ls='--', label=lb)
ax.set_xlabel('Clinician MMSE total'); ax.set_ylabel('Count')
ax.set_title('b  Reference score', loc='left')
ax.legend(title='Screen-positive if ≤', title_fontsize=6.5, loc='upper left', fontsize=6.5)

# (c) device abstention
ax = axes[2]
u = mg.n_unscorable.value_counts().sort_index()
u = u[u.index <= 6]
ax.bar(u.index, 100 * u.values / len(mg), color='#8e5ea2')
ax.set_xlabel('Unscorable items'); ax.set_ylabel('% of assessments')
ax.set_title('c  Device abstention', loc='left')
ax.text(0.95, 0.9, '%.1f%% of assessments\ncontain ≥1 unscorable item'
        % (100 * (mg.n_unscorable > 0).mean()),
        transform=ax.transAxes, ha='right', va='top', fontsize=6.5, style='italic')

prep.save(fig, 'Fig1_cohort', OUT)

# ---- Table 1: the demographics removed from the old panels b and c ----
def msd(s): return '%.2f \u00b1 %.2f' % (s.mean(), s.std())
def npct(k, n): return '%d (%.1f%%)' % (k, 100 * k / n)

n = len(mg)
rows = [('Total assessments, n', str(n)),
        ('Age, years (mean \u00b1 SD)', msd(mg.age.dropna()))]
for b in ['<60', '60-69', '70-79', '>=80']:
    rows.append(('  %s' % b, npct(int((mg.ageband == b).sum()), n)))
rows.append(('Sex, n (%)', ''))
for s, lb in [('F', '  Female'), ('M', '  Male')]:
    rows.append((lb, npct(int((mg.sex == s).sum()), n)))
rows.append(('Education, years (mean \u00b1 SD)', msd(mg.eduy.dropna())))
for b in ['0 y', '1-6 y', '7-9 y', '10-12 y', '>12 y']:
    rows.append(('  %s' % b, npct(int((mg.eduband == b).sum()), n)))
rows.append(('Test language, n (%)', ''))
rows.append(('  Mandarin', npct(int(mg.mandarin.sum()), n)))
rows.append(('  Dialect', npct(int((~mg.mandarin).sum()), n)))
rows.append(('MMSE total (mean \u00b1 SD, max 30)', ''))
rows.append(('  Clinician', msd(mg.sum_d)))
rows.append(('  AI-MMSE', msd(mg.sum_m)))
rows.append(('Assessments with \u22651 unscorable item',
             npct(int((mg.n_unscorable > 0).sum()), n)))
pd.DataFrame(rows, columns=['Variable', 'Value']).to_csv(
    os.path.join(DAT, 'table1.csv'), index=False)
print('Fig1: 3 panels; Table 1 -> data/table1.csv')
