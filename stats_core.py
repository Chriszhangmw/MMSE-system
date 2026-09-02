"""Core statistics for the AI-MMSE item-level reanalysis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np, json
from sklearn.metrics import cohen_kappa_score
from scipy import stats

DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DAT, exist_ok=True)

rng = np.random.default_rng(0)
mg = prep.load()
PH = prep.VISION                                   # 6 vision-AI items
VB = [i for i in range(1, 31) if i not in PH]      # 24 speech-AI items (incl. Q9)

# ---------- item-level agreement ----------
rows = []
for i in range(1, 31):
    d, m = mg['d%d' % i], mg['m%d' % i]
    ok = d.notna() & m.notna()
    a, b = d[ok].astype(int), m[ok].astype(int)
    k = cohen_kappa_score(a, b)
    boot = []
    idx = np.arange(len(a))
    for _ in range(1000):
        s = rng.choice(idx, len(idx), replace=True)
        aa, bb = a.values[s], b.values[s]
        if len(set(aa)) < 2 or len(set(bb)) < 2:
            continue
        boot.append(cohen_kappa_score(aa, bb))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    rows.append(dict(item=i, label=prep.SHORT[i], cls=prep.item_class(i),
                     domain=prep.DOMAIN[i], n=int(ok.sum()), kappa=k, lo=lo, hi=hi,
                     agree=(a == b).mean(),
                     doc_mean=a.mean(), ai_mean=b.mean(),
                     bias=b.mean() - a.mean(),
                     unscorable=mg['u%d' % i].mean()))
it = pd.DataFrame(rows)
it.to_csv(os.path.join(DAT, 'item_stats.csv'), index=False)

# ---------- block sums ----------
mg['ph_d'] = mg[['d%d' % i for i in PH]].sum(axis=1)
mg['ph_m'] = mg[['m%d' % i for i in PH]].fillna(0).sum(axis=1)
mg['vb_d'] = mg[['d%d' % i for i in VB]].sum(axis=1)
mg['vb_m'] = mg[['m%d' % i for i in VB]].fillna(0).sum(axis=1)
mg['err'] = mg.sum_m - mg.sum_d
mg['abserr'] = mg.err.abs()
mg['ph_err'] = mg.ph_m - mg.ph_d
mg['vb_err'] = mg.vb_m - mg.vb_d
# hybrid: clinician scores the 6 physical items, AI scores the rest
mg['hyb'] = mg.vb_m + mg.ph_d
mg['hyb_err'] = mg.hyb - mg.sum_d

# ---------- screening decisions with education-adjusted cut-offs ----------
mg['cut'] = mg.eduy.apply(prep.edu_cutoff)
for src, tag in [('sum_d', 'ref'), ('sum_m', 'ai'), ('hyb', 'hy')]:
    mg['pos_' + tag] = (mg[src] <= mg['cut']).astype(float).where(mg.cut.notna())


def screen(tag, sub=None):
    s = mg if sub is None else sub
    s = s.dropna(subset=['pos_ref', 'pos_' + tag])
    tp = ((s.pos_ref == 1) & (s['pos_' + tag] == 1)).sum()
    fn = ((s.pos_ref == 1) & (s['pos_' + tag] == 0)).sum()
    fp = ((s.pos_ref == 0) & (s['pos_' + tag] == 1)).sum()
    tn = ((s.pos_ref == 0) & (s['pos_' + tag] == 0)).sum()
    return dict(n=int(len(s)), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1),
                kappa=cohen_kappa_score(s.pos_ref, s['pos_' + tag]),
                overcall=(s['pos_' + tag].mean() - s.pos_ref.mean()))


summary = dict(
    n=int(len(mg)),
    icc_auto=prep.icc21(mg.sum_d, mg.sum_m),
    icc_hyb=prep.icc21(mg.sum_d, mg.hyb),
    bias_auto=float(mg.err.mean()), sd_auto=float(mg.err.std()),
    bias_hyb=float(mg.hyb_err.mean()), sd_hyb=float(mg.hyb_err.std()),
    r_auto=float(np.corrcoef(mg.sum_d, mg.sum_m)[0, 1]),
    r_hyb=float(np.corrcoef(mg.sum_d, mg.hyb)[0, 1]),
    kappa_phys=float(it[it.cls == 'Vision AI'].kappa.mean()),
    kappa_verb=float(it[it.cls == 'Speech AI'].kappa.mean()),
    mwu_p=float(stats.mannwhitneyu(it[it.cls == 'Vision AI'].kappa,
                                   it[it.cls == 'Speech AI'].kappa).pvalue),
    share_abs_err_physical=float(mg.ph_err.abs().sum() / (mg.ph_err.abs().sum() + mg.vb_err.abs().sum())),
    screen_auto=screen('ai'), screen_hyb=screen('hy'),
    unscorable_any=float((mg.n_unscorable > 0).mean()),
)

# ---------- subgroup error ----------
sub = []
for var, band in [('eduband', 'Education'), ('ageband', 'Age'), ('langlab', 'Language')]:
    if var == 'langlab':
        mg['langlab'] = np.where(mg.mandarin, 'Mandarin', 'Dialect')
    for g, s in mg.groupby(var, observed=True):
        if len(s) < 25: continue
        sub.append(dict(var=band, group=str(g), n=len(s),
                        abserr=s.abserr.mean(), ph=s.ph_err.abs().mean(),
                        vb=s.vb_err.abs().mean(), hyb=s.hyb_err.abs().mean(),
                        bias=s.err.mean()))
pd.DataFrame(sub).to_csv(os.path.join(DAT, 'subgroup.csv'), index=False)

# ---------- decile composition (replication of the manuscript's Fig 2a) ----------
q = mg.dropna(subset=['abserr']).copy()
q['dec'] = pd.qcut(q.abserr.rank(method='first'), 10, labels=False) + 1
dec = q.groupby('dec').apply(lambda s: pd.Series({
    'old': (s.age > 70).mean(), 'lowedu': (s.eduy <= 9).mean(),
    'dialect': (~s.mandarin).mean(), 'abserr': s.abserr.mean()}), include_groups=False)
dec.to_csv(os.path.join(DAT, 'deciles.csv'))
summary['base_old'] = float((mg.age > 70).mean())
summary['base_lowedu'] = float((mg.eduy <= 9).mean())
summary['base_dialect'] = float((~mg.mandarin).mean())

# ---------- temporal split ----------
tr = mg[mg.year <= 2022]; te = mg[mg.year >= 2023]
summary['temporal'] = {
    'n_train': int(len(tr)), 'n_test': int(len(te)),
    'icc_auto_tr': prep.icc21(tr.sum_d, tr.sum_m), 'icc_auto_te': prep.icc21(te.sum_d, te.sum_m),
    'icc_hyb_tr': prep.icc21(tr.sum_d, tr.hyb), 'icc_hyb_te': prep.icc21(te.sum_d, te.hyb),
    'bias_tr': float(tr.err.mean()), 'bias_te': float(te.err.mean()),
}

# linear recalibration of AI total fitted on 2021-22, applied to 2023-24
b, a = np.polyfit(tr.sum_m, tr.sum_d, 1)
te = te.copy(); tr = tr.copy()
te['cal'] = a + b * te.sum_m
bh, ah = np.polyfit(tr.hyb, tr.sum_d, 1)
te['calh'] = ah + bh * te.hyb
summary['temporal'].update(
    mae_auto=float((te.sum_m - te.sum_d).abs().mean()),
    mae_cal=float((te.cal - te.sum_d).abs().mean()),
    mae_hyb=float((te.hyb - te.sum_d).abs().mean()),
    mae_hyb_cal=float((te.calh - te.sum_d).abs().mean()),
    slope=float(b), intercept=float(a))

with open(os.path.join(DAT, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
mg.to_pickle(os.path.join(DAT, 'mg.pkl'))
print(json.dumps(summary, indent=1, default=lambda x: round(float(x), 4)))
