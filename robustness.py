"""Robustness analyses requested at review: patient-level independence and
clustered CIs, prevalence-robust agreement (Gwet AC1), and decomposition of
disagreement into abstention-driven versus confident-error components.

All statistics are computed on the exact 1,899-assessment analysis set. Unique-
patient identity needs the raw clinical workbook (names are stripped from the
distributed CSV); the resulting per-assessment patient ids are cached in
data/person_ids.csv so the package reproduces the clustered bootstrap and patient
counts without the workbook.

Writes data/robustness.json and data/item_ac1.csv.
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep, pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import stats

DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
mg = pd.read_pickle(os.path.join(DAT, 'mg.pkl'))
rng = np.random.default_rng(0)
PH = prep.VISION
VB = [i for i in range(1, 31) if i not in PH]
out = {}
RAW_OK = os.path.exists(prep.SRC)
IDS = os.path.join(DAT, 'person_ids.csv')


# ---------- prevalence-robust agreement (Gwet AC1) ----------
def gwet_ac1(a, b):
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    po = (a == b).mean()
    pe = sum(((a == k).sum() + (b == k).sum()) / (2 * n) *
             (1 - ((a == k).sum() + (b == k).sum()) / (2 * n)) for k in (0, 1))
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan


rows = []
for i in range(1, 31):
    dd = mg['d%d' % i]; mm = mg['m%d' % i]; ok = dd.notna() & mm.notna()
    a = dd[ok].astype(int); b = mm[ok].astype(int)
    rows.append(dict(item=i, cls=prep.item_class(i),
                     kappa=cohen_kappa_score(a, b), ac1=gwet_ac1(a, b),
                     raw_agree=(a == b).mean()))
ac = pd.DataFrame(rows)
ac.to_csv(os.path.join(DAT, 'item_ac1.csv'), index=False)
out['ac1_speech'] = float(ac[ac.cls == 'Speech AI'].ac1.mean())
out['ac1_vision'] = float(ac[ac.cls == 'Vision AI'].ac1.mean())
out['ac1_mwu_p'] = float(stats.mannwhitneyu(ac[ac.cls == 'Speech AI'].ac1,
                                            ac[ac.cls == 'Vision AI'].ac1).pvalue)


# ---------- error decomposition (denominator = disagreement events) ----------
# For each modality: among all item x assessment instances that DISAGREE with the
# clinician, what fraction is due to abstention (device abstained, clinician scored 1)
# versus a returned-but-wrong score. This is the share OF DISAGREEMENT, not of all
# observations.
def decomp(items):
    abst = wrong = 0
    for i in items:
        dd = mg['d%d' % i]; mm = mg['m%d' % i]; u = mg['u%d' % i]
        abst += int(((u == 1) & (dd == 1)).sum())                       # abstention that costs a point
        wrong += int(((u == 0) & mm.notna() & dd.notna() & (mm != dd)).sum())  # returned wrong score
    return abst, wrong


va, vw = decomp(PH); sa, sw = decomp(VB)
out['vision_abstention_share_of_disagreement'] = float(100 * va / (va + vw))
out['speech_abstention_share_of_disagreement'] = float(100 * sa / (sa + sw))
out['vision_confident_wrong_events'] = int(vw)
out['vision_abstention_events'] = int(va)
# also report abstention as a share of all vision observations, for transparency
n_obs = len(mg) * len(PH)
out['vision_abstention_share_of_observations'] = float(100 * (mg[['u%d' % i for i in PH]].sum().sum()) / n_obs)


# ---------- per-assessment modality gap ----------
mg['sp'] = (mg[['m%d' % i for i in VB]].fillna(0).sum(1) - mg[['d%d' % i for i in VB]].sum(1)).abs() / 24
mg['vi'] = (mg[['m%d' % i for i in PH]].fillna(0).sum(1) - mg[['d%d' % i for i in PH]].sum(1)).abs() / 6
out['modality_gap_per_point'] = float((mg['vi'] - mg['sp']).mean())


# ---------- patient identity on the exact analysis set ----------
def build_person_ids():
    """Assign a patient id to each of the 1,899 analysis rows.

    Reconstructs the analysis set with names by replicating prep.load_raw's filters
    exactly, so the count is over the same 1,899 assessments and not a superset.
    A patient is (name, sex) with birth year (year - age) within a 2-year band.
    """
    d = pd.read_excel(prep.SRC, sheet_name='医生评分')
    m = pd.read_excel(prep.SRC, sheet_name='机器评分')
    d.columns = [c.strip() for c in d.columns]; m.columns = [c.strip() for c in m.columns]
    d = d.rename(columns={'姓    名': 'name', '性    别': 'sex', '年    龄': 'age',
                          '受教育年限': 'eduy', '测试日期': 'date', '总分': 'tot_d'})
    m = m.rename(columns={'姓    名': 'name', '测试日期': 'date'})
    itemcols = [c for c in d.columns if re.match(r'^\d+、', c)]
    numc = {c: int(c.split('、')[0]) for c in itemcols}
    for df in (d, m):
        df['name'] = df['name'].astype(str).str.strip()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['key'] = df['name'] + '|' + df['date'].astype(str)
    mm = d[['key', 'name', 'date', 'sex', 'age', 'eduy'] + itemcols].merge(
        m[['key'] + itemcols], on='key', suffixes=('_d', '_m')).drop_duplicates('key')
    # replicate prep item cleaning and the exact row filters
    for c in itemcols:
        i = numc[c]
        dv = pd.to_numeric(mm[c + '_d'], errors='coerce')
        mv = pd.to_numeric(mm[c + '_m'], errors='coerce')
        mm['d%d' % i] = dv.where(dv.isin([0, 1]))
        mm['m%d' % i] = mv.where(mv.isin([0, 1]))
        mm['u%d' % i] = (mv < 0).astype(int)
    mm['age'] = pd.to_numeric(mm['age'], errors='coerce').where(lambda s: s.between(18, 110))
    mm['eduy'] = pd.to_numeric(mm['eduy'], errors='coerce').where(lambda s: s.between(0, 25))
    dcols = ['d%d' % i for i in range(1, 31)]; mcols = ['m%d' % i for i in range(1, 31)]
    mm['sum_d'] = mm[dcols].sum(axis=1, min_count=30)
    n_m_missing = mm[mcols].isna().sum(axis=1) - mm[['u%d' % i for i in range(1, 31)]].sum(axis=1)
    mm = mm[mm['sum_d'].notna() & (n_m_missing <= 0)]
    mm['year'] = mm['date'].dt.year
    mm = mm[mm['year'].between(2021, 2024)]
    mm['byr'] = mm['year'] - mm['age']
    keys = {}
    for (nm, sx), g in mm.groupby(['name', 'sex']):
        gb = g.sort_values('byr'); cur = None; k = 0
        for idx, b in gb['byr'].items():
            if cur is None:
                cur = b
            elif abs(b - cur) > 2:
                k += 1; cur = b
            keys[idx] = f'{nm}|{sx}|{k}'
    mm['person'] = mm.index.map(keys)
    mm['pid'] = mm['person'].astype('category').cat.codes
    mm['period'] = np.where(mm['year'] <= 2022, 'dev', 'test')
    mm['sig'] = (mm['date'].dt.strftime('%Y-%m-%d') + '|' + mm['age'].astype('Int64').astype(str)
                 + '|' + mm['sex'].map({'男': 'M', '女': 'F'}).astype(str)
                 + '|' + mm['sum_d'].astype('Int64').astype(str))
    return mm


STAT_CACHE = os.path.join(DAT, 'patient_stats.json')

if RAW_OK:
    mm = build_person_ids()
    npat = int(mm['person'].nunique())
    reps = int(len(mm) - npat)
    pstats = {
        'n_persons': npat,
        'n_assessments_for_id': int(len(mm)),
        'n_repeat_assessments': reps,
        'pct_repeat': float(100 * reps / len(mm)),
        'n_multi_visit_patients': int((mm['person'].value_counts() > 1).sum()),
        'max_visits': int(mm['person'].value_counts().max()),
        'n_persons_both_periods': int((mm.groupby('person')['period'].nunique() > 1).sum()),
    }
    # export the per-signature id (for clustering) and the exact stats (for reporting)
    mm[['sig', 'pid']].drop_duplicates('sig').to_csv(IDS, index=False)
    json.dump(pstats, open(STAT_CACHE, 'w'), indent=1)
else:
    pstats = json.load(open(STAT_CACHE))
out.update(pstats)

# align patient id onto mg by signature for the clustered bootstrap. Signature
# collisions cause a small undercount of clusters, which is conservative for the CI;
# the reported patient counts come from the exact name-based identity above, not from
# this mapping.
sigmap = pd.read_csv(IDS).drop_duplicates('sig').set_index('sig')['pid']
mg['sig'] = (pd.to_datetime(mg['date']).dt.strftime('%Y-%m-%d') + '|'
             + mg['age'].astype('Int64').astype(str) + '|' + mg['sex'].astype(str)
             + '|' + mg['sum_d'].astype('Int64').astype(str))
mg['pid'] = mg['sig'].map(sigmap)
out['bootstrap_person_coverage'] = float(mg['pid'].notna().mean())
mg['clust'] = mg['pid'].astype('object').where(mg['pid'].notna(),
                                               'row_' + mg.index.to_series().astype(str))
gi = np.array([np.asarray(v) for v in mg.groupby('clust').groups.values()], dtype=object)
boot = []
for _ in range(2000):
    idx = np.concatenate([gi[j] for j in rng.choice(len(gi), len(gi), replace=True)])
    s = mg.loc[idx]
    boot.append((s['vi'] - s['sp']).mean())
out['modality_gap_ci'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

json.dump(out, open(os.path.join(DAT, 'robustness.json'), 'w'), indent=1)
print('robustness: patients=%d (repeat %.1f%%) | AC1 %.2f/%.2f P=%.1e | vision abstention %.1f%% of disagreement | gap %.3f CI [%.3f,%.3f]'
      % (out['n_persons'], out.get('pct_repeat', 0), out['ac1_speech'], out['ac1_vision'],
         out['ac1_mwu_p'], out['vision_abstention_share_of_disagreement'],
         out['modality_gap_per_point'], out['modality_gap_ci'][0], out['modality_gap_ci'][1]))
