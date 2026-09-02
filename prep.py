"""Shared data preparation for the AI-MMSE item-level reanalysis."""
import pandas as pd, numpy as np, re

import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Primary input: the de-identified, cleaned analysis dataset shipped with this package.
CLEAN = os.path.join(ROOT, 'data', 'analysis_dataset.csv')
# Fallback: rebuild from the raw clinical workbook (not distributed).
SRC = os.environ.get('AIMMSE_RAW',
                     os.path.join(ROOT, 'data', '机器数据更新_2024_12_23_.xlsx'))

# --- item taxonomy ---
# Two AI modalities administer the 30-item MMSE:
#   Speech AI (ASR): 24 items eliciting a spoken response, including item 9 (floor),
#                    which the system scores from the patient's spoken answer.
#   Vision AI:       6 items requiring a visual judgement of a written product or an action.
# Within Vision AI the six items are handled by different computer-vision components:
#   static  = optical character / figure recognition of a produced artefact (write, copy)
#   dynamic = pose/action estimation of a movement (eyes, hand, fold, lap)
VISION       = [25, 26, 27, 28, 29, 30]
VISION_STATIC  = [29, 30]              # write a sentence; copy pentagons  (OCR / figure recognition)
VISION_DYNAMIC = [25, 26, 27, 28]     # close eyes; take (R hand); fold; place on lap (pose/action)
SPEECH_EXTRA = [9]                     # floor: scored from the spoken answer, grouped with speech

SHORT = {
 1:'Q1 year', 2:'Q2 season', 3:'Q3 month', 4:'Q4 date', 5:'Q5 weekday',
 6:'Q6 city', 7:'Q7 district', 8:'Q8 street', 9:'Q9 floor', 10:'Q10 place',
 11:'Q11 reg. ball', 12:'Q12 reg. flag', 13:'Q13 reg. tree',
 14:'Q14 100-7', 15:'Q15 93-7', 16:'Q16 86-7', 17:'Q17 79-7', 18:'Q18 72-7',
 19:'Q19 rec. ball', 20:'Q20 rec. flag', 21:'Q21 rec. tree',
 22:'Q22 name pencil', 23:'Q23 name watch', 24:'Q24 repeat sentence',
 25:'Q25 close eyes', 26:'Q26 take paper (R hand)', 27:'Q27 fold paper',
 28:'Q28 place on lap', 29:'Q29 write sentence', 30:'Q30 copy pentagons',
}

DOMAIN = {}
for i in range(1, 11):  DOMAIN[i] = 'Orientation'
for i in range(11, 14): DOMAIN[i] = 'Registration'
for i in range(14, 19): DOMAIN[i] = 'Attention/calc.'
for i in range(19, 22): DOMAIN[i] = 'Recall'
for i in range(22, 31): DOMAIN[i] = 'Language/praxis'


def item_class(i):
    """Top-level sensing modality: which AI subsystem scores the item."""
    if i in VISION: return 'Vision AI'
    return 'Speech AI'


def vision_subtype(i):
    """Within Vision AI, which class of computer-vision task."""
    if i in VISION_STATIC:  return 'Static image (OCR / figure)'
    if i in VISION_DYNAMIC: return 'Dynamic action (pose)'
    return None


def load():
    """Return the analysis set.

    Reads data/analysis_dataset.csv when present (de-identified, already cleaned).
    Otherwise rebuilds it from the raw workbook via load_raw().
    """
    if os.path.exists(CLEAN):
        mg = pd.read_csv(CLEAN, parse_dates=['date'])
        mg['eduband'] = pd.Categorical(mg['eduband'],
                                       ['0 y', '1-6 y', '7-9 y', '10-12 y', '>12 y'], ordered=True)
        mg['ageband'] = pd.Categorical(mg['ageband'], ['<60', '60-69', '70-79', '>=80'], ordered=True)
        return mg
    return load_raw()


def load_raw():
    d = pd.read_excel(SRC, sheet_name='医生评分')
    m = pd.read_excel(SRC, sheet_name='机器评分')
    d.columns = [c.strip() for c in d.columns]
    m.columns = [c.strip() for c in m.columns]
    d = d.rename(columns={'姓    名': 'name', '性    别': 'sex', '年    龄': 'age',
                          '受教育年限': 'eduy', '测试语言': 'lang', '测试日期': 'date',
                          '总分': 'tot_d', '认知功能': 'label', '测试地点': 'site'})
    m = m.rename(columns={'姓    名': 'name', '测试日期': 'date', '总分': 'tot_m',
                          '认知功能': 'label_m'})

    itemcols = [c for c in d.columns if re.match(r'^\d+、', c)]
    num = {c: int(c.split('、')[0]) for c in itemcols}

    for df in (d, m):
        df['name'] = df['name'].astype(str).str.strip()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['key'] = df['name'] + '|' + df['date'].astype(str)

    keep_d = ['key', 'name', 'date', 'sex', 'age', 'eduy', 'lang', 'site', 'tot_d', 'label'] + itemcols
    keep_m = ['key', 'tot_m', 'label_m'] + itemcols
    mg = d[keep_d].merge(m[keep_m], on='key', suffixes=('_d', '_m'))
    mg = mg.drop_duplicates('key')

    # tidy numeric item matrix.
    # Machine sheet uses -1 / -2 as sentinel codes for "unscorable" (not a score of 0).
    # Clinician sheet contains a handful of impossible values (11, 31) -> treat as missing.
    new = {}
    for c in itemcols:
        i = num[c]
        dv = pd.to_numeric(mg[c + '_d'], errors='coerce')
        mv = pd.to_numeric(mg[c + '_m'], errors='coerce')
        new['d%d' % i] = dv.where(dv.isin([0, 1]))
        new['m%d' % i] = mv.where(mv.isin([0, 1]))
        new['u%d' % i] = (mv < 0).astype(int)          # unscorable flag
    mg = pd.concat([mg, pd.DataFrame(new, index=mg.index)], axis=1)
    mg['tot_d'] = pd.to_numeric(mg['tot_d'], errors='coerce')
    mg['tot_m'] = pd.to_numeric(mg['tot_m'], errors='coerce')
    # Implausible demographic entries (data-entry errors) -> missing.
    # age: 7 records outside 18-110 (values of 0, 123, 377, 976)
    # eduy: 5 records above 25 completed school years (30, 32, 70, 77, 83)
    age = pd.to_numeric(mg['age'], errors='coerce')
    edu = pd.to_numeric(mg['eduy'], errors='coerce')
    mg['age'] = age.where(age.between(18, 110))
    mg['eduy'] = edu.where(edu.between(0, 25))
    mg['lang'] = mg['lang'].astype(str).str.strip()
    mg['mandarin'] = mg['lang'].eq('普通话')
    mg['year'] = mg['date'].dt.year

    # Rebuild totals from items so hybrid arithmetic is internally consistent.
    # An unscorable item contributes 0, which is the device's clinical behaviour.
    dcols = ['d%d' % i for i in range(1, 31)]
    mcols = ['m%d' % i for i in range(1, 31)]
    mg['sum_d'] = mg[dcols].sum(axis=1, min_count=30)
    mg['sum_m'] = mg[mcols].fillna(0).sum(axis=1)
    mg['n_unscorable'] = mg[['u%d' % i for i in range(1, 31)]].sum(axis=1)
    mg['n_m_missing'] = mg[mcols].isna().sum(axis=1) - mg['n_unscorable']

    mg = mg[mg['sum_d'].notna() & (mg['n_m_missing'] <= 0)].copy()
    mg = mg[mg['year'].between(2021, 2024)].copy()
    mg['eduband'] = pd.cut(mg['eduy'], [-1, 0, 6, 9, 12, 30],
                           labels=['0 y', '1-6 y', '7-9 y', '10-12 y', '>12 y'])
    mg['ageband'] = pd.cut(mg['age'], [0, 60, 70, 80, 120],
                           labels=['<60', '60-69', '70-79', '>=80'])

    # ---- normalise categorical labels to ASCII so figures are locale-independent ----
    mg['sex'] = mg['sex'].astype(str).str.strip().map({'女': 'F', '男': 'M'})
    mg['langlab'] = np.where(mg['mandarin'], 'Mandarin', 'Dialect')

    # ---- de-identification: drop names and free-text keys, keep a stable surrogate id ----
    mg = mg.sort_values(['date']).reset_index(drop=True)
    mg.insert(0, 'pid', ['P%04d' % (i + 1) for i in range(len(mg))])
    keep = (['pid', 'date', 'year', 'sex', 'age', 'eduy', 'eduband', 'ageband',
             'lang', 'langlab', 'mandarin', 'site', 'sum_d', 'sum_m', 'n_unscorable']
            + ['d%d' % i for i in range(1, 31)]
            + ['m%d' % i for i in range(1, 31)]
            + ['u%d' % i for i in range(1, 31)])
    return mg[keep].copy()


def icc21(x, y):
    """ICC(2,1), two-way random, absolute agreement, single measure."""
    a = np.vstack([np.asarray(x, float), np.asarray(y, float)]).T
    a = a[~np.isnan(a).any(1)]
    n, k = a.shape
    gm = a.mean()
    msr = k * ((a.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((a.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def edu_cutoff(eduy):
    """Education-adjusted Chinese MMSE screening cut-offs (Zhang et al. 1990 convention).
    Screen-positive if total <= cut-off."""
    if pd.isna(eduy): return np.nan
    if eduy == 0:  return 17
    if eduy <= 6:  return 20
    return 24


# --- shared matplotlib style ---
def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.dpi': 130, 'savefig.dpi': 400, 'font.size': 8,
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 9, 'axes.labelsize': 8, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'legend.frameon': False,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
        # --- Illustrator compatibility ---
        # Type 42 (TrueType) keeps every label as live, editable text in the PDF.
        # The matplotlib default (Type 3) is rendered by Illustrator as outlines.
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'pdf.compression': 0,          # readable object stream, easier to edit
        'svg.fonttype': 'none',
        'figure.autolayout': False,
        'svg.hashsalt': 'ai-mmse-reanalysis',
    })


def save(fig, name, out_dir):
    """Write a figure as PDF + SVG + PNG with deterministic, timestamp-free metadata,
    so a fresh run reproduces byte-identical files."""
    import os
    base = os.path.join(out_dir, name)
    fig.savefig(base + '.pdf', metadata={'CreationDate': None})
    fig.savefig(base + '.svg', metadata={'Date': None})
    fig.savefig(base + '.png')
    return base


PAL = {'Speech AI': '#2c6fbb', 'Vision AI': '#c0392b'}
PAL_SUB = {'Static image (OCR / figure)': '#e08a1e', 'Dynamic action (pose)': '#c0392b'}
