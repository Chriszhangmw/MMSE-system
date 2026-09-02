#!/usr/bin/env python3
"""Reproduce the full AI-MMSE figure set.

Usage
-----
    python code/run_all.py

Reads  data/analysis_dataset.csv  (de-identified, already cleaned)
Writes data/*.csv, data/summary.json   -- intermediate statistics
       figures/Fig1..Fig6 .pdf / .png  -- final figures

To rebuild the analysis dataset from the raw clinical workbook instead:
    AIMMSE_RAW=/path/to/机器数据更新_2024_12_23_.xlsx python code/run_all.py --from-raw
"""
import os, sys, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--from-raw', action='store_true',
                help='rebuild data/analysis_dataset.csv from the raw workbook')
args = ap.parse_args()

if args.from_raw:
    import prep
    print('[0/7] rebuilding analysis dataset from raw workbook ...')
    mg = prep.load_raw()
    out = os.path.join(ROOT, 'data', 'analysis_dataset.csv')
    mg.to_csv(out, index=False)
    print('      wrote %s  (n = %d)' % (out, len(mg)))

steps = ['stats_core.py', 'robustness.py'] + ['fig%d.py' % i for i in range(1, 6)] + ['figS1.py', 'figS2.py']
for i, s in enumerate(steps, 1):
    print('[%d/%d] %s' % (i, len(steps), s))
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        sys.exit('FAILED at %s' % s)

print('\nDone. Vector figures (Type 42 fonts, editable in Illustrator):')
figs = sorted(f for f in os.listdir(os.path.join(ROOT, 'figures')) if f.endswith('.pdf'))
for f in figs:
    print('   figures/' + f)
