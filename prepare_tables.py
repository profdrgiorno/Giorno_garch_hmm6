#!/usr/bin/env python3
"""Render published CSV aggregates into Markdown without re-estimation."""
from pathlib import Path
import argparse, csv
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default=str(ROOT/'results'/'RELEASE_TABLES.md'))
    a=ap.parse_args()
    aout=Path(a.out);aout.parent.mkdir(parents=True,exist_ok=True)
    recovery=rows(ROOT/'data/simulated/summary/SCENARIO_RECOVERY_AUDIT.csv')
    paired=rows(ROOT/'data/simulated/summary/PATH_MATCHED_HMM6_VS_19_MODELS.csv')
    market=rows(ROOT/'data/market/summary/GT_GARCH_HMM_All_8_Verified_Log_Scores.csv')
    s=['# Archived tables (descriptive; 29 October 2026)','',
       '## Identification: latent-state persistence','',
       '| Scenario | True ρ | Estimated mean ρ | State accuracy | Majority baseline |','|:--|--:|--:|--:|--:|']
    def ff(v):
        try: return f'{float(v):.4f}'
        except (TypeError,ValueError):return 'n/a'
    for r in recovery:s.append('| '+ ' | '.join([r['scenario'],ff(r['true_rho']),ff(r['estimated_rho_mean']),ff(r['state_accuracy_mean']),ff(r['constant_state_majority_accuracy'])])+' |')
    s.extend(['','## HMM minus APARCH (symmetric-power): paired OOS mean log scores','','| Scenario | Mean diff. | 95% MC bootstrap CI | Valid pairs |','|:--|--:|:--|--:|'])
    for r in paired:
        if r['model']=='APARCH11-power-symmetric-t':
            s.append(f"| {r['scenario']} | {ff(r['mean_hmm_minus_competitor'])} | [{ff(r['bootstrap_ci95_lo'])}, {ff(r['bootstrap_ci95_hi'])}] | {r['valid_paired_paths']} |")
    s.extend(['','## Market holdouts — descriptive log score','','| Market | TF | Best | HMM−GARCH | HMM−IID mixture |','|:--|:--|:--|--:|--:|'])
    for r in market:s.append(f"| {r['market']} | {r['timeframe']} | {r['best_log_score_model']} | {ff(r['hmm_minus_t'])} | {ff(r['hmm_minus_mix'])} |")
    s.extend(['','Do not read numerical EGARCH penalty scores as genuine predictive densities. Source data: archived CSVs in `data/`.',''])
    aout.write_text('\n'.join(s),encoding='utf-8')
    print('Wrote',aout)
if __name__=='__main__':main()
