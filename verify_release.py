#!/usr/bin/env python3
"""Lightweight read-only release validation (not a full scientific replication)."""
from pathlib import Path
import csv, json, hashlib, sys

ROOT=Path(__file__).resolve().parents[1]
SIM=ROOT/'data/simulated/summary'
MARKET=ROOT/'data/market/summary'

def csv_records(name, where=SIM):
    p=where/name
    if not p.is_file():raise AssertionError('Missing '+str(p.relative_to(ROOT)))
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    for p in [ROOT/'paper/HMM6_Integrated_Referee_Draft.pdf',
              ROOT/'paper/HMM6_Integrated_Referee_Draft.tex',
              ROOT/'code/simulation/01_generate_and_recover_hmm6.py',
              ROOT/'code/simulation/02_blind_econometric_competitors.py',
              ROOT/'code/theorem/GT_Proposition_A_Targeted_v3.py']:
        assert p.is_file() and p.stat().st_size>100,str(p)
    manifest=csv_records('DGP_MANIFEST.csv')
    recover=csv_records('HMM6_PARAMETER_RECOVERY_ALL.csv')
    pairing=csv_records('PATH_MATCHED_HMM6_VS_19_MODELS.csv')
    rank=csv_records('MODEL_RANKING_WITH_MC_SE.csv')
    flags=csv_records('EGARCH_AND_OTHER_NUMERICAL_PENALTY_PATHS.csv')
    m_ids=[r['id'] for r in manifest]
    r_ids=[r['id'] for r in recover]
    assert len(m_ids)==4000 and len(set(m_ids))==4000,'Manifest expected 4,000 unique paths'
    assert len(r_ids)==4000 and set(r_ids)==set(m_ids),'HMM records mismatch'
    assert set(r['n'] for r in manifest)=={'6000'},'Production sample length changed'
    m_scen=set(r['scenario'] for r in manifest)
    assert len(m_scen)==8 and all(sum(r['scenario']==s for r in manifest)==500 for s in m_scen),'Expected eight 500-path DGPs'
    assert len(rank)==152 and len(set((r['scenario'],r['model']) for r in rank))==152,'19-model aggregate coverage mismatch'
    assert len(pairing)==152 and len(set((r['scenario'],r['model']) for r in pairing))==152,'Paired aggregate coverage mismatch'
    assert len(set(r['model'] for r in pairing))==19,'Expected 19 models'
    assert len(flags)==11,'Expected 11 reported EGARCH penalty paths'
    assert all(r['model']=='EGARCH11-t' for r in flags),'Unexpected model in numerical flag list'
    density=csv_records('GT_GARCH_HMM_All_8_Verified_Log_Scores.csv',MARKET)
    var=csv_records('GT_GARCH_HMM_All_8_Verified_VaR.csv',MARKET)
    assert len(density)==8 and len(var)==24,'Eight-cell intraday coverage mismatch'
    assert sum(row['best_log_score_model']=='garch_hmm' for row in density)==5,'Observed market score ranking changed'
    with (SIM/'study_design_PUBLIC.json').open(encoding='utf-8') as f:design=json.load(f)
    assert design['command_line_options']['reps']==500 and design['command_line_options']['n']==6000
    print('PASS: 4,000 unique paths; 8 DGPs; 19 competitors; 152 paired/model aggregates; 11 flagged numerical cases.')
    print('PASS: 8 intraday score cells; 24 VaR cells; 5/8 descriptive HMM wins.')
    print('NOTE: This is a release-integrity check, not independent verification of econometric likelihoods.')
    return 0

if __name__=='__main__':sys.exit(main())
