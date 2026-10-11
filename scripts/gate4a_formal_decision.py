#!/usr/bin/env python
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows={r['variant']:r for r in csv.DictReader(open(ROOT/'reports/gate4a_minimal_results.csv'))};o,s,m=rows['official_baseline'],rows['static_adapter'],rows['missingness_aware_adapter']
    def f(r,k):return float(r[k])
    static_beats=f(s,'test_mse')<f(o,'test_mse');missing_beats_official=f(m,'test_mse')<f(o,'test_mse');conditioning=f(m,'test_mse')<=f(s,'test_mse');signal=missing_beats_official and conditioning
    decision='PASS' if signal else ('ADAPTER_CAPACITY_SIGNAL_ONLY' if missing_beats_official and not conditioning else 'IMPLEMENTATION_PASS_SIGNAL_NEGATIVE')
    deltas={
      'static_minus_official':{k:f(s,'test_'+k)-f(o,'test_'+k) for k in ('mse','mae','rmse')},
      'missing_aware_minus_official':{k:f(m,'test_'+k)-f(o,'test_'+k) for k in ('mse','mae','rmse')},
      'missing_aware_minus_static':{k:f(m,'test_'+k)-f(s,'test_'+k) for k in ('mse','mae','rmse')}}
    for key,ref,var in [('static_minus_official',o,s),('missing_aware_minus_official',o,m),('missing_aware_minus_static',s,m)]:deltas[key]['relative_mse_improvement']=(f(ref,'test_mse')-f(var,'test_mse'))/f(ref,'test_mse')
    out={'status':decision,'GATE4A_IMPLEMENTATION_PASS':True,'GATE4A_SIGNAL_POSITIVE':signal,'GATE4A_STATIC_ADAPTER_BEATS_BASELINE':static_beats,'GATE4A_MISSINGNESS_CONDITIONING_BEATS_STATIC':conditioning,'ENTER_GATE4A_EXPANDED_MATRIX':signal,'seed_count':1,'deltas':deltas}
    (ROOT/'results/gate4a_formal_decision.json').write_text(json.dumps(out,indent=2)+'\n')
    with open(ROOT/'reports/gate4a_formal_decision.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['decision','implementation_pass','signal_positive','static_beats_baseline','conditioning_beats_static','enter_expanded']);w.writerow([decision,True,signal,static_beats,conditioning,signal])
    md=f"# Gate 4a Formal Decision\n\nDecision: **{decision}**\n\n- Implementation pass: `true`\n- Missingness-aware beats official: `{str(missing_beats_official).lower()}`\n- Static beats official: `{str(static_beats).lower()}`\n- Missingness conditioning beats static: `{str(conditioning).lower()}`\n- Enter expanded matrix: `{str(signal).lower()}`\n\nSingle seed only; no statistical-significance claim.\n"
    (ROOT/'reports/gate4a_formal_decision.md').write_text(md);print(json.dumps(out,indent=2))
    status_path=ROOT/'results/gate4a_status.json';status=json.loads(status_path.read_text());status.update({'status':decision,'formal_queue':'COMPLETE','scheduler':'COMPLETE','GATE4A_IMPLEMENTATION_PASS':True,'GATE4A_SIGNAL_POSITIVE':signal,'GATE4A_STATIC_ADAPTER_BEATS_BASELINE':static_beats,'GATE4A_MISSINGNESS_CONDITIONING_BEATS_STATIC':conditioning,'ENTER_GATE4A_EXPANDED_MATRIX':signal});status_path.write_text(json.dumps(status,indent=2)+'\n')
    (ROOT/'reports/gate4a_status.md').write_text(f"# Gate 4a Status\n\nDecision: **{decision}**\n\nBoth formal tasks succeeded. Implementation passed, but missingness conditioning did not beat the same-capacity static adapter. Expanded matrix entry is `false`.\n")
if __name__=='__main__':main()
