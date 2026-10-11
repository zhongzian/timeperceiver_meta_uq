# Gate 5b Tail Diagnostic

Status: **PASS**

```json
{
  "status": "PASS",
  "rows": 15,
  "datasets": {
    "etth1": {
      "status": "PASS",
      "quantile_sample_count": 1871520,
      "total_element_count": 1871520,
      "consistency": {
        "element_count_complete": true,
        "gaussian_nll_matches_formal": true,
        "fixed_t8_nll_matches_formal": true
      }
    },
    "weather": {
      "status": "PASS",
      "quantile_sample_count": 1914101,
      "total_element_count": 21055104,
      "consistency": {
        "element_count_complete": true,
        "gaussian_nll_matches_formal": true,
        "fixed_t8_nll_matches_formal": true
      }
    },
    "electricity": {
      "status": "PASS",
      "quantile_sample_count": 1989558,
      "total_element_count": 159164640,
      "consistency": {
        "element_count_complete": true,
        "gaussian_nll_matches_formal": true,
        "fixed_t8_nll_matches_formal": true
      }
    }
  },
  "evaluation_only": true,
  "training": false,
  "backward": false,
  "descriptive_only": true,
  "formal_gate_evidence": false,
  "statistical_significance_claim": false
}
```
