# Probly probability-bound comparison

Two actual probability-bound representations, plus a DAPPr-loss baseline; one-seed small-CNN pilot.
Source inference has candidate-reference access and uses the unlabeled test histogram. This is not an equal-information comparison.

| Dataset | MC width AUROC | Source width AUROC | DAPPr EU AUROC |
|---|---:|---:|---:|
| mnist | 0.9306 | 0.5000 | 0.8748 |
| emnist | 0.7909 | 0.5000 | 0.7467 |
| fashion | 0.7916 | 0.5000 | 0.7299 |
| blood | 0.4486 | 0.5000 | 0.4480 |

Maximum Probly raw vs tie-correct AUROC discrepancy: 0.500000.
Source class-event rows using undefined-denominator [0,1] policy: 43516.
Probability endpoint vectors need not sum to one. Their width is imprecision, not MI. Candidate membership is a separate event.
No true conditional probabilities are observed for real images; empirical interval coverage is not claimed.
Read SOURCES.md and the executed notebook for the full assumptions and derivations.

### Post-run interpretation of the source-mixture result

In this saved run, the maximum known-class probability width is exactly **1 for every test image in all four datasets**. Its tie-correct AUROC is therefore **0.5**, not evidence of successful OOD separation. The dropout envelope and DAPPr baseline also perform below chance on BloodMNIST in this one-seed run; no general improvement claim is supported.

The source implementation returns 43,516 class-event rows with the explicit undefined-denominator policy. In EMNIST and BloodMNIST, the candidate-only law itself lies inside the deployment marginal box: an all-candidate mixture is compatible, so known-conditional membership can be undefined. In MNIST and Fashion, candidate-only mixtures are excluded globally, but compatible models can still assign zero known-source mass to individual probe cells. Other extrema can also be vacuous even when their denominator is positive.

**Fundamental:** an interval representation does not create information. Sparse reference data, a coarse observation map, conservative coordinate boxes and conditioning on possibly absent known membership can make the requested probability unidentified or uninformative. A subsequent experiment can change these information/model choices, but should not silently tighten bounds to improve AUROC.

This audit was computed after execution from the saved constraints and score arrays. See `source_informativeness_audit.csv`. It does not use test labels to modify the inference. Candidate references and transductive inputs remain an information advantage of the source branch; its failure here does not establish that every source-mixture implementation fails.
