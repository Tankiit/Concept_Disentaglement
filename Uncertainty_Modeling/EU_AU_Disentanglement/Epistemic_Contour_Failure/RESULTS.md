# Cross-experiment synthesis record

## 24 October 2026 — narrative research report

Created [RESEARCH_REPORT.md](RESEARCH_REPORT.md) at the project root in response to the request for an integrated account of experiments 01–10. The report states the main claim, working hypothesis, and objective; explains the experimental progression; and includes four existing figures and four numbered results tables. Limitations are collected in a dedicated section.

### Synthesis decisions and their evidence

- Organize the narrative around **identification → uncertainty target → frequentist evaluation → implemented scores → sensitivity**. This order connects the analytic obstruction in 01/05 to the exact target diagnostic in 02 and the observed behavior in 06/08–10.
- State the central claim for selected measures and information regimes. The supporting evidence comprises exact ambiguity and coverage arguments, an exact frequentist counterexample, and numerical studies. These types of evidence are identified explicitly in the report.
- Keep the main framework frequentist. Use experiments 03–04 to explain the influence of source assumptions and to carry forward their compatible-family and resolution findings.
- Preserve the successful controls: contour-membership MI retains ambiguity in experiment 02, and the contamination-aware reference retains coverage under its declared assumptions. Include the large-spread outcomes in experiment 10 and the clean-data tail-island example.
- Distinguish empirical inclusion frequencies for defined confidence sets from the descriptive neural member radius. Keep MI in bits for experiment 02 and nats for the probly studies. Separate contour probability mass from confidence level.
- Present the high-sample sensitivity range 0.348–0.376 as a range of mean errors across configurations and the two fixed clean truths. The same 20 datasets are reused across experiment 10 configurations; the worlds are evaluated on paired data.
- Treat within-cell geometric sensitivity as an interpretation of the observed tail-island behavior. A partition-resolution study remains the next discriminating experiment.

This synthesis introduces no new experimental runs or theoretical proofs. The reasoning above consolidates decisions and findings already recorded in the experiment reports.

### Sources consulted for the report

Read the project proposal, research requirements, foundational and frequentist experiment reports, probly adapter findings, and neural sensitivity reports/results. Checked experiment 10's summary CSV and diagnostic JSON. Inspected the existing foundational, MI-map, and full-cell comparison figures; the neural tail-island figure had also been inspected during experiment 10.

Rechecked the primary papers by Chen, Gao and Ren; Houlsby and colleagues; Wimmer and colleagues; and Lakshminarayanan, Pritzel and Blundell. Reopened the Clopper–Pearson scan. The attempt to reopen Efron's paper timed out, so its quotation and consultation scope are explicitly carried forward from experiment 06. Full paper titles, author/year information, links, relevant sections, and short quotations at the points of use appear in the report's narrative and reference table. Historical posterior-literature attribution remains linked to experiments 03–05.

### Verification

Checked all local report links and all four image paths. Verified the principal numerical tables against the experiment reports and saved sensitivity summary. Reviewed narrative wording to keep contrastive qualifications in the limitations section. Original experiment artifacts were preserved.
