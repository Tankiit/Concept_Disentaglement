### Sources, dataset choice and scope

- [UCI Adult official documentation](https://archive.ics.uci.edu/dataset/2/adult) and [download](https://archive.ics.uci.edu/static/public/2/adult.zip): recorded census-derived variables, official files, missing-value conventions and provenance. The archive hash is saved.
- [WILDS dataset documentation](https://wilds.stanford.edu/datasets/): considered for later image experiments with actual domain metadata. No WILDS data are used here.
- Hüllermeier & Waegeman, [uncertainty concepts](https://doi.org/10.1007/s10994-021-05946-3): predictive epistemic MI and uncertainty over probabilities.
- [Probly](https://github.com/pwhofman/probly): independently check the MI representation on posterior samples for the selected pair.
- Local `acquisition_counterexample.ipynb`: the exact toy proof being tested observationally; `real/beta_acquisition.py`: equations implemented below.

The four cached image datasets contain images and one class label, but no separate writer/hospital/domain identity. Manufacturing such a field would preserve a synthetic mechanism. Adult has distinct, co-recorded sector and income fields. Sector is a **population stratum**, not acquisition provenance or a causal treatment. We make no causal claims or recommendations about individuals.

This pilot models only sector and income. Other census features are deliberately excluded, so it does not establish that unlabeled features cannot help. It does not train or evaluate DAPPr. The claim under test is the acquisition sufficiency of predictive epistemic summaries, not a defect in a specific neural model.