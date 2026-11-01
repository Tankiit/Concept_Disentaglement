9## Key Outcomes

- **Research direction:** The team will focus on analyzing the **blind spot of epistemic uncertainty** using cached features and the **PyTorch OOD** library to create a stronger story for **CVPR** and **ICML**. 
- **Technical approach:** Mateus will extend the current uncertainty quantification to include **epistemic uncertainty** as a set (upper and lower bounds) rather than a single score. 
- **Paper strategy:** The CVPR story will emphasize a simple **cradle adapter** as a post-hoc trainer to improve confidence/calibration, while the ICML story may pivot toward **likelihood-based** or **decalibration-based** aspects. 
- **Logistics:** Mateus is likely to visit in **February** (not January) and may stay with Mukherjee in **Brussels** to facilitate intense work. 

## Decisions Made

- **Dataset/Library:** Use **PyTorch OOD** (specifically `PyTorchODDataset.osim`) for open-set simulation, including known-unknown and unknown-unknown conditions. 
- **Evaluation setup:** Stick with **ID/OOD data** for now, but extend the analysis to include **epistemic uncertainty** quantification. 
- **Compute:** Use **cached features** to avoid heavy GPU requirements; Mateus has two GPUs available if needed. 
- **Paper positioning:** The CVPR paper should position the **cradle adapter** as a simple, promising alternative to complex multiple-adapter approaches. 

## Technical Requirements

- **Epistemic uncertainty representation:** The output should be a **set** (upper and lower bounds) rather than a single score. 
- **Scoring methods:** Potential ways to derive a score from the interval include using the **median**, **average**, or passing it through a **multi-layer perceptron** or **logistic regression for interval-valued sets**. 
- **Code implementation:** Extend the `RepresentationBuilder` base class to create an **epistemic set representer** that outputs a list/set of contours. 
- **Data handling:** Use a **tabular format** (e.g., pandas CSV) to store and import values for comparison across methods. 

## Pending Confirmation

- **Cradle instance inclusion:** Mukherjee is unsure whether to include the **cradle instance** in the CVPR story, as it might weaken the ICML narrative. 
- **Visit dates:** Mateus needs to discuss the **February** visit with his partner before finalizing. 
- **Grant status:** Mateus believes the chance of getting the grant is high, but it is not yet confirmed. 

## Open Questions

- **Score derivation:** How to best convert the epistemic uncertainty interval into a usable score (median, average, or learned model). 
- **Competitor comparison:** Which specific competitor methods to include in the tabular comparison. 
- **ICML focus:** Whether to pivot the ICML story more toward **likelihood-based** or **decalibration-based** aspects. 

## Action Items

- **Mateus:** Replicate the last paper discussed, applying **cradle/epistemic sets** and testing on **ID/OOD** conditions using **PyTorch OOD**. 
- **Mateus:** Send Mukherjee his **bandwidth** details regarding datasets and compute availability today. 
- **Mukherjee:** Send Mateus the **competitor papers/links** and a **concrete code outline** by the end of the day. 
- **Mukherjee:** Check with **Ziad** regarding the logistics for Mateus's visit. 
- **Mateus:** Review the `make_OSS` code and correct the **MNIST class assignments** (known: 1, 2, 4, 6, 8; unknown: 0, 3, 5, 7, 9). 
