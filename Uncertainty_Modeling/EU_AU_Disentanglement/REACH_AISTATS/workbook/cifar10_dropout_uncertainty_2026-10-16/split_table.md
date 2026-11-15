# CIFAR-10 split table

Seed: 0. Counts refer to images from the official CIFAR-10 test set, repartitioned for this experiment.

| Class ID | Class | Group | Train | Validation | Test |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | airplane | KKC | 700 | 200 | 100 |
| 1 | automobile | KKC | 700 | 200 | 100 |
| 2 | bird | UUC | 0 | 0 | 1000 |
| 3 | cat | UUC | 0 | 0 | 1000 |
| 4 | deer | UUC | 0 | 1000 | 0 |
| 5 | dog | UUC | 0 | 0 | 1000 |
| 6 | frog | UUC | 0 | 1000 | 0 |
| 7 | horse | UUC | 0 | 0 | 1000 |
| 8 | ship | KKC | 700 | 200 | 100 |
| 9 | truck | UUC | 0 | 0 | 1000 |
| **Total** | | | **2100** | **2600** | **5300** |

KKC = known classes. UUC = classes withheld from head training. No KUC classes are used.

`split_examples.csv` lists all 10,000 images by their original cache/Parquet row index, class, split, and group. Validation/test rows also contain saved predictions and uncertainty scores; those fields are blank for training images, which were not scored. Predictions use original class IDs.

`split_summary.csv` contains the aggregate table above.
