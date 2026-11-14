# Data and results

The [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) hosts the original training and test releases. The included download scripts retrieve the corresponding releases from the official CREATIS server and save the source terms. The repository contains research code and aggregate outcomes. Obtain images and masks from the original source for non-commercial scientific research under the [dataset terms](https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/item/66e290e0961576b1bad4ee3c/download) and [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Cite O. Bernard et al., “Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?”, IEEE Transactions on Medical Imaging, 37(11), 2514–2525, 2018. [Original article](https://doi.org/10.1109/TMI.2018.2837502)

`results/annotation-evidence-aggregate.json` holds patient-aggregated outcome estimates and resampling intervals. `results/annotation-target-audit.json` counts whether selected slice–class nominations matched their reference masks by class, phase and state. The corresponding scripts reconstruct summaries from local evaluation output. The audit estimates annotation effort in masks rather than clinician time.
