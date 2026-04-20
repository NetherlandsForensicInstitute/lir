Terminology
===========

- **lr system**: An algorithm to calculate likelihood ratios.
- **lr system architecture**: A way to compose an LR system, e.g. feature based, specific source, etc.
- **source**: Something that can generate instances, or where instances are derived from. This is typically the level
  relevant to the forensic question and the hypotheses. Examples**: a glass pane, a person whose face may be pictured, a
  speaker of the voice, a shoe.
- **instance**: A single manifestation of a source. Traces and reference samples are instances of a source. In a feature
  based system, instances are used as the building blocks for modeling hypotheses. Examples**: the measurements on a
  fragment of glass, a face image, a voice recording, a shoe print.
- **pair**: A combination of two groups of instances. The instance groups may be same source or different source. In a
  common-source system, pairs are used as the building blocks for modeling hypotheses. A group may contain only one
  instance, or it may consist of multiple repeated measurements of the same source that are compared as one unit.
- **data set**: A set of instances and/or pairs, labeled or unlabeled for source or for hypothesis, that may be used for calculating likelihood ratios.
- **label**: The ground-truth value for an instance or a pair. The label may be on the level of the hypothesis (e.g. H1,
  H2), or on the level of the source (e.g. Speaker1, Speaker2). Hypothesis labels may derived from source labels.
  In case of a labeled data set, the ground truth (i.e. labels) is known. This will typically be the data that is used
  for development, analysis or validation of an LR system. Unlabeled data has no ground truth. This will typically be
  the application data, or case data in a forensic setting.
- **binary data**: A data set with exactly two different labels, for specific source evaluation (e.g. H1, H2).
- **multiclass data**: A data set with an arbitrary number of labels, for common source evaluation or for reduction to
  binary data for specific source evaluation.
- **data strategy**: The way in which the data are assigned to different applications (validation, calibration, etc.) within the lr system. Well known strategies are train/test split, cross-validation, leave-one-out.
- **data provider**: A method for making a data set available, e.g. by reading from disk and processing it.
- **run**: Calculations for an lr system on a specific data set, as part of an experiment.
- **experiment**: A series of one or more runs to calculate lrs, to measure system performance, to evaluate the effect of varying system parameters, or to optimize system parameters.
- **experiment strategy**: A strategy for specifying system parameter values, e.g. single run, grid search, etc.
