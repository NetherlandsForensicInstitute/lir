Assessing LR system performance
===============================

This page shows how LiR can be used to assess the `performance`_ of an LR system, in particular discriminatory power
and `consistency`_. These metrics are typically used to judge whether an LR system is good enough for use in casework,
or to find the best LR system among several candidates.

**Note:** in literature, the term 'calibration' is sometimes used to refer to whether an LR system is 'well-calibrated'
or 'consistent'. Here, we reserve `calibration` for the process, while we use 'consistency' for the quality of the LLRs.


Metrics
-------

A `widely used`_ metric of performance is Cllr, which measures both discrimination and consistency.
Other metrics may have practical use as well. See the table below for a list of metrics.

.. _performance: https://doi.org/10.1016/j.forsciint.2016.03.048
.. _widely used: https://doi.org/10.1016/j.fsisyn.2024.100466
.. _consistency: https://doi.org/10.1016/j.forsciint.2021.110722

+-------------------------------------+--------------------------------+
| Metric                              | Assessment of                  |
+=====================================+================================+
| log likelihood ratio cost (`cllr`_) | discrimination and consistency |
+-------------------------------------+--------------------------------+
| minimized cllr (`cllr_min`_)        | discrimination                 |
+-------------------------------------+--------------------------------+
| calibration loss (`cllr_cal`_)      | consistency                    |
+-------------------------------------+--------------------------------+
| rate of misleading evidence         | mostly consistency             |
+-------------------------------------+--------------------------------+
| `devPAV`_                           | consistency                    |
+-------------------------------------+--------------------------------+
| expected LR for both hypotheses     | discrimination                 |
+-------------------------------------+--------------------------------+

By definition, the log likelihood ratio cost ``cllr`` equals ``cllr_min`` + ``cllr_cal``.

.. _cllr: api/lir.metrics.html#lir.metrics.cllr
.. _cllr_min: api/lir.metrics.html#lir.metrics.cllr_min
.. _cllr_cal: api/lir.metrics.html#lir.metrics.cllr_cal
.. _devPAV: api/lir.metrics.html#lir.metrics.devpav.devpav


Visualizations
--------------

While a one-dimensional metric is often useful, visualizations give more insight in the behavior of the LR system.
Examples of visualizations are:

- `LR histogram`_, for assessment of discrimination
- Pool adjacent violators (`PAV`_) transformation, for assessment of consistency
- Empirical cross-entropy (`ECE`_)
- `Tippett`_

.. _LR histogram: api/lir.plotting.html#lir.plotting.lr_histogram
.. _PAV: api/lir.plotting.html#lir.plotting.pav
.. _ECE: api/lir.plotting.html#lir.plotting.expected_calibration_error.plot_ece
.. _Tippett: api/lir.plotting.html#lir.plotting.tippett


Let's see how these metrics and visualizations behave on different types of data.


Neutral LLRs
------------

First, non-informative data, where all LLRs are zero (i.e. neutral). These data are not discriminative, but perfectly
consistent!

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from lir.data.models import LLRData
    from lir.metrics import cllr, cllr_min, cllr_cal
    from lir.metrics.devpav import devpav
    from lir.plotting import lr_histogram, pav, tippett
    from lir.plotting.expected_calibration_error import plot_ece

    plt.rcParams.update({'font.size': 9})

    def llr_metrics_and_visualizations(llrs: LLRData):
        # print the metrics
        print(f'cllr: {cllr(llrs)}')
        print(f'cllr_min: {cllr_min(llrs)}')
        print(f'cllr_cal: {cllr_cal(llrs)}')
        print(f'devpav: {devpav(llrs)}')

        # initialize the plot
        fig, ((ax_lrhist, ax_pav), (ax_ece, ax_tippett)) = plt.subplots(2, 2)
        fig.set_figwidth(10)

        # create the visualizations
        lr_histogram(ax_lrhist, llrs)
        pav(ax_pav, llrs)
        plot_ece(ax_ece, llrs)
        tippett(ax_tippett, llrs)

        # generate the image
        fig.tight_layout()
        fig.show()

    # generate neutral LLRs
    llrs = LLRData(features=np.zeros((6, 1)), labels=np.array([0, 0, 0, 1, 1, 1]))

    # show results
    print('results for neutral (non-informative) LLR values')
    llr_metrics_and_visualizations(llrs)

Observe that:

- the value for ``cllr`` is 1;
- the value for ``cllr_min`` is 1;
- the value for ``cllr_cal`` is 0;
- the LR histogram shows a single bar;
- in the ECE plot, the LRs line, the reference line and the PAV LRs line are the equivalent;
- the PAV plot and the Tippett plot hardly make sense if all LLRs have the same value.


Well-calibrated LLRs
--------------------

Now, we have LLRs that are both discriminative and consistent, and data of both hypotheses are drawn from a normal
distribution. It visualizes as follows.

.. jupyter-execute::

    from lir.algorithms.logistic_regression import LogitCalibrator
    from lir.data.data_strategies import BinaryTrainTestSplit
    from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalData, SynthesizedNormalBinaryData

    # set the parameters for H1 data and H2 data
    h1_data = SynthesizedNormalData(mean=1, std=1, size=1000)
    h2_data = SynthesizedNormalData(mean=-1, std=1, size=1000)

    # generate the data
    instances = SynthesizedNormalBinaryData(h1_data, h2_data, seed=42).get_instances()

    # split the data into a 50% training set and a 50% test set
    training_instances, test_instances = next(BinaryTrainTestSplit(test_size=.5).apply(instances))

    # build a simple LR system for these data
    calibrator = LogitCalibrator()

    # train the system on the training set, and calculate the LLRs for the test set
    llrs = calibrator.fit(training_instances).apply(test_instances)

    # assess performance
    print('results for well-calibrated LLR values')
    llr_metrics_and_visualizations(llrs)


Observe that, for discriminative and well-calibrated LLRs:

- the value for ``cllr`` is lower than 1;
- the value for ``cllr_min`` is close to ``cllr``;
- the value for ``cllr_cal`` is close to 0;
- the LR histogram shows distinct distributions;
- in the LR histogram, the peak of the overlap of both distributions is at 0;
- the PAV plot approximately follows the diagonal;
- in the ECE plot, the LRs line is close to the PAV-LRs line, and the reference line is wel above both of them.


Badly calibrated data
---------------------

LR systems may misbehave in several ways, resulting in inconsistent LLRs.
If this happens, check if the the training data is suitable for the test data. Inconsistent LLRs can be caused, for
example, when the training data are measurements of a different type of glass, when training data are from voice
recordings of a microphone versus telephone interception in the test data, or any other kind of mismatch between the
training set and the test set.

LLRs may be inconsistent in several ways, including:

- bias towards H1, meaning that the LLRs are too big;
- bias towards H2, meaning that the LLRs are too small;
- overestimation, meaning that the LLRs are too extreme;
- underestimation, meaning that the LLRs are too close to 0.

Below are the results of each of such inconsistent sets of LLRs.
Let's have a look at the metrics and visualizations for each of those.


.. jupyter-execute::

    print('all LLR values are *shifted* towards H1')
    biased_llrs_towards_h1 = llrs.replace(features=llrs.features + 2)
    llr_metrics_and_visualizations(biased_llrs_towards_h1)


Observations for LLRs that are biased towards H1:

- the value for ``cllr`` is increased from well-calibrated data;
- the value for ``cllr_min`` is the same as in well-calibrated data;
- the value for ``cllr_cal`` is greater than 0;
- the LR histogram still shows distinct distributions, but they are shifted to the **right**;
- in the LR histogram, the peak of the overlap of both distributions is to the right of 0;
- the PAV plot is **below** the diagonal;
- in the ECE plot, the LRs line is evidently above the PAV-LRs line and closer to the reference line (if the LLRs are
  wildly biased, the LRs line may even be partially above the reference line);
- the Tippett plot is shifted to the left.


.. jupyter-execute::

    print('all LLR values are *shifted* towards H2')
    biased_llrs_towards_h2 = llrs.replace(features=llrs.features - 2)
    llr_metrics_and_visualizations(biased_llrs_towards_h2)


Observations for LLRs that are biased towards H2:

- the value for ``cllr``, ``cllr_min`` and ``cllr_cal`` are similar to data biased towards H1;
- the LR histogram still shows distinct distributions, but they are shifted to the **left**;
- in the LR histogram, the peak of the overlap of both distributions is to the right of 0;
- the PAV plot is **above** the diagonal;
- in the ECE plot, the LRs line is evidently above the PAV-LRs line, similar to the LLRs shifted towards H1;
- the Tippett plot is shifted to the right.


.. jupyter-execute::

    print('the LLRs are *increased* towards the extremes on both sides')
    overestimated_llrs = llrs.replace(features=llrs.features * 2)
    llr_metrics_and_visualizations(overestimated_llrs)


Observations for overestimated LLRs:

- the value for ``cllr``, ``cllr_min`` and ``cllr_cal`` are similar to biased data;
- the LR histogram still shows distinct distributions, but the scale on the X-axis is increased;
- the PAV plot is flatter than the diagonal, and crosses it near the origin;
- in the ECE plot, the LRs line is slightly further away from the PAV-LRs line, and may cross the reference line near
  the extremes;
- the scale of the Tippett plot is increased.


.. jupyter-execute::

    print('the LLRs are *reduced* towards neutrality')
    underestimated_llrs = llrs.replace(features=llrs.features / 2)
    llr_metrics_and_visualizations(underestimated_llrs)


Observations for overestimated LLRs:

- the value for ``cllr``, ``cllr_min`` and ``cllr_cal`` are similar to biased data;
- the LR histogram still shows distinct distributions, but the scale on the X-axis is increased;
- the PAV plot is steeper than the diagonal, and crosses it near the origin;
- in the ECE plot, the LRs line is slightly further away from the PAV-LRs line;
- the scale of the Tippett plot is decreased.


That's all for now!
