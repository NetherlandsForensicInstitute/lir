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
.. _devPAV: api/lir.metrics.html#lir.algorithms.devpav.devpav


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

The `PAV transformation`_ is particularly useful to inspect a system (or a set of LLRs, actually) for consistency. It
optimizes a set of LLRs (for which the ground truth is known) for consistency without changing the order of the LLRs.
The corresponding visualization shows a scatter plot of the original, "pre-calibrated" LLRs versus the "post-calibrated"
LLRs, after transformation.

.. _PAV transformation: https://en.wikipedia.org/wiki/Isotonic_regression

How to read a PAV plot? The example below shows how to interpret the different sections of the plot.

.. jupyter-execute::
    :hide-code:

    from matplotlib.patches import Polygon
    from sklearn.preprocessing import StandardScaler

    from lir import plotting
    from lir.algorithms.bayeserror import ELUBBounder
    from lir.algorithms.logistic_regression import LogitCalibrator
    from lir.datasets.glass import GlassData
    from lir.data_strategies import PredefinedTrainTestSplit
    from lir.lrsystems.score_based import ScoreBasedSystem
    from lir.transform import as_transformer
    from lir.transform.distance import ManhattanDistance
    from lir.transform.pairing import SourcePairing
    from lir.transform.pipeline import Pipeline

    instances = GlassData(cache_dir='glass-data').get_instances()
    train, test = next(PredefinedTrainTestSplit().apply(instances))

    scoring = Pipeline(steps=[('diff', ManhattanDistance()), ('calib', LogitCalibrator(random_state=0)), ('elub', ELUBBounder())])
    lrsystem = ScoreBasedSystem(preprocessing_pipeline=as_transformer(StandardScaler()), evaluation_pipeline=scoring, pairing_function=SourcePairing(ratio_limit=1, seed=0))

    lrsystem.fit(train)
    llrs = lrsystem.apply(test)

    with plotting.show() as ax:
        ax.pav(llrs)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        poly = Polygon([(max(x0, y0), max(x0, y0)), (min(x1, y1), min(x1, y1)), (x0, y1)], facecolor='orange', edgecolor='0.5', alpha=.2)
        ax.add_patch(poly)
        ax.text(x0*.5, y1*.25, "bias towards H2", horizontalalignment='center', fontsize=14)

        poly = Polygon([(max(x0, y0), max(x0, y0)), (min(x1, y1), min(x1, y1)), (x1, y0)], facecolor='blue', edgecolor='0.5', alpha=.2)
        ax.add_patch(poly)
        ax.text(x1*.5, y0*.5, "bias towards H1", horizontalalignment='center', fontsize=14)

- LLRs that are **on the diagonal** remained unchanged and appear to be well-calibrated. Ideally, all LLRs are somewhere
  near the diagonal. In that case, the calibration loss will be close to 0.
- LLRs that appear **above the diagonal** are increased after optimization, and the original LLRs were therefore
  **biased towards H2**.
- LLRs thet appear **below the diagonal** are decreased after optimization, and the original LLRs were therefore
  **biased towards H1**.

Another way to look at it, is to distinguish between overestimated and underestimated LLRs.

.. jupyter-execute::
    :hide-code:

    with plotting.show() as ax:
        ax.pav(llrs)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        #poly = Polygon([(0, 0), (x1, 0), (x1, y0), (0, y0)], facecolor='red', edgecolor='0.5', alpha=.2)
        #ax.add_patch(poly)
        #ax.text(x1/2, y0*.5, "misleading", horizontalalignment='center', fontsize=14)

        #poly = Polygon([(0, 0), (x0, 0), (x0, y1), (0, y1)], facecolor='red', edgecolor='0.5', alpha=.2)
        #ax.add_patch(poly)
        #ax.text(x0/2, y1*.25, "misleading", horizontalalignment='center', fontsize=14)

        poly = Polygon([(0, 0), (min(x1, y1), min(x1, y1)), (x1, y0), (0, y0)], facecolor='orange', edgecolor='0.5', alpha=.2)
        ax.add_patch(poly)
        ax.text(x1*.65, y0*.5, "overestimated", horizontalalignment='center', fontsize=14)

        poly = Polygon([(0, 0), (max(x0, y0), max(x0, y0)), (x0, y1), (0, y1)], facecolor='orange', edgecolor='0.5', alpha=.2)
        ax.add_patch(poly)
        ax.text(x0*.65, y1*.25, "overestimated", horizontalalignment='center', fontsize=14)

        poly = Polygon([(0, 0), (min(x1, y1), min(x1, y1)), (0, y1)], facecolor='blue', edgecolor='0.5', alpha=.2)
        ax.add_patch(poly)
        ax.text(x1*.35, y1*.75, "underestimated", horizontalalignment='center', fontsize=14)

        poly = Polygon([(0, 0), (max(x0, y0), max(x0, y0)), (0, y0)], facecolor='blue', edgecolor='0.5', alpha=.2)
        ax.add_patch(poly)
        ax.text(x0*.35, y0*.75, "underestimated", horizontalalignment='center', fontsize=14)

- Positive LLRs that appear above the diagonal, and negative LLRs that appear below the diagonal, became stronger after
  optimization (i.e. further away from 0), and the original LLRs were therefore **underestimated**.
- Positive LLRs that appear below the diagonal, and negative LLRs that appear above the diagonal, became weaker after
  optimization (i.e. closer to 0), and the original LLRs were therefore **overestimated**.

In any case, bias and overestimation does not say anything about the ground truth of individual instances. In a
particular dataset, LLRs of 3 can be underestimated, but that doesn't mean that there cannot be an instance with LLR=3
whose ground truth is H2!


Appearance plots and metrics
----------------------------

Let's see how these metrics and visualizations behave on different types of data.


Neutral LLRs
^^^^^^^^^^^^

First, non-informative data, where all LLRs are zero (i.e. neutral). These data are not discriminative, but perfectly
consistent!

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from lir.data.models import LLRData
    from lir.metrics import cllr, cllr_min, cllr_cal
    from lir.algorithms.devpav import devpav
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
    llrs = LLRData(features=np.zeros((6, 1)), hypothesis_labels=np.array([0, 0, 0, 1, 1, 1]))

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
^^^^^^^^^^^^^^^^^^^^

Now, we have LLRs that are both discriminative and consistent, and data of both hypotheses are drawn from a normal
distribution. It visualizes as follows.

.. jupyter-execute::

    from lir.algorithms.logistic_regression import LogitCalibrator
    from lir.data_strategies import TrainTestSplit
    from lir.datasets.synthesized_normal_binary import SynthesizedNormalData, SynthesizedNormalBinaryData

    # set the parameters for H1 data and H2 data
    h1_data = SynthesizedNormalData(mean=1, std=1, size=1000)
    h2_data = SynthesizedNormalData(mean=-1, std=1, size=1000)

    # generate the data
    instances = SynthesizedNormalBinaryData(h1_data, h2_data, seed=42).get_instances()

    # split the data into a 50% training set and a 50% test set
    training_instances, test_instances = next(TrainTestSplit(test_size=.5).apply(instances))

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
^^^^^^^^^^^^^^^^^^^^^

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
