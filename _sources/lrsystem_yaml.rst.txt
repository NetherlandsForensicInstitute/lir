LR System selection helper
===========================

This page is written for researchers that have collected data and wish to make an LR system using that data.

Before you begin, make sure that you have a [working version of LiR](index.html).

An LR system is an automated statistical method in which two hypotheses are modeled and which can produce LRs for the hypothesis pair.
These LRs can be made based on validation data for which it is known what the true hypothesis is, so that the LR system can be validated.
Alternatively, LRs can be produced for data for which the true hypothesis is unknown, e.g. case data.

Typically, an LR system will model variants of these two hypotheses:

    - Hypothesis 1: the trace is from the same source as the reference
    - Hypothesis 2: the trace is from another source than the reference 

This document will help you choose a 'yaml-file' where you can make design choices and put in the specifics and which will let you run an experiment using the LR system.
A yaml-file is a text file in which you write what the code will do for you. The yaml-files are pre-made and can be found in this repository. 
The pre-made yaml-files are templates, for you to adapt to your situation.

To choose which yaml-file is appropriate for you, you need to answer some questions about the data you have and which type of LR system you need.
Then, you will be instructed to organize your data. This ensures the data is in a format that can be processed by LiR.
Next, you can go to the yaml-file, where you can build your LR system and set up an experiment. You will be guided by comments in the yaml-file.

- **Do you have one specific (case-related) reference source along with data from other sources, and do you want to model the first hypothesis exclusively with data from that source?**

  YES: You probably want a 'specific source' LR system. Go here: :ref:`specific-source-system`
  
  NO: You probably have data from multiple sources that are not case-related.
  That means you probably want a 'common source' LR system. Continue with the following question:

  - **Does your data contain trace / reference pairs and some score for each pair?**

    YES: You probably want a 'common source - pre-scored' LR system. Go here: :ref:`pre-scored-system`

    NO: Your data probably contain single instances of traces / references and measurements / features of those traces. You want a 'score-based common 
    source' LR system or a 'feature-based common source' LR system, go here: :ref:`common-source-system`.


.. _specific-source-system:

Specific source system
----------------------

In this LR system, you will:
model Hypothesis 1 'trace and reference come from the same source' with data from one specific source.
model Hypothesis 2 'trace and reference come from different sources' with data from other sources.

You'll need your data in csv or txt-format.

Here's a minimal example of data that would work:
'hypothesis_label,feature1,feature2,feature3'

It is also possible to use sourceIDs instead of hypothesis label and some other columns are optional, for more information see HERE

The template-yaml that you should use is: `specificsource.yaml`_ (TODO)

.. _specificsource.yaml: https://raw.githubusercontent.com/NetherlandsForensicInstitute/lir/refs/heads/main/examples/specificsource.yaml


.. _pre-scored-system:

Pre-scored common source system
--------------------------------

In this LR system, you will:
    - model Hypothesis 1 'trace and reference come from the same source' with data pairs from multiple sources.
    - model Hypothesis 2 'trace and reference come from different sources' with data pairs from multiple sources.

The rows in your data describe source comparisons, and not measurements or features of individual instances. Typically you will have one numerical value per comparison.
The pairs could be made up of sources or of instances from sources.

You'll need your data in csv or txt-format.

Here's a minimal example of data that would work:
'sourceID1,sourceID2,score'

sourceID1 and sourceID2 will be the same value for same-source-comparisons. Some other columns are optional, for more information see HERE
The template-yaml that you should use is: `prescored_commonsource.yaml`_ (TODO)

.. _prescored_commonsource.yaml: https://raw.githubusercontent.com/NetherlandsForensicInstitute/lir/refs/heads/main/examples/prescored_commonsource.yaml


.. _common-source-system:

Common source system
--------------------

In this LR system, you will:
    - model Hypothesis 1 'trace and reference come from the same source' with data pairs from multiple sources.
    - model Hypothesis 2 'trace and reference come from different sources' with data pairs from multiple sources.

You'll need your data in csv or txt-format.

Here's a minimal example of data that would work:
'sourceID,feature1,feature2,feature3'

Some other columns are optional, for more information see HERE

It is possible to make a score-based common-source LR system or a feature-based common-source LR system.

If you don't know what that distinction means, you can:
  - choose 'score-based' and you will build an LR system similar to [here] (link to practitioner's guide).
  - ask advice from your favourite forensic statistician.

If you do know what the distinction means but you are unsure what is best for your data, you can use this repository to build multiple LR systems, perform validation and compare the results.

The template-yaml that you could use is: `scorebased_commonsource.yaml`_ (TODO)

The template-yaml that you could use is: `featurebased_commonsource.yaml`_ (TODO)


.. _scorebased_commonsource.yaml: https://raw.githubusercontent.com/NetherlandsForensicInstitute/lir/refs/heads/main/examples/scorebased_commonsource.yaml

.. _featurebased_commonsource.yaml: https://raw.githubusercontent.com/NetherlandsForensicInstitute/lir/refs/heads/main/examples/featurebased_commonsource.yaml
