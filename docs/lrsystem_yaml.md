# How to use this repository to make an LR system from your data

This page is written for researchers that have collected data and wish to make an LR system using that data.

An LR system is an automated statistical method in which two hypotheses are modeled and which can produce LRs for the hypothesis pair.
These LRs can be made based on validation data for which it is known what the true hypothesis is, so that the LR system can be validated.
Alternatively, LRs can be produced for data for which the true hypothesis is unknown, e.g. case data.

Typically, an LR system will model variants of these two hypotheses:

- Hypothesis 1: the trace is from the same source as the reference
- Hypothesis 2: the trace is from another source than the reference 

This document will help you choose a 'yaml-file' where you can make design choices and put in the specifics and which will let you run an experiment using the LR system.
The yaml-files are pre-made and can be found in this repository. 

To choose which yaml-file is appropriate for you, you need to answer some questions about the data you have and which type of LR system you need.
Then, you will be instructed to organize your data. This ensures the data is in a format that can be parsed by the repository.
Next, you can go to the yaml-file, where you can build you LR system and set up an experiment. You will be guided by comments in the yaml-file.

Before you begin, make sure that you have a working version of LiR. Here's how: [installation guide](index.html).

- Do you have one specific (case-related) reference source along with data from other sources, and do you want to model the first hypothesis exlusively with data from that source?

  --> You probably want a 'specific source' LR system. Go here: [specific source system](#specific-source-system)
  
- Do you have data from multiple sources that are not case-related?

  --> You probably want a 'common source' LR system.

  Answer the follow up questions:
  
  - Does your data contain trace / reference pairs and some score for each pair?

    --> You probably want a 'common source - pre-scored' LR system. Go here: [pre-scored system](#pre-scored-common-source-system)
      
  - Does your data contain single instances of traces / references and measurements / features of those traces?

    --> You want a 'score-based common source' LR system or a 'feature-based common source' LR system:

      - [score-based common source system](#common-source-system)
      - [feature based common source system](#common-source-system)
 
    If you don't know what that distinction means, you can:
      - choose 'score-based' and you will build an LR system similar to [here] (link to practitioner's guide).
      - ask advice (link met e-mail van Ivo)

    If you do know what the distinction means but you are unsure what is best for your data, you can use this repository to build multiple LR systems, perform validation and compare the results.

## Specific source system

In this LR system, you will model hypothesis 1 'trace and reference come from the same source'
(frida)

(uitleggen use case)

(organiseer je data)

(Build an LR system)

(Set up an experiment)

## Feature-based common source system

glass.yaml

(uitleggen wat is feature-based)

(organiseer je data)

(Build an LR system)

(Set up an experiment)

## Score-based common source system

glass.yaml

(uitleggen wat is score-based)

(organiseer je data)

(Build an LR system)

(Set up an experiment)

## Pre-scored common source system

(vocalise/scratch scores, of toch ook frida?)

(uitleggen use case)

(organiseer je data)

(Build an LR system)

(Set up an experiment)

