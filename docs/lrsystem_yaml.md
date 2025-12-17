# How to set up LR experiments from a YAML file

This page shows how to set up an experiment generate results by building a YAML file.

Wat is een pannekoek zonder kaas? Een panne-oe

1. (eerst bepalen hoe je modelleert)
2. Organize your data
3. Build an LR system
4. Set up an experiment

Before you begin, make sure that you have a working [`lir` command](index.html).

The LR system will generally model variants of two hypotheses:

- Hypothesis 1: the trace is from the same source as the reference
- Hypothesis 2: the trace is from another source than the reference 

(eerst bepalen hoe je modelleert)

(klik hier voor uitgebreide uitleg over wat voor soort lrsystemen er zijn)

- There is one specific reference and plenty of data from that reference? --> [specific source system](#specific-source-system)
- The LR system should be reusable for multiple traces
  - The data already contains trace/reference pairs and their comparison scores --> [pre-scored system](#pre-scored-common-source-system)
  - Wil je score-based of feature-based? Als je niet weet wat dat betekent, kies score-based of (klik hier) laat je adviseren. als je wel weet wat dat betekent maar je weet niet wat het beste is: experimenteer met verschillende systemen [score-based common source system](#common-source-system) [feature based common source system](#common-source-system)

## Specific source system

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

