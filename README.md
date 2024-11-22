# Gray's Cutoff Finder for Survival Analysis
Gray's Cutoff Finder for Survival Analysis

This repository contains a Python implementation for identifying optimal cutoff points for continuous prognostic factors in survival analysis. The method leverages Gray's Test for competing risks to determine the most statistically significant cutoff values.

## Features

Calculates optimal cutoff points for a continuous predictor in survival analysis.
Supports datasets with competing risks and censored data.
Provides Gray's test statistic (U_tilde), its standardized form (Z), and approximated p-values.
Works with survival times and event indicators coded for censored, event of interest, and competing events.

## Use Cases

Survival Analysis: Find meaningful thresholds for predictors in datasets with survival times.
Competing Risks: Analyze outcomes with competing events (e.g., death from other causes).
Data Exploration: Discover statistically significant cutoffs for further research.

## How It Works
The core algorithm:
Loops through all possible cutoff values of a continuous predictor.
Splits data into two groups based on each cutoff.
Calculates Gray's test statistic for group differences in cumulative incidence functions (CIFs).
Returns cutoff points ranked by the absolute value of the test statistic.
