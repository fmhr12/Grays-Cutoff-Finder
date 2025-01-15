# Gray's Cutoff Finder for Survival Analysis

This repository contains a Python implementation for identifying optimal cutoff points for continuous prognostic factors in survival analysis. The method leverages Gray's Test for competing risks to determine the most statistically significant cutoff values.

## Features

Calculates optimal cutoff points for a continuous predictor in survival analysis.
Supports datasets with competing risks and censored data.
Provides Gray's test statistic (U_tilde), its standardized form (Z), and approximated p-values.
Works with survival times and event indicators coded for censored, event of interest, and competing events.

## Use Cases

Find meaningful thresholds for predictors in datasets with survival times. Discover statistically significant cutoffs for further research.

## How It Works
The core algorithm:
Loops through all possible cutoff values of a continuous predictor.
Splits data into two groups based on each cutoff.
Calculates Gray's test statistic for group differences in cumulative incidence functions (CIFs).
Returns cutoff points ranked by the absolute value of the test statistic.

## References
1. Woo S, Kim S, Kim J. Determining cutoff values of prognostic factors in survival data with competing risks. Comput Stat 2016;31:369–86.
2. Kim J, Ng HKT, Kim SW. Assessing the optimal cutpoint for tumor size in patients with lung cancer based on linear rank statistics in a competing risks framework. Yonsei Med J 2019;60:517-524.

