# Optimizing Activity Performance without Model Retraining

The precision and recall of fishing event detection can be fine-tuned through several parameters:

## 1. Subpath Creation Frequency

Increasing subpath creation frequency enhances the number of messages processed for inference. This likely improves recall but has an indeterminate effect on precision across distinct events.

Adjustment can be made via constants in `cpd/constants.py`.

## 2. Subpath Service Data Request Parameters

The subpath service feeding the activity pipeline is governed by three key parameters:
- Minimum messages
- Maximum messages
- Maximum lookback days

Reducing the minimum messages may increase recall. Decreasing the maximum lookback days constrains the data to more recent periods, potentially improving signal quality but potentially reducing inference frequency.

Implementation of these adjustments requires rigorous testing and monitoring in the integrated environment.
