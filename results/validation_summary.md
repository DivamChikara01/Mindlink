# MindLink Validation Summary

## Task
Baseline vs stress-pattern change detection using WESAD wrist-sensor features.

## Validation method
Subject-independent GroupKFold cross-validation. This tests on people the model did not train on.

## Results
- Average F1: 0.7862
- Average accuracy: 0.8426
- Stress recall: 0.7514
- Baseline recall: 0.8961

## Baseline comparisons
- Dumb model average F1: 0.0000
- Shuffled-label average F1: 0.2491

## Interpretation
If the real model is much better than the dumb and shuffled-label models, it is probably learning real sensor patterns instead of random noise.

## Limitation
This is still not a medical device and not emotion detection. It needs original MindLink device data later.
