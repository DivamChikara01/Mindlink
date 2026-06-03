# How the MindLink model works

## Model type

Random Forest classifier.

A Random Forest is a group of decision trees. Each tree votes, and the forest makes the final prediction.

## Input

The input is a 60-second wearable sensor window.

Signals:

- ACC = movement
- BVP = blood volume pulse / pulse waveform
- EDA = skin conductance
- TEMP = skin temperature

## Features

For each signal window, the code extracts:

- mean
- standard deviation
- min
- max
- range
- median
- skew
- kurtosis

## Output

- `0 = baseline`
- `1 = possible stress-pattern change`

## Why this model first?

Random Forest is better for the first prototype because:

- it works well on small/medium datasets
- it is easier to explain than a neural network
- it handles tabular features well
- it is faster to train
- it gives a good baseline before deep learning

## What this model cannot claim

Do not claim:

- "detects emotions"
- "diagnoses anxiety"
- "predicts meltdowns"
- "medical-grade accuracy"

Safer claim:

"Detects physiological stress-pattern changes compared to baseline."
