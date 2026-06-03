# MindLink: Personalized Baseline Deviation for Wearable Stress-Pattern Detection

## One-sentence summary
MindLink is a caregiver-support wearable AI concept that learns an individual's physiological baseline and translates stress-pattern changes into simple color signals.

## Problem
Some children may struggle to communicate internal stress, overload, or discomfort early. Caregivers may miss early physiological warning signs before escalation.

## Research question
Can personalized physiological baseline deviation improve wearable stress-pattern alerting compared to a general machine-learning stress classifier?

## Dataset and signals
This prototype uses WESAD wrist-sensor data with wearable features from:
- EDA: skin conductance
- BVP: pulse waveform
- TEMP: skin temperature
- ACC: movement

## Method
The system extracts statistical features from 60-second windows. It compares:
1. General SVM classifier only
2. Personalized baseline deviation only
3. Combined SVM + personalized baseline scoring

Testing uses subject-holdout validation, meaning each subject is tested after tuning/training on other subjects.

## Key result
Best personalized-system result:
- System: **Personalized baseline only**
- Mean accuracy: **0.915**
- Mean stress recall: **0.916**
- Mean stress F1: **0.876**
- False stress alerts: **52**
- Missed stress windows: **30**

Compared with SVM-only:
- SVM-only mean stress F1: **0.798**
- SVM-only missed stress windows: **51**
- SVM-only false stress alerts: **106**

## Sensor ablation finding
Best sensor-ablation result:
- Sensor set: **EDA_BVP_TEMP**
- System: **Combined SVM + baseline**
- Mean accuracy: **0.929**
- Mean stress recall: **0.947**
- Mean stress F1: **0.901**
- Missed stress windows: **19**

## Honest limitations
- This is not emotion detection.
- This is not medical diagnosis.
- WESAD is lab data, not original MindLink wristband data.
- Personalized baseline uses labeled baseline windows as a simulated calibration period.
- Real use would require ethical approval, consent, original sensor data, and user-specific validation.

## Next step
Build a cleaned GitHub repository, prepare a 60–90 second demo video, and seek research feedback on improving the personalized baseline method.
