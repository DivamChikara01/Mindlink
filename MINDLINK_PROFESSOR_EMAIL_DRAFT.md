Subject: Wearable AI project — personalized baseline stress-pattern detection

Hi Professor Yiwen,

I’m a high school student working on MindLink, a wearable AI caregiver-support project focused on detecting physiological stress-pattern changes relative to an individual's baseline.

I built a prototype using WESAD wrist-sensor data. The system extracts features from EDA, BVP, temperature, and accelerometer signals, then compares general stress classification with personalized baseline-deviation methods.

My main research question is:

Can personalized physiological baseline deviation improve wearable stress-pattern alerting compared to a general machine-learning classifier?

In subject-holdout testing, the strongest sensor-ablation result used EDA_BVP_TEMP with Combined SVM + baseline. It reached about 0.901 mean stress F1 and 0.947 mean stress recall. This outperformed the general SVM-only approach, which reached about 0.798 mean stress F1.

I know this is not emotion detection or a medical device yet. The current work is a prototype using public lab data, and my next step is improving the personalized baseline method and planning ethical original data collection later.

Would you be willing to give feedback on my validation approach or suggest how I should improve the personalization method?

Thank you,
Divam
