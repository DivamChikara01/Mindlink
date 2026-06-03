# MindLink v2 Technical Summary

## Core problem

Some children may struggle to communicate internal stress, overload, or discomfort early. Caregivers may miss the early signs until the situation escalates.

## Product concept

MindLink is a personalized caregiver-support wristband that learns a user's physiological baseline and translates changes into simple color signals.

## Current AI system

The current prototype uses WESAD wrist-sensor data:

- EDA: skin conductance
- BVP: pulse waveform
- TEMP: skin temperature
- ACC: movement

The model extracts statistical features from 60-second windows and trains a Random Forest classifier for:

- baseline
- stress-pattern change

## v2 upgrade

The v2 system adds personalized baseline deviation.

Instead of only asking:

> Does this look like general stress?

It also asks:

> Is this far away from this person's own baseline?

## Color signal logic

- GREEN: baseline-like
- YELLOW: rising or uncertain change
- ORANGE: possible stress-pattern change or unusual physiology
- RED: strong stress-pattern + strong baseline deviation

These are not emotion labels. They are support signals.

## Why this is stronger

A generic stress detector is weak because every person is different. MindLink's stronger claim is personalization.

## Limitations

- Public dataset only
- No original MindLink hardware data yet
- Not medical-grade
- Not emotion detection
- Needs human-subject ethics review before collecting data from other people

## Next research question

Can personalized baseline deviation improve caregiver-facing stress-pattern alerts compared to a general stress classifier alone?
