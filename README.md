# Group 11 Readme
### JANSSON Hampus - HELLUM Jacob - CHAKOV Ryan LEQUEU Pierre-Antoine - NIELSEN Sigurd Natural Language Processing - SNU 2023




## Motivation
The primary goal of this project is to develop a model capable of determining key human features and personality traits. Value-Persona classification is a broad field of study with numerous applications. Some motivating examples include enhancing system user experience in socially intelligent chatbots (e.g., ELIZA, PARRY, Persona-chat, chatGPT), healthcare communication, demographic classification, social media moderation, and applications in political campaigns, marketing, and advertising.

## Dataset Details
### ValueNet Dataset:
- 21,000+ scenarios
- Distributed among 10 values
    - Benevolence
    - Power
    - Tradition
    - Universalism
    - Stimulation
    - Self-Direction
    - Security
    - Hedonism
    - Conformity
    - Achievemen

## Proposal of Idea and Explanation
The project is based on the 10 basic human values identified in the ValueNet dataset. The aim is to create a value classification for a sentence and determine the value profile of a set of sentences. Three methods are explored for this purpose:
1. Naive Bayes
2. Ensemble of BERT models
3. LSTM (Long Short-Term Memory networks)

## Naive Bayes
Used both BOW Tokenization and TF-IDF Tokenization


## Ensemble of BERT Models
- Create 10 different datasets
- Create 10 different models
- Implement a training function
- Train each model
- Create an ensemble of models
- Pass a new sentence in and observe the results

## LSTM
### First Approach:
- Clean all entries with a value of zero for the label
- Train ten classifiers (one for each value)
- Result accuracy on test data:
    - Benevolence: 0.8631
    - Power: 0.6023
    - Tradition: 0.9349
    - Universalism: 0.8202
    - Stimulation: 0.7899
    - Self-Direction: 0.6308
    - Security: 0.7375
    - Hedonism: 0.6227
    - Conformity: 0.9125
    - Achievement: 0.7193

### Second Approach:
- Treat the data the same way as in other models (do not clean zeros)
- Train two models for each value (20 in total)
- Get softmax scores from both models and take the difference as a result
- Example: `softmax(BENEVOLENCE) - softmax(NOT_BENEVOLENCE)`
- Note: This approach encountered challenges and the model gave softmax of 1.0 to all entries. Further investigation and troubleshooting may be required.
