# CIS400Project
 The github repository for our CIS 400 project

 # Ideas/TODO
 1. We should train our MNB classifier with multiple datasets: e.g. 1. 10000 labled tweets from nltk.twitter_samples 2. x amount of tweets from our set labled with VADER 3. 20000 tweets from nltk.twitter_samples (come unlabled) labeled using VADER (or maybe some other classifier that has already been created) 

2. after all prior preprocessing is done remove any word that still isnt english.

3. fix pandas training data (weird logic to get the class labels in the table)

4. Correct word shortenings e.g. tmr -> tomorrow  .... do this before spelling correction so tmr -> tar doesnt happen

5. Don't remove emojis then tack them on to the end of the sentence normalize them into english words e.g :) == good :D == very good :( == bad D: == very bad