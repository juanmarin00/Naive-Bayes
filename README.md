# Naive-Bayes

## Creating our own email classifier from scratch using only class probabilities and the Naive-Bayes algorithm.

## Project structure:
* **Data**:
  Two separate .csv files one for testing and another for training. They both have the same structure. Each row corresponds to one email message. The first column is the response variable and describes whether a message is spam (1.) or ham (0.). The remaining 54 columns are features that you will use to build a classifier. These features correspond to 54 different keywords (such as "money", "free", and "receive") and special characters (such as ":", "!", and "$"). A feature has the value 1. if the keyword appears in the message and 0. otherwise.

* **Naive_bayes.ipynb:**
  This jupyter notebook includes all the code from the project structured in sub sections, in addition to explenations of each section, the probabilistic approach taken and testing for each function.
  
* **Naive_bayes.py:**
  This .py file contains all the code and documentation from the jupyter notebook. 
