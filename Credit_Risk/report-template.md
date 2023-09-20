# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The main objective of this analysis was to develop a machine learning model capable of predicting loan status based on various financial parameters. By using historical data, the analysis provides insights that can assist in decision-making, to identify likelihood of a loan being repaid. 

* Explain what financial information the data was on, and what you needed to predict.
loan size: The amount of loan that has been taken:  
interest rate:The rate of interest applicable to the loan.,  
borrower's income: The income of the borrower., 
Total Debt: The total amount of debt the borrower has.


* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
The main idea is to predict the loan status, which indicates whether a loan is likely to default (1) or not (0). 

* Describe the stages of the machine learning process you went through as part of this analysis.
1.Data Import and Exploration: Initial exploration involved reviewing the dataset using methods like head() and tail() to understand the data structure and available variables.

2.Data Preprocessing: separating the dataset into features (X) and labels (y). You also performed basic data inspections, such as checking for null values and obtaining descriptive statistics.

3.Data Splitting:  train_test_split to segregate the data into training and testing sets, which helps in validating the model's performance on unseen data.

4.Model Building and Training:
instantiated and trained a logistic regression model using the training data, utilizing the LogisticRegression class from the scikit-learn library.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
utilized the Logistic Regression model, a popular algorithm for binary classification problems.LogisticRegression class from the scikit-learn library.

evaluating the model's performance using various metrics, imbalance and optimizing the model through hyperparameter 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
Accuracy: 
The accuracy score was approximately 99.22%, Predictions is about 99.22% of the time on the test data.

Precision:
Class 0 (Healthy Loan): 100% makes the loans that were labeled as healthy.
Class 1 (High-risk Loan): 86% - This suggests that when the model labeled a loan as high risk, it was correct about 86% of the time.

Recall:
Class 0 (Healthy Loan): 100% - 
Class 1 (High-risk Loan): 91% - This means that the model was able to correctly identify 91% of the actual high-risk loans.

F1-Score:
Class 0 (Healthy Loan): 100% 
Class 1 (High-risk Loan): 88% 


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
Accuracy: 
The model had an accuracy of about 99%, which means that it correctly predicted the loan status in 99% of the cases in the test dataset.

Precision:
Class 0 (Healthy Loan): 100%  the model labeled a loan as healthy, it was correct all the time.
Class 1 (High-risk Loan): 85% - This means that when the model labeled a loan as high-risk, it was correct 85% of the time.

Recall:
Class 0 (Healthy Loan): 99% the model was able to correctly identify 99% of the actual healthy loans.
Class 1 (High-risk Loan): 99% - This shows that the model could correctly identify 99% of all actual high-risk loans.

F1-Score:
Class 0 (Healthy Loan): 100% 
Class 1 (High-risk Loan): 92% 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?


* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
If you do not recommend any of the models, please justify your reasoning.

Based on the analysis, Logistic Regression Model with Resampled Training Data seems to perform the best because higher accuracy is 99%, precision is(85%), better balance of precision 99% and recall 92%.

Therefore, I would recommend using logistic regression model that was fitted with the oversampled data suggests that the model performs very well in predicting both 0 (healthy loan) and 1 (high-risk loan) labels.
