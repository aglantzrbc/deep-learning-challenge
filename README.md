# [Deep Learning Challenge](https://bootcampspot.instructure.com/courses/3337/assignments/54017?module_item_id=962033)

Glantz Adam Bootcamp RUT-VIRT-DATA-PT-04-2023-U-LOLC-MWTH - Module 21

Application Screening Tool for Alphabet Soup

## TABLE OF CONTENTS

1. Overview
2. Results
3. Summary
4. Contributing
5. Acknowledgements
6. Licenses

## 1. Overview:

 - The _purpose_ of the [analysis](https://bootcampspot.instructure.com/courses/3337/assignments/54017?module_item_id=962033) is to create a tool for the nonprofit foundation Alphabet Soup that can help it select the applicants for funding with the best chance of success in their ventures.
 - The _data source_ is an .csv file compiled by Alphabet Soup’s business team containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.
 - The _instrumentality_ is [machine learning](https://en.wikipedia.org/wiki/Machine_learning) and [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network), which are employed to contruct a [binary classifier](https://en.wikipedia.org/wiki/Binary_classification) that can predict whether applicants will be successful if funded by Alphabet Soup, thereby achieving the organization's goal.
 - The business team's file contains the following metadata, each with a definition (**Figure 1**):

![image](https://github.com/aglantzrbc/deep-learning-challenge/assets/127694342/5ef4e180-6606-4ba9-b999-c3d343d8e359)

**Figure 1** | *Variable metadata with definitions*

## 2. Results:

* The purpose of the [analysis](https://bootcampspot.instructure.com/courses/3337/assignments/54015?module_item_id=961978) was to build a model that can identify the creditworthiness of borrowers based on historical lending activity data for a [peer-to-peer lending](https://www.investopedia.com/terms/p/peer-to-peer-lending.asp#:~:text=Peer%2Dto%2Dpeer%20(P2P)%20lending%20is%20a%20form,terms%20and%20enables%20the%20transactions.) services company. [Supervised machine learning](https://en.wikipedia.org/wiki/Supervised_learning) techniques were used to accomplish this goal.
* The financial data included these fields: `loan_size`, `interest_rate`,	`borrower_income`, `debt_to_income`,	`num_of_accounts`,	`derogatory_marks`, `total_debt`, and `loan_status`. It's assumed that the first seven datapoints are the basis for generating the `loan_score` value, which determines the overall disposition of the particular loan. For this analysis, the first seven fields were therefore collectively employed as an [independent variable](https://en.wikipedia.org/wiki/Dependent_and_independent_variables) to predict the eighth field, `loan_status`, the dependent variable.
* The analysis involves a [binary classification](https://en.wikipedia.org/wiki/Binary_classification). The dependent variable, `loan_status`, can only take one of two discrete status values: `0` for `Healthy Loan`, presumably a loan that counts in the applicant's favor for future lending, and '1' for `High-Risk Loan`, which is probably a flag for special scrutiny by the lender.
* The analysis proceeded as follows:
  -  The lending history data was read into a [Python](https://www.python.org/) [Pandas](https://pandas.pydata.org/) [DataFrame](https://www.w3schools.com/python/pandas/pandas_dataframes.asp).
  -  The data was split into the `y` (dependent) variable, or _label_ (i.e., `loan_status`) values and the X (independent) variable, or _feature_, values.
  -  The volumes by `y` were verified.
  -  The data was split into training and testing subsets using the [_train_test_split_](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function.
  -  A [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model was instantiated using the [_LogisticRegression_](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier and fitted using the training data.
  -  The model was then used to make predictions from the testing data.
  -  The model's performance was evaluated by calculating its [balanced accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html), generating a [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html), and creating a [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
  -  In a subsequent round, [_RandomOverSampler_](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) resampled the data to make the quantities artificially equal for each value of `y`.
  -  A new logistic regression model was instantiated using the _LogisticRegression_ classifier and fitted using the resampled training data.
  -  The model was then used to make new predictions from the testing data.
  -  The revised model's performance was evaluated by calculating its balanced accuracy score, generating a confusion matrix, and creating a classification report.
* The analysis used the following two methods:
  -  In the first iteration, the data was used as-is, even though there was a large imbalance in volume between values of the dependent variable. _train_test_split_ was used to divide the data into training and testing batches, the _LogisticRegression_ module created a predictive model, the training data was fitted to it, and predictions were made on the testing data.
  -  In the second round, the data was artificially resampled using the _RandomOverSampler_ function, so that both possible values of `y` have the same volume. As before, the _LogisticRegression_ module created a predictive model, the resampled training data was fitted to it, and new predictions were made on the testing data.

* **Key performance indicators definitions:**
  - **Accuracy:** This is the proportion of the total number of predictions that were correct. It is calculated as (True Positives + True Negatives) / Total Observations. If the dataset is imbalanced, it's better to use a **balanced accuracy** score, which is calculated as the arithmetic mean of sensitivity (a.k.a., recall - the true positive rate) and specificity (true negative rate), effectively taking both classes into account in a balanced manner. Balanced accuracy compensates for bias in favor of the majority class by giving equal weight to each class's sensitivity.
  - **Precision:** This is the ratio of correctly predicted positive observations to the total predicted positives (True Positives / (True Positives + False Positives)). High precision indicates that false positive error is low.
  - **Recall:** Also known as sensitivity, this is the ratio of correctly predicted positive observations to all observations in the actual class. The formula is (True Positives / (True Positives + False Negatives)).
* **Machine Learning Model 1 - data used as-is, with imbalanced volumes for the two values of the `y` variable:**
  - **Accuracy**: The accuracy score is a nearly perfect 99%. The balanced accuracy score is a little lower, reflecting the dataset's bias toward healthy loans, though still high at 94%. Predictions were almost always correct, with only 147 incorrect out of 19,384 cases.
  - **Precision:** A commanding 100% of loans the model predicted as healthy were actually healthy, but a less impressive 87% of loans the model predicted as high-risk were actually high-risk. The model is considerably better at predicting healthy than high risk loans.
  - **Recall:** 100% of the healthy loans in the dataset were identified correctly as healthy, but a lower proportion (89%) of the dataset's high-risk loans were identified as such. Presumably, 11% of the dataset's high-risk loans were incorrectly classified as healthy.
* **Machine Learning Model 2 - data artificially resampled using the _RandomOverSampler_ function, so that both possible values of `y` have the same volume:**
  - **Accuracy**: Both the accuracy and balanced accuracy scores are 100%, mirroring the model's success in the original round, but also converging now that the volume imbalance has been addressed. Predictions were only incorrect for a bare 93 out of 19,384 cases.
  - **Precision:** A full 100% of loans the model predicted as healthy was actually healthy, but only 87% of loans the model predicted as high-risk were actually high-risk. These are identical to the findings from the first iteration before resampling, so precision hasn't improved. The model is "over-predicting", labeling some healthy loans as high-risk. The fact that balanced accuracy and recall are now both 100% across the board means that the small volume of error elided when these values are rounded up is clustering in the precision value for high-risk loans.
  - **Recall:** 100% of the healthy loans in the dataset were identified correctly as healthy and 100% of the high-risk loans were also identified correctly as such. Oversampling improved recall for high-risk loans.

## 3. Summary:

* **I recommend using the model from iteration 2,** in which artificial oversampling was employed to equalize the volume of each value of the `y` variable. The model has high, often nearly perfect, scores across the indicators of balanced accuracy, precision, and recall. When compared to the first iteration, it improved accuracy and recall for the smaller class (i.e., the `High-Risk Loan`, or `1`, class) without lowering it for the larger class (i.e., the 'Healthy Loan", or `0` class). In absolute terms, misclassification of cases is vanishingly rare.
* I suggest that gauging the performance of a model can only be meaningfully accomplished in the context of the problem one trying to solve. **In this case, identifying the high-risk loans is more important than identifying the healthy ones.** Misclassifying a healthy loan as a high-risk loan only incurs an opportunity cost or perhaps the occasional heavy lift by customer service to explain and rectify a lapse, but letting a high-risk loan pass as a healthy loan can cause real damage to a lending institution. Since the number of high-risk loans in the data is small, it was challenging for a machine learning model to predict them with a high level of accuracy; it required inflating the number of high-risk loan records in the training set to optimize the ability to discern them.
* As a caveat, **it is important to continue evaluating a model that achieves such high (e.g., 99-100%) scores**. High metrics could be a sign of data leakage, where information from the test set has leaked into the training process. This can lead to overly optimistic performance estimates that do not generalize well to new data. Moreover, while oversampling can improve performance on the minority class, it can also lead to overfitting on those samples, resulting in inflated metrics. The model might memorize the oversampled examples, including their "noise", rather than learning meaningful patterns. I suggest the following:
  -  **Validating the model on new data beyond the testing set**, to ensure that the strong findings weren't a fluke.
  -  **Use validation techniques such as cross-validation** to assess the model's performance on different subsets of the data and see if the rosy performance survives the change.
  -  **Employ different random states besides "1" with the _RandomOverSampler_ function** to assess whether the model's performance remains consistent across multiple random samples.

## 4. Contributing:

- [Glantz, Adam](https://www.linkedin.com/in/adam-glantz/): Annapolis, Maryland, USA, September 2023, email: adamglantz@yahoo.com

## 5. Acknowledgements:

In addition to using the resources listed above, the author acquired query responses in OpenAI's [ChatGPT](https://chat.openai.com/) versions 3.5 and 4 apps, and the [VSCode GitHub Copilot](https://github.com/features/copilot) app V1.

The author also consulted code and results from similar projects publicly accessible in [GitHub](https://github.com/) repositories and recoverable through [Google](https://www.google.com/) and comparable search engines:

- [Absughe, Khadra](mailto:k.absughe@gmail.com): United Kingdom, September 2022. [deep-learning-challenge](https://github.com/khadra1/deep-learning-challenge)
- [Janer, Jordan](https://www.linkedin.com/in/jordan-janer/): Los Angeles, California, USA, April 2022. [deep-learning-challenge](https://github.com/JordanJaner/deep_learning_challenge)
- [Mathues, Kasey](https://www.linkedin.com/in/kaseymathues/): Philadelphia, Pennsylvania, USA, January 2023. [deep-learning-challenge](https://github.com/kclm40/deep-learning-challenge)
- [Tallant, Jeremy](https://www.linkedin.com/in/jeremy-tallant-717075220/): San Antonio, Texas, USA, March 2023. [deep-learning-challenge](https://github.com/JeremyTallant/deep-learning-challenge)

## 6. Licenses:

- This program is allowed for free use via the [Creative Commons Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) license
