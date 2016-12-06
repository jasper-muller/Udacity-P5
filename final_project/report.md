# Report
Introduction

## Project goals
>Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it.

The goal of this project is to identify people that participated in the Enron fraud. The starting point for this investigation is a dataset on Enron employees, which contains information on both employee financials and emails. More importantly, part of the dataset was manually labelled to distinguish between people of interest (POI) and people that were not of interest (no POI).

We use the labelled data to train a machine learning algorithm that will help us to identify persons of interest.

> Give some background on the dataset and how it can be used to answer the project question.

The dataset used for training our machine-learning algorithm contains information on 146 Enron employees across 21 features. In total, 18 of these employees were labelled as a person of interest. The 21 features can roughly be divided in the following four categories:

Related to financials (in USD):
- **bonus**: financial bonus received by this person
- **salary**: yearly salary received by this person
- **total_stock_value**: total value of the stocks owned by this person
- **total_payments**: sum of all payments by Enron to this person
- **expenses**: the expenses done by Enron for this person
- **director_fees**: the fees received by this person because he/she is a director
- **exercised_stock_options**: stock options that were traded for money before Enron's bankruptcy
- **long_term_incentive**: some kind of bonus that is paid for achieving long term results

Related to emails:
- **email_address**: the person's email address
- **from_messages**: the amount of email messages sent by this person
- **to_messages**: the amount of email messages received by this person
- **from_poi_to_this_person**: the amount of email messages from a person of interest to this person
- **from_this_person_to_poi**: the amount of email messages from this person to a person of interest

Related to persons of interest:
- **poi**: flag to identify whether a person is a person of interest. E.g. because of lawsuits etc.

And lastly, the features for which I do not yet understand their meaning:
- deferral_payments
- deferred_income
- loan_advances
- other
- restricted_stock
- restricted_stock_deferred
- shared_receipt_with_poi

It must be obvious that I will not use the features for which I do not understand their meaning. For all other features, I will try and see how well they are able to classify an Enron employee as a person of interest.

> Were there any outliers in the data when you got it, and how did you handle those?

For the full exploration of outliers, please refer to the notebook included with this submission, `data exploration.ipynb`. Here I will summarize my findings:

The first outlier that immediately caught my eye, was the entry *TOTAL*. I found this outlier by plotting a histogram of employee salaries, which showed a value of 26.7 million USD all the way to the right. Assuming that this entry describes the total for all numeric columns, I removed the entry from the dataset.

Apart from this obvious outlier, there was one other employee that I excluded: *The Travel Agency In The Park*. This entry suggested that it did not describe an employee but rather a company of some sort.

## Feature selection
> What features did you end up using in your POI identifier, and what selection process did you use to pick them?

> Did you have to do any scaling? Why or why not?

> As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.

> If you used an algorithm like a decision tree, please also give the feature importances of the features that you use

> If you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

#### Create new features
At least one new feature is implemented. Justification for that feature is provided in the written response, and the effect of that feature on the final algorithm performance is tested. The student is not required to include their new feature in their final feature set.

#### Properly scale features
If algorithm calls for scaled features, feature scaling is deployed.

#### Intelligently select features
- Univariate or recursive feature selection is deployed, or features are selected by hand (different combinations of features are attempted, and the performance is documented for each one).
- Features that are selected are reported and the number of features selected is justified. For an algorithm that supports getting the feature importances (e.g. decision tree) or feature scores (e.g. SelectKBest), those are documented as well.

## Algorithm selection
> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

#### Pick an algorithm
At least 2 different algorithms are attempted and their performance is compared, with the more performant one used in the final analysis.

## Parameter tuning
>What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

#### Tune the algorithm
At least one important parameter tuned with at least 3 settings investigated systematically, or any of the following are true:

GridSearchCV used for parameter tuning
Several parameters tuned
Parameter tuning incorporated into algorithm selection (i.e. parameters tuned for more than one algorithm, and best algorithm-tune combination selected for final analysis).


## Validation
>What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]


#### Validation strategy
Response addresses what validation is and why it is important.

Performance of the final algorithm selected is assessed by splitting the data into training and testing sets or through the use of cross validation, noting the specific type of validation performed.

## Evaluation metrics
>Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

#### Usage of evaluation metrics
At least two appropriate metrics are used to evaluate algorithm performance (e.g. precision and recall), and the student articulates what those metrics measure in context of the project task.
