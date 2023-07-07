# IsChurn
Attempting to predict customer churn on the following kaggle dataset: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

using Genify's Brec model: https://github.com/genifyai/brec_ijcnn2022

## Setup
1. Install the requirements
2. Download the dataset from the kaggle link above and place it in the data folder
3. Run the preprocessing scripts in notebooks folder to generate the train and test data
4. Run the following command to train the model:

```python model/transformer_model.py```

## Data
The data is a colleciton of csv files containing information about users, their listening patterns, transactions, and whetrher they churned or not. 

The user_logs files were too big to fit in memory, additionally, they needed to be merged on user id with the train and test data. These datasets already contained 30 millions and 15 million rows respectively, so merging them with the user_logs would have been too expensive. However, user_logs would add a lot of information to the dataset if incorporated.

Additionally, the classes are imbalanbced, with 93-94% of the users not churning. 

## Model
The model is a transformer model originally tailored for recommending products for users based on their previous purchases. 

This was changed to only handle a binary classification problem, and the model was trained on the train data and evaluated on the test data.

The train and test data consisted of rows, where rows contained user meta data, transaction dates, transaction information, expiration dates, auto renewal information, and the label was whether the user churned or not.

The model was trained on the train data and evaluated on the test data.

## Results
I have experimented with the model for a few days, but I have not been able to get the model to converge. I have tried different hyperparameters, but I have not been able to get the model to converge. I have also tried to train the model on a smaller subset of the data but without luck, leading me to believe that there is something wrong with the data preprocessing steps. 

