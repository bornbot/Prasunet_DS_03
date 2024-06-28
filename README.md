# Decision Tree Classifier for Predicting Customer Purchases
This project demonstrates the use of a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. We use a dataset similar to the Bank Marketing dataset from the UCI Machine Learning Repository, which contains a wide range of features that provide insights into the customers' profiles and their interaction history with the bank.

- [Table of Contents](#)
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build a decision tree classifier to predict whether a customer will subscribe to a term deposit. This involves:

1. Data preprocessing to prepare the dataset for modeling.
2. Building and training a decision tree classifier.
3. Evaluating the model's performance using accuracy, confusion matrix, and classification report.
4. Visualizing the results and understanding the decision-making process of the classifier.

## Dataset Description
The dataset used in this project is similar to the Bank Marketing dataset from the UCI Machine Learning Repository. It includes the following features:

- **age** : Age of the customer
- **job**: Type of job
- **marital**: Marital status
- **education**: Level of education
- **default**: Has credit in default?
- **balance**: Average yearly balance in euros
- **housing**: Has housing loan?
- **loan**: Has personal loan?
- **contact**: Type of communication contact
- **day**: Last contact day of the month
- **month**: Last contact month of the year
- **duration**: Last contact duration in seconds
- **campaign**: Number of contacts performed during this campaign
- **pdays**: Number of days since the customer was last contacted
- **previous**: Number of contacts performed before this campaign
- **poutcome**: Outcome of the previous marketing campaign
- **y**: Target variable indicating whether the customer subscribed to a term deposit

## Installation
To run this project, you need to have Python and Jupyter Notebook installed on your machine. Additionally, install the required Python libraries using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn 
```

## Usage
1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/decision-tree-classifier.git
cd decision-tree-classifier
```
2. Open the Jupyter Notebook:

```bash
jupyter notebook DS_03.ipynb
```
3. Run the notebook cells to see the data preprocessing, model training, evaluation, and visualization steps.

## Model Training and Evaluation

### Data Preprocessing
The dataset is first loaded into a pandas DataFrame and preprocessed to handle any missing values, encode categorical variables, and split the data into training and testing sets.

### Model Training
We use the DecisionTreeClassifier from the scikit-learn library to train our model. The training process involves fitting the model to the training data (X_train and y_train).

### Model Evaluation
The model's performance is evaluated on the test set (X_test and y_test). We calculate the accuracy, generate a confusion matrix, and produce a classification report to understand the model's precision, recall, and F1-score for each class.

## Results
The results section includes detailed metrics and visualizations that illustrate the performance of the decision tree classifier:

- Accuracy: The proportion of correctly classified instances out of the total instances.
- Confusion Matrix: A table that shows the number of correct and incorrect predictions for each class.
- Classification Report: A comprehensive summary of the precision, recall, and F1-score for each class.

These results help us understand how well the model performs in predicting customer purchases and highlight areas for potential improvement.

## Contributing
Contributions to this project are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the [LICENSE](#license) file for more details.
