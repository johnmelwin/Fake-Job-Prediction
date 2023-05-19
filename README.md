# Fake Job Prediction using Machine learning and NLP

💼 This project aims to predict fake job postings using machine learning techniques and natural language processing (NLP). The project utilizes a dataset containing information about job postings, such as job titles, locations, descriptions, and requirements. By analyzing these textual features, the model predicts whether a job posting is fraudulent or genuine.

## Project Description

The project consists of the following main components:

- **TrainTest.py**: This script loads the dataset, preprocesses the data, splits it into training and testing sets, trains the model, and evaluates its performance using classification metrics. The script uses the `my_model` class from the `Model.py` module.

- **Model.py**: This module contains the `my_model` class, which implements the machine learning model for fake job prediction. The class includes methods for data preprocessing, feature engineering, model training, and prediction.

- **MyEvaluation.py**: This module provides evaluation functions for assessing the performance of the model. The `my_evaluation` class is used in `TrainTest.py` to calculate classification metrics such as F1 score.

## Getting Started

To get started with the project, follow these steps:

1. Install the required dependencies mentioned in the project code.

2. Download the dataset file `job_train.csv` and place it in the `data` folder.

3. Open the project in an environment with Python installed (preferably Python 3).

4. Run the `TrainTest.py` script. This script will load the dataset, preprocess the data, train the model, and evaluate its performance.

## Results

After running the `TrainTest.py` script, the model achieved impressive results:

- The model achieved an accuracy of 98% ✅, indicating that it correctly classified the job postings as genuine or fraudulent with high precision.

- The F1 score, a metric that combines precision and recall, was 0.77 💪. This score demonstrates the model's ability to find a balance between identifying fraudulent job postings (class 1) and correctly classifying genuine ones (class 0).

🚀 These results highlight the effectiveness of the machine learning model and its capability to accurately predict fake job postings using the provided dataset.

## Contributing

🤝 Contributions to this project are welcome. If you have any suggestions or improvements, please submit a pull request or open an issue. Let's work together to make this project even better!
