# Titanic Survival Prediction using Artificial Neural Network (ANN)

This project utilizes an Artificial Neural Network (ANN) to predict the survival of passengers aboard the Titanic. The dataset includes various features such as passenger class, age, gender, and survival status.

## Files Included

- `titanic_train.py`: Python script for training the ANN model.
- `titanic_predict.py`: Python script for predicting survival using a trained model.
- `titanic-new.csv`: CSV file containing new data for prediction.
- `Titanic_names.csv`: CSV file containing passenger names.
- `Titanic_data.csv`: CSV file containing Titanic passenger data.

## Procedure

### 1. Data Pre-processing and Exploration

- Load and merge the passenger data files into a Pandas DataFrame.
- Familiarize yourself with the data using `describe()` and `hist()`.
- Remove invalid categories and handle missing values.

### 2. Dividing the Data

- Separate the features (X) and the target variable (y).
- Select relevant columns for X, such as 'PClass', 'Age', and 'Gender'.
- Define y as the 'Survived' column.

### 3. Converting Categorical Variables

- Convert categorical variables like 'Gender' and 'PClass' into numerical dummy variables.
- Use `ColumnTransformer` and `OneHotEncoder` for dummy conversion.
- Remove original categorical columns from X.

### 4. Splitting the Data

- Split the dataset into training and testing data using `train_test_split`.
- Scale the feature data using `StandardScaler`.

### 5. Creating and Training the ANN Model

- Initialize a Sequential model using TensorFlow's Keras.
- Add input, hidden, and output layers with appropriate activation functions.
- Compile the model with the Adam optimizer.
- Fit the model to the training data.

### 6. Model Evaluation

- Test the trained model using the testing data.
- Evaluate the model's performance using confusion matrix and accuracy metrics.

### 7. Saving the Model

- Save the trained model, encoder, and scaler to disk using appropriate file formats.

### 8. Prediction with New Data

- Load the trained model, encoder, and scaler.
- Read new data from a CSV file and preprocess it.
- Use the model to predict survival probabilities for new data.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- TensorFlow
- Scikit-learn
- Seaborn

## Usage

- Run `titanic_train.py` to train the model and save it.
- Run `titanic_predict.py` to load the trained model and make predictions on new data.
