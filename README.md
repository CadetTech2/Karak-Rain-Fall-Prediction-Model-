# Karak-Rain-Fall-Prediction-Model-
Rainfall Prediction Model for Karak District, Pakistan. This project uses machine learning to predict monthly rainfall based on historical weather data, including temperature, humidity, and month. Built with Python, Pandas, and scikit-learn, the model aids agricultural planning and water management in the region.
# Rainfall Prediction Model using Machine Learning

## Project Overview

This project is a **Rainfall Prediction Model** built to predict the amount of rainfall based on weather parameters such as **temperature**, **humidity**, and **month**. The model is implemented using **Machine Learning** algorithms, specifically **Linear Regression**, and is trained on historical weather data from **Karak District, Pakistan**.

The goal of this project is to predict rainfall based on input features and to visualize the results. The model can be used to make predictions for future rainfall based on temperature and humidity, which is valuable for agriculture, water management, and planning in regions like Karak.

## Technologies Used

- **Python**: Main programming language used for model implementation.
- **Pandas**: Data manipulation and cleaning.
- **Matplotlib**: Data visualization (for graphs).
- **scikit-learn**: Machine Learning library for creating and training the Linear Regression model.
- **Jupyter Notebook**: Used for testing and exploration.

## Dataset

The dataset used in this project contains historical weather data for the **Karak District** of Pakistan, including columns such as:

- **Month**: The month of the year.
- **Temperature**: The average temperature in °C.
- **Humidity**: The average humidity percentage.
- **Rainfall**: The actual rainfall amount (target variable).

### Sample Data:

| Month | Temperature (°C) | Humidity (%) | Rainfall (mm) |
|-------|------------------|--------------|---------------|
| 1     | 14               | 65           | 25            |
| 2     | 17               | 60           | 20            |
| 3     | 23               | 55           | 15            |
| 4     | 30               | 45           | 10            |
| 5     | 35               | 35           | 5             |

## Model Description

The model uses **Linear Regression** from the **scikit-learn** library to predict the rainfall based on temperature, humidity, and month. The dataset is split into training and testing sets using the `train_test_split()` function.

- The features used for training are: **Temperature**, **Humidity**, and **Month**.
- The target variable to predict is: **Rainfall**.

### Steps to Train the Model:
1. Load and clean the dataset.
2. Preprocess the data and prepare features.
3. Split the data into training and testing sets.
4. Train the **Linear Regression** model.
5. Evaluate the model using metrics like **Mean Squared Error (MSE)** and **R² score**.
6. Make predictions and visualize the results using **Matplotlib**.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rainfall-prediction.git
