# Project 13: Time Series Prediction Ensemble for Taxi Order Forecasting

## Overview
Sweet Lift Taxi has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the number of taxi orders for the next hour. The goal was to build a model to make this prediction and ensure the RMSE on the test set does not exceed 48.

## Dataset Description
The data is stored in the file `taxi.csv`. The number of orders is in the column `num_orders`.

## Project Structure
- `data_preparation_analysis.py`: Code for loading, cleaning, and analyzing the dataset.
- `feature_engineering.py`: Code for creating additional features from the time series data.
- `model_training.py`: Code for training various machine learning models on the dataset.
- `model_evaluation.py`: Code for testing and evaluating the models.
- `results_summary.py`: Code for summarizing the results and providing conclusions.

## Dataset Description
The data is stored in the file `taxi.csv`. The number of orders is in the column `num_orders`.

## Setup
1. Clone the repository:
```sh
git clone https://github.com/your_username/project_17_TimeSeries_Prediction_Ensemble_taxi_order_forecasting.git
cd project_17_TimeSeries_Prediction_Ensemble_taxi_order_forecasting
```

2. Install the required packages:
```sh
pip install -r requirements.txt
```
