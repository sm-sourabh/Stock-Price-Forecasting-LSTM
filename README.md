# LSTM-Driven Stock Price Forecasting
Welcome to the LSTM-Driven Stock Price Forecasting project repository! In this project, we delve into the realm of machine learning and data analysis to predict stock price returns using the LSTM neural network. Additionally, we create an interactive dashboard for comprehensive stock analysis using the Plotly Dash framework.

## Project Overview
This project encompasses two pivotal aspects:

### Stock Price Prediction using LSTM:
In the initial phase, we harness the power of the LSTM (Long Short-Term Memory) neural network to predict stock price returns. By training the LSTM model on historical stock data, we aim to capture intricate temporal patterns in the stock market, allowing us to make accurate predictions.

### Interactive Dashboard for Stock Analysis using Plotly Dash:
The subsequent phase involves crafting an interactive dashboard using Plotly Dash. This versatile platform seamlessly integrates data from various sources, including the NSE TATA GLOBAL dataset and a multi-stock dataset featuring prominent companies such as Apple, Microsoft, and Facebook. Users can explore dynamic visualizations and tools to conduct comprehensive stock analysis.

## Repository Structure
The repository is organized as follows:
- **assets:** This directory contains the CSS file used for styling the dashboard.
- **data:** Essential data files are stored here, including `lstm_prediction.csv` files, the trained LSTM model `saved_lstm_model.h5`, raw stock data, and `model_performance.xlsx` for detailed model evaluation.
- **images:** Screenshots capturing the essence of the dashboard and key visualizations are housed in this directory.
- **plots:** Every plot generated in the project, such as actual closing stock prices, data split plots, and various iterations of predicted values, can be found here.
- **README.md**: You are here! This file provides an overview of the project, its components, and instructions to set up and run the code.
- **stock_app.py:** The code for the Plotly dashboard, where users can interactively explore stock data and predictions.
- **stock_pred.py:** The Python script implementing the LSTM model for stock price prediction.


# Getting Started
1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/Meet110201/stock-price-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd stock-price-prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Explore the project components and run the `stock_app.py` file to launch the interactive dashboard.


## Stock Price Prediction:
Follow the instructions in the code/ directory to train the LSTM model using the NSE TATA GLOBAL dataset. The model is saved as saved_model.h5 for later use.

## Dashboard Creation:
Run the stock_app.py script to launch the interactive dashboard built with Plotly Dash. Access the dashboard in your web browser to perform detailed stock analysis and visualization.

*Note: Run the `stock_app.py` first. it will automatically run the model and make the predictions.*

## Dashboard Screenshots

![Dashboard-1](/images/Dashboard-1.png)
![Dashboard-2](/images/Dashboard-2.png)
![Dashboard-3](/images/Dashboard-3.png)


## Results and Insights

The project results, including LSTM model predictions and comprehensive analyses, are presented in the `plots` directory. Detailed evaluations and performance metrics can be found in the `data` directory.

## Contributing

Contributions are welcome! If you have ideas for improvements or find any issues, feel free to open an [issue](https://github.com/Meet110201/stock-price-prediction/issues) or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments
We extend our gratitude to the open-source community for creating tools such as TensorFlow, Plotly Dash, and other libraries that have made this project possible.

