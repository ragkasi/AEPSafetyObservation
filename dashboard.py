import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

# Load classified data
def load_classified_data(file_path):
    return pd.read_csv(file_path)

# Generate a risk distribution plot
def plot_risk_distribution(data):
    fig = px.histogram(data, x='predicted_risk_level', title='Distribution of Predicted Risk Levels')
    return fig

# Generate a bar plot for observation types vs. risk levels
def plot_observation_types(data):
    fig = px.bar(data, x='Observation Type', color = 'predicted_risk_level', title = 'Risk Levels by Observation Type')
    return fig

# Set up the dashboard
def create_dashboard(data):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1('Safety Observations Dashboard'),
        dcc.Graph(figure=plot_risk_distribution(data)),
        dcc.Graph(figure=plot_observation_types(data))
    ])

    return app

if __name__ == "__main__":
    data = load_classified_data('classified_test_data.csv')
    app = create_dashboard(data)
    app.run_server(debug=True)
