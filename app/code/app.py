"""
Car Price Predictor , Classification this time (A3)
--------------------------------
Three pages:
  /         -> Old model  (Random Forest, A1)
  /new      -> New model  (custom regularized linear regression, A2)
  /classify -> Classification model (custom softmax logistic regression, A3)

All three share the same preprocessing artifacts (scaler, imputers, metadata) since the feature engineering pipeline is identical
Only the model object different. 
"""

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import os
    
import logistic_regression 
import model_classes

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# shared preprocessing artifacts 
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
median_imputer = joblib.load(os.path.join(MODELS_DIR, "median_imputer.pkl"))
mode_imputer = joblib.load(os.path.join(MODELS_DIR, "mode_imputer.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))

# Loading all three models
model_v1 = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))       # Random Forest (A1)
model_v2 = joblib.load(os.path.join(MODELS_DIR, "model_v2.pkl"))    # custom regression (A2)
model_v3 = joblib.load(os.path.join(MODELS_DIR, "model_v3.pkl"))    # custom classification (A3)
bin_edges = joblib.load(os.path.join(MODELS_DIR, "bin_edges.pkl"))  # price-bucket boundaries for /classify

BRAND_OPTIONS = metadata["brand"]
FUEL_OPTIONS = metadata["fuel"]
SELLER_TYPE_OPTIONS = metadata["seller_type"]
TRANSMISSION_OPTIONS = metadata["transmission"]
OWNER_MAP = metadata["owner_map"]
MEDIAN_COLS = metadata["median_cols"]
MODE_COLS = metadata["mode_cols"]
SCALE_COLS = metadata["scale_cols"]

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Car Price Predictor"


def labeled_input(label, id_, input_type="number", placeholder="", options=None):
    if options is not None:
        field = dcc.Dropdown(
            id=id_,
            options=[{"label": o, "value": o} for o in options],
            placeholder=placeholder,
            style={"width": "100%"},
        )
    else:
        field = dcc.Input(
            id=id_, type=input_type, placeholder=placeholder,
            style={"width": "100%", "padding": "6px"},
        )
    return html.Div(
        [html.Label(label, style={"fontWeight": "bold", "marginBottom": "4px", "display": "block"}), field],
        style={"marginBottom": "16px"},
    )


def build_form(id_prefix):
    """id_prefix distinguishes v1/v2/v3 form fields so all pages can coexist."""
    return html.Div(
        [
            labeled_input("Brand", f"{id_prefix}-brand", options=BRAND_OPTIONS, placeholder="e.g. Maruti"),
            labeled_input("Year", f"{id_prefix}-year", placeholder="e.g. 2018"),
            labeled_input("Km Driven", f"{id_prefix}-km_driven", placeholder="e.g. 45000"),
            labeled_input("Fuel Type", f"{id_prefix}-fuel", options=FUEL_OPTIONS, placeholder="e.g. Petrol"),
            labeled_input("Seller Type", f"{id_prefix}-seller_type", options=SELLER_TYPE_OPTIONS),
            labeled_input("Transmission", f"{id_prefix}-transmission", options=TRANSMISSION_OPTIONS),
            labeled_input("Owner", f"{id_prefix}-owner",
                          options=[o for o in OWNER_MAP.keys() if o != "Test Drive Car"]),
            labeled_input("Mileage (kmpl)", f"{id_prefix}-mileage", placeholder="e.g. 20.5"),
            labeled_input("Engine (CC)", f"{id_prefix}-engine", placeholder="e.g. 1200"),
            labeled_input("Max Power (bhp)", f"{id_prefix}-max_power", placeholder="e.g. 85"),
            labeled_input("Seats", f"{id_prefix}-seats", placeholder="e.g. 5"),
        ],
        style={"maxWidth": "500px", "margin": "0 auto"},
    )


NAV = html.Div(
    [
        dcc.Link("Old Model (Random Forest)", href="/", style={"marginRight": "20px"}),
        dcc.Link("New Model (Regression)", href="/new", style={"marginRight": "20px"}),
        dcc.Link("Classification Model", href="/classify"),
    ],
    style={"textAlign": "center", "padding": "16px", "borderBottom": "1px solid #ddd", "marginBottom": "20px"},
)


def raw_row_from_inputs(brand, year, km_driven, fuel, seller_type, transmission,
                         owner, mileage, engine, max_power, seats):
    row = {
        "year": year, "km_driven": km_driven, "fuel": fuel,
        "seller_type": seller_type, "transmission": transmission,
        "owner": OWNER_MAP.get(owner) if owner else None,
        "mileage": mileage, "engine": engine, "max_power": max_power,
        "seats": seats, "brand": brand,
    }
    df = pd.DataFrame([row])

    df[MEDIAN_COLS] = median_imputer.transform(df[MEDIAN_COLS])
    df[MODE_COLS] = mode_imputer.transform(df[MODE_COLS])

    if not df.loc[0, "fuel"]:
        df.loc[0, "fuel"] = FUEL_OPTIONS[0]
    if not df.loc[0, "seller_type"]:
        df.loc[0, "seller_type"] = SELLER_TYPE_OPTIONS[0]
    if not df.loc[0, "transmission"]:
        df.loc[0, "transmission"] = TRANSMISSION_OPTIONS[0]
    if not df.loc[0, "brand"]:
        df.loc[0, "brand"] = BRAND_OPTIONS[0]
    if pd.isna(df.loc[0, "owner"]):
        df.loc[0, "owner"] = 1
    if not df.loc[0, "year"]:
        df.loc[0, "year"] = 2015
    if not df.loc[0, "km_driven"]:
        df.loc[0, "km_driven"] = 60000

    df = pd.get_dummies(df, columns=["brand", "fuel", "seller_type", "transmission"], drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df


def bucket_label(c):
    """Turn a predicted class index into a human-readable price range."""
    low = bin_edges[c]
    high = bin_edges[c + 1]
    if c == 0:
        return f"Budget (under {high:,.0f})"
    if c == len(bin_edges) - 2:
        return f"Premium (over {low:,.0f})"
    return f"Mid-range ({low:,.0f} - {high:,.0f})"


# Page layouts **

def old_model_page():
    return html.Div([
        NAV,
        html.H1("Car Price Predictor — Old Model", style={"textAlign": "center"}),
        html.P("This page uses the original Random Forest model from Assignment 1.",
               style={"textAlign": "center", "color": "#555", "maxWidth": "600px", "margin": "0 auto 30px auto"}),
        build_form("v1"),
        html.Div(html.Button("Predict Price", id="predict-button-v1", n_clicks=0,
                              style={"padding": "10px 30px", "fontSize": "16px", "cursor": "pointer"}),
                 style={"textAlign": "center", "marginTop": "10px"}),
        html.Div(id="prediction-output-v1",
                 style={"textAlign": "center", "marginTop": "30px", "fontSize": "22px", "fontWeight": "bold"}),
    ], style={"fontFamily": "Arial, sans-serif", "padding": "40px"})


def new_model_page():
    return html.Div([
        NAV,
        html.H1("Car Price Predictor — New Model", style={"textAlign": "center"}),
        html.P("This page uses a custom-built, from-scratch regularized linear regression model (Assignment 2).",
               style={"textAlign": "center", "color": "#555", "maxWidth": "600px", "margin": "0 auto 30px auto"}),
        build_form("v2"),
        html.Div(html.Button("Predict Price", id="predict-button-v2", n_clicks=0,
                              style={"padding": "10px 30px", "fontSize": "16px", "cursor": "pointer"}),
                 style={"textAlign": "center", "marginTop": "10px"}),
        html.Div(id="prediction-output-v2",
                 style={"textAlign": "center", "marginTop": "30px", "fontSize": "22px", "fontWeight": "bold"}),
    ], style={"fontFamily": "Arial, sans-serif", "padding": "40px"})


def classify_model_page():
    return html.Div([
        NAV,
        html.H1("Car Price Predictor — Classification Model", style={"textAlign": "center"}),
        html.P(
            "This page uses a custom-built, from-scratch multinomial logistic regression "
            "model (Assignment 3). Instead of predicting an exact price, it predicts which "
            "price bracket the car most likely falls into.",
            style={"textAlign": "center", "color": "#555", "maxWidth": "600px", "margin": "0 auto 30px auto"},
        ),
        build_form("v3"),
        html.Div(html.Button("Predict Price Bracket", id="predict-button-v3", n_clicks=0,
                              style={"padding": "10px 30px", "fontSize": "16px", "cursor": "pointer"}),
                 style={"textAlign": "center", "marginTop": "10px"}),
        html.Div(id="prediction-output-v3",
                 style={"textAlign": "center", "marginTop": "30px", "fontSize": "22px", "fontWeight": "bold"}),
    ], style={"fontFamily": "Arial, sans-serif", "padding": "40px"})


# Routing **

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/new":
        return new_model_page()
    if pathname == "/classify":
        return classify_model_page()
    return old_model_page()


# Prediction callbacks **

@app.callback(
    Output("prediction-output-v1", "children"),
    Input("predict-button-v1", "n_clicks"),
    State("v1-brand", "value"), State("v1-year", "value"), State("v1-km_driven", "value"),
    State("v1-fuel", "value"), State("v1-seller_type", "value"), State("v1-transmission", "value"),
    State("v1-owner", "value"), State("v1-mileage", "value"), State("v1-engine", "value"),
    State("v1-max_power", "value"), State("v1-seats", "value"),
    prevent_initial_call=True,
)
def predict_price_v1(n_clicks, brand, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, max_power, seats):
    df = raw_row_from_inputs(brand, year, km_driven, fuel, seller_type, transmission,
                              owner, mileage, engine, max_power, seats)
    pred_log = model_v1.predict(df)[0]
    pred_price = np.exp(pred_log)
    return f"Estimated Selling Price: {pred_price:,.0f}"


@app.callback(
    Output("prediction-output-v2", "children"),
    Input("predict-button-v2", "n_clicks"),
    State("v2-brand", "value"), State("v2-year", "value"), State("v2-km_driven", "value"),
    State("v2-fuel", "value"), State("v2-seller_type", "value"), State("v2-transmission", "value"),
    State("v2-owner", "value"), State("v2-mileage", "value"), State("v2-engine", "value"),
    State("v2-max_power", "value"), State("v2-seats", "value"),
    prevent_initial_call=True,
)
def predict_price_v2(n_clicks, brand, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, max_power, seats):
    df = raw_row_from_inputs(brand, year, km_driven, fuel, seller_type, transmission,
                              owner, mileage, engine, max_power, seats)
    X_new = df.values.astype(float)
    X_new = np.concatenate([np.ones((X_new.shape[0], 1)), X_new], axis=1)  # v2 needs intercept col
    pred_log = model_v2.predict(X_new)[0]
    pred_price = np.exp(pred_log)
    return f"Estimated Selling Price: {pred_price:,.0f}"


@app.callback(
    Output("prediction-output-v3", "children"),
    Input("predict-button-v3", "n_clicks"),
    State("v3-brand", "value"), State("v3-year", "value"), State("v3-km_driven", "value"),
    State("v3-fuel", "value"), State("v3-seller_type", "value"), State("v3-transmission", "value"),
    State("v3-owner", "value"), State("v3-mileage", "value"), State("v3-engine", "value"),
    State("v3-max_power", "value"), State("v3-seats", "value"),
    prevent_initial_call=True,
)
def predict_price_v3(n_clicks, brand, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, max_power, seats):
    df = raw_row_from_inputs(brand, year, km_driven, fuel, seller_type, transmission,
                              owner, mileage, engine, max_power, seats)
    X_new = df.values.astype(float)
    X_new = np.concatenate([np.ones((X_new.shape[0], 1)), X_new], axis=1)  # A3 also needs intercept col so we do it again
    pred_class = model_v3.predict(X_new)[0]
    return f"Predicted Price Bracket: {bucket_label(pred_class)}"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)