# src/webapp/callbacks/processed_signal_callbacks.py
from dash import Input, Output
from webapp.app import app


@app.callback(Output("output-div", "children"), [Input("input-box", "value")])
def update_output(value):
    return f"You entered: {value}"
