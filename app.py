#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pickle
import pandas as pd
import gradio as gr

# Load the trained model
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

def predict_is_fraud(amount, log_amount, balance_diff_orig, balance_diff_dest, type_CASH_OUT, type_TRANSFER, amount_mean_rolling, amount_oldbalanceOrg):
    # Prepare the input data as a DataFrame
    data = pd.DataFrame({
        'amount': [amount],
        'log_amount': [log_amount],
        'balance_diff_orig': [balance_diff_orig],
        'balance_diff_dest': [balance_diff_dest],
        'type_CASH_OUT': [type_CASH_OUT],
        'type_TRANSFER': [type_TRANSFER],
        'amount_mean_rolling': [amount_mean_rolling],
        'amount_oldbalanceOrg': [amount_oldbalanceOrg]
    })

    # Perform the prediction
    prediction = model.predict(data)[0]
    return "Fraudulent" if prediction == 1 else "Not Fraudulent"

# Create the input components
input_components = [
    gr.inputs.Number(label="Amount"),
    gr.inputs.Number(label="Log Amount"),
    gr.inputs.Number(label="Balance Difference Origin"),
    gr.inputs.Number(label="Balance Difference Destination"),
    gr.inputs.Checkbox(label="Type CASH_OUT"),
    gr.inputs.Checkbox(label="Type TRANSFER"),
    gr.inputs.Number(label="Amount Mean Rolling"),
    gr.inputs.Number(label="Amount Old Balance Origin")
]

# Create the interface
interface = gr.Interface(
    fn=predict_is_fraud,
    inputs=input_components,
    outputs="text",
    title="Fraud Detection",
    description="Predict if a transaction is fraudulent."
)

# Launch the interface
interface.launch()


# In[ ]:




