import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
from models.adaptive_neural_model import MatrixGGLEN
from training.trainer import GAIOptimizer
from training.narrative import NarrativeGenerator
from training.visualization.dashboard import render_metrics_chart, render_evolution_log, render_ip_vault

# Page Config
st.set_page_config(page_title="GAI Discovery Platform", page_icon="🧬", layout="wide")

# Session State Init
if 'history' not in st.session_state: st.session_state['history'] = []
if 'logs' not in st.session_state: st.session_state['logs'] = []
if 'best_score' not in st.session_state: st.session_state['best_score'] = -float('inf')
if 'model' not in st.session_state: st.session_state['model'] = None

# Sidebar
st.sidebar.title("🧬 GAI Platform")
st.sidebar.markdown("Self-Evolving Symbolic AI")

data_source = st.sidebar.radio("Data Source", ["Synthetic (Demo)", "Upload CSV"])
dataset = None
input_dim = 1

if data_source == "Synthetic (Demo)":
    st.sidebar.info("Generates: y = sin(3x) + 0.5*x")
    # Generate Synthetic Data
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(3 * X) + 0.5 * X + np.random.normal(0, 0.1, X.shape)
    dataset = (X, y)
    input_dim = 1
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        target = st.sidebar.selectbox("Target Column", df.columns)
        features = [c for c in df.columns if c != target]
        X = df[features].values
        y = df[target].values
        dataset = (X, y)
        input_dim = len(features)

epochs = st.sidebar.slider("Evolution Epochs", 50, 2000, 500)
temp = st.sidebar.slider("Initial Temperature", 0.1, 1.0, 0.5)

# Main Dashboard
st.title("AI Discovery Engine")

col1, col2 = st.columns([2, 1])

# Placeholders
with col1:
    chart_container = st.empty()

with col2:
    log_container = st.empty()

start_btn = st.button("🚀 Start Discovery Process", type="primary")

if start_btn and dataset:
    X_train, y_train = dataset
    # Split
    split = int(len(X_train) * 0.8)
    X_t, y_t = X_train[:split], y_train[:split]
    X_v, y_v = X_train[split:], y_train[split:]
    
    # Init
    model = MatrixGGLEN(input_dim=input_dim, hidden_dim=8, num_chains=2, chain_depth=3)
    narrator = NarrativeGenerator()
    
    # Reset State
    st.session_state['history'] = []
    st.session_state['logs'] = []
    
    # Observer Callback
    def training_observer(event_type, details):
        if event_type == 'epoch_end':
            # Update Chart
            st.session_state['history'].append(details['score'])
            if details['epoch'] % 20 == 0:
                with chart_container.container():
                    render_metrics_chart(st.session_state['history'])
                    
        elif event_type == 'mutation':
            # Generate Narrative
            log_entry = narrator.generate_event_log(event_type, details)
            st.session_state['logs'].append(log_entry)
            
            # Update Log
            with log_container.container():
                render_evolution_log(st.session_state['logs'])
                
    # Run
    optimizer = GAIOptimizer(model, lr=0.01, patience=20, grace_period=10, initial_temp=temp, observer=training_observer)
    
    with st.spinner("Evolving Mathematical Structure..."):
        optimizer.fit(X_t, y_t, X_v, y_v, epochs=epochs, name="Demo")
        
    st.session_state['model'] = model
    st.session_state['best_score'] = optimizer.best_score
    st.success("Discovery Complete!")

# Final Vault
if st.session_state['model']:
    render_ip_vault(st.session_state['model'], st.session_state['best_score'], ["x"] if input_dim==1 else [f"x{i}" for i in range(input_dim)])
