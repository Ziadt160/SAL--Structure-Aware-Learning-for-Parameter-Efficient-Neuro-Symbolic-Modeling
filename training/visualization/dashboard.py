import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict

def render_metrics_chart(history: List[float]):
    """
    Renders the live Loss/Accuracy chart.
    """
    if not history:
        st.info("Waiting for training data...")
        return
        
    df = pd.DataFrame({'Score': history})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['Score'], mode='lines', name='Validation Score', line=dict(color='#00CC96')))
    
    fig.update_layout(
        title="Model Evolution (Real-time)",
        xaxis_title="Epochs",
        yaxis_title="Validation Score (Neg Loss)",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_evolution_log(logs: List[Dict]):
    """
    Renders the narrative log as a chat-like interface.
    """
    st.subheader("🧬 Discovery Narrative")
    
    # CSS for chat bubbles
    st.markdown("""
    <style>
    .insight-box {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ccc;
    }
    .insight-success { background-color: #e6fffa; border-color: #00cc96; color: #064e3b; }
    .insight-alert { background-color: #fff5f5; border-color: #fc8181; color: #742a2a; }
    .insight-info { background-color: #f7fafc; border-color: #90cdf4; color: #2d3748; }
    .insight-warning { background-color: #fffff0; border-color: #f6e05e; color: #744210; }
    </style>
    """, unsafe_allow_html=True)
    
    container = st.container()
    
    # Reverse to show newest first
    for log in reversed(logs):
        msg = log['message']
        l_type = log.get('type', 'info')
        
        icon = "ℹ️"
        if l_type == 'success': icon = "✅"
        elif l_type == 'alert': icon = "🚨"
        elif l_type == 'warning': icon = "⚠️"
        
        st.markdown(f"""
        <div class="insight-box insight-{l_type}">
            <strong>{icon}</strong> {msg}
        </div>
        """, unsafe_allow_html=True)

def render_ip_vault(model, best_score, input_cols):
    """
    Renders the 'IP Vault' section for downloading the model.
    """
    st.markdown("---")
    st.header("🔐 IP Vault")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Best Validation Score:** `{best_score:.4f}`")
        if hasattr(model, 'chains'):
            formula = model.chains[0].export_formula(input_cols)
            st.latex(r"y = " + formula[:100] + "...") # Truncate for display
            st.code(formula, language="python")
        else:
            st.info("Model structure not fully symbolic.")
            
    with col2:
        st.success("IP Generation Complete")
        st.download_button(
            label="Download Symbolic Python Model (.py)",
            data=f"# GAI Generated Model\n# Score: {best_score}\n\ndef predict(x):\n    # TODO: Paste formula logic\n    pass",
            file_name="gai_model.py",
            mime="text/x-python"
        )
        st.download_button(
            label="Download C++ Header (.h)",
            data="// C++ Export\n double predict(double* x) { return 0.0; }",
            file_name="gai_model.h",
            mime="text/plain"
        )
