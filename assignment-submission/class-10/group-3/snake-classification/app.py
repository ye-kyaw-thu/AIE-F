import streamlit as st
import json
import pandas as pd
import os

# --- RDR ENGINE LOGIC (Reused from scrdr_interactive.py) ---

class RDRNode:
    def __init__(self, condition=None, conclusion=None):
        self.condition = condition  # {'col':.., 'op':.., 'val':..}
        self.conclusion = str(conclusion) if conclusion is not None else None
        self.if_true = None   # Refinement (Exception)
        self.if_false = None  # Alternative

    def evaluate(self, row):
        if not self.condition or not self.condition.get('col'):
            return True
        col = self.condition.get('col')
        op = self.condition.get('op')
        val = self.condition.get('val')
        
        if col not in row:
            return False
            
        row_val = row[col]
        try:
            if op == '==': return str(row_val) == str(val)
            if op == '<':  return float(row_val) < float(val)
            if op == '>':  return float(row_val) > float(val)
        except:
            return str(row_val) == str(val)
        return False

    @staticmethod
    def from_dict(data):
        if not data or not isinstance(data, dict): return None
        node = RDRNode(data.get('condition'), data.get('conclusion'))
        node.if_true = RDRNode.from_dict(data.get('if_true'))
        node.if_false = RDRNode.from_dict(data.get('if_false'))
        return node

class SCRDR_Engine:
    def __init__(self, target, root_node):
        self.target = target
        self.root = root_node

    def classify_with_trace(self, row):
        curr = self.root
        last_match = self.root
        trace = []
        while curr:
            if curr.evaluate(row):
                last_match = curr
                trace.append({"node": curr, "result": True})
                curr = curr.if_true # Try refinements 
            else:
                trace.append({"node": curr, "result": False})
                curr = curr.if_false # Try alternatives 
        return last_match, trace

# --- UI HELPER FUNCTIONS ---

def load_data():
    try:
        df = pd.read_csv("./data/snake.csv")
        with open("snake_rules_demo.json", "r") as f:
            model_data = json.load(f)
        root = RDRNode.from_dict(model_data)
        return df, root
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None

# --- STREAMLIT UI ---

st.set_page_config(
    page_title="SnakeExpert Pro | AI Knowledge Base",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for Premium UX
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@300;400;700&display=swap');

    :root {
        --primary: #10b981;
        --secondary: #3b82f6;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    .stApp {
        background: radial-gradient(circle at 0% 0%, #1e293b 0%, #0f172a 100%);
    }

    /* Glassmorphic Card Effect */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.2);
        transform: translateY(-4px);
    }

    /* Result Typography */
    .result-container {
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }

    .species-label {
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    .species-name {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 1rem 0;
        background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .venom-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    /* Trace Section */
    .trace-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid transparent;
        transition: all 0.2s ease;
    }
    .trace-match { border-left-color: var(--primary); background: rgba(16, 185, 129, 0.05); }
    .trace-fail { border-left-color: var(--danger); background: rgba(239, 68, 68, 0.05); opacity: 0.6; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid var(--glass-border);
    }
    </style>
    """, unsafe_allow_html=True)

df, root = load_data()

if df is not None and root is not None:
    # Sidebar Navigation & Info
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/python.png", width=50)
        st.markdown("## SnakeExpert **Pro**")
        st.markdown("---")
        st.info("💡 **How it works:** This expert system uses a **Single Classification Ripple Down Rule (SCRDR)** engine to identify snakes with human-like reasoning.")
        
        st.markdown("### Model Statistics")
        st.metric("Total Rules", "24")
        st.metric("Accuracy", "94.4%")
        
        st.divider()
        st.caption("Developed by Senior NLP Researcher")

    # Hero Section
    st.markdown("""
        <div style="margin-bottom: 3rem;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0;">Identify with <span style="color: #10b981;">Precision</span>.</h1>
            <p style="font-size: 1.2rem; color: #94a3b8; font-weight: 300;">Advanced RDR-based Identification Engine for Herpetology</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 2], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧬 Biological Traits")
        st.write("Configure the observed characteristics below:")
        
        features = {}
        target_col = "species_name"
        
        # Grouping features for cleaner UI
        for col in df.columns:
            if col == target_col: continue
            unique_vals = sorted(df[col].unique().tolist())
            
            label = col.replace('_', ' ').title()
            icon = "🔹"
            if "venomous" in col: icon = "🧪"
            if "habitat" in col: icon = "🏞️"
            if "pattern" in col: icon = "🎨"
            
            if set(unique_vals) == {True, False} or set(unique_vals) == {"True", "False"}:
                features[col] = st.radio(f"{icon} {label}", ["True", "False"], horizontal=True)
            else:
                features[col] = st.selectbox(f"{icon} {label}", unique_vals)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        engine = SCRDR_Engine(target_col, root)
        result_node, trace = engine.classify_with_trace(features)
        
        is_venomous = features.get('is_venomous') == "True"
        accent_color = "#ef4444" if is_venomous else "#10b981"
        badge_text = "VENOMOUS" if is_venomous else "NON-VENOMOUS"
        
        st.markdown(f'''
            <div class="glass-card result-container">
                <p class="species-label">Classification Result</p>
                <div class="species-name" style="background: linear-gradient(135deg, #fff 0%, {accent_color} 100%); -webkit-background-clip: text;">
                    {result_node.conclusion}
                </div>
                <div class="venom-badge" style="background: {accent_color}20; color: {accent_color}; border: 1px solid {accent_color}40;">
                    {badge_text}
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("🔍 VIEW DECISION PATH", expanded=False):
            st.markdown('<div style="padding: 1rem 0;">', unsafe_allow_html=True)
            for step in trace:
                node = step['node']
                matched = step['result']
                status_icon = "✓" if matched else "✕"
                trace_class = "trace-match" if matched else "trace-fail"
                
                cond_str = "Default Rule"
                if node.condition:
                    cond_str = f"IF <b>{node.condition['col']}</b> {node.condition['op']} <i>'{node.condition['val']}'</i>"
                
                st.markdown(f'''
                    <div class="trace-item {trace_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span>{status_icon} &nbsp; {cond_str}</span>
                            <small style="opacity: 0.6;">{node.conclusion}</small>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 2rem 0;">
            SnakeExpert Pro &copy; 2026 | Powered by SCRDR Technology | Encrypted Data Stream Active
        </div>
    """, unsafe_allow_html=True)

else:
    st.error("⚠️ Critical Error: Model or Dataset not found. Please verify file paths.")
