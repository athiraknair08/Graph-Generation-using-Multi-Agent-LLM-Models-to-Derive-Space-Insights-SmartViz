# app.py

import streamlit as st
import os
import pandas as pd
import plotly.express as px
from chatbot_evaluator import log_and_score


from agents.query_agent import detect_intent_and_entities
from agents.graph_and_insight_agents import ( 
    generate_graph_from_query,
    generate_insight_from_data,
   
)

import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")


#Silence benign PyTorch extension warning
os.environ.setdefault("PYTORCH_JIT", "0")

st.set_page_config(page_title="SmartViz Chatbot", layout="wide")

st.markdown("""
    <style>
    .stChatMessage { margin-bottom: 1rem; }
    .stTextInput, .stTextArea { border-radius: 10px; }
    .element-container:has(.stPlotlyChart) {
        padding: 0.5rem 1rem;
        background: #f9f9f9;
        border-radius: 12px;
    }
    .element-container:has(.stChatMessage) {
        padding: 0.25rem 0;
    }
    .stTextArea textarea {
        background-color: #f0f0f0 !important;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <script>
    const streamlitDoc = window.parent.document;
    const chatDiv = streamlitDoc.querySelector('.block-container');
    chatDiv.scrollTo(0, chatDiv.scrollHeight);
    </script>
""", unsafe_allow_html=True)

st.title("🤖 SmartViz Chatbot")
st.markdown("Ask about **room metrics**, trends for graph and insights.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()



# Display chat history with unique keys
for i, entry in enumerate(st.session_state.chat_history):
    if entry["role"] == "user":
        st.chat_message("user").write(entry["content"])
    else:
        with st.chat_message("assistant"):
            if entry.get("graph"):
                fig = entry["graph"]
                graph_df = entry.get("graph_data", pd.DataFrame())
                chart_type = entry.get("chart_type", "time_series")

                # Only re-draw as time series when the generator says so
                if chart_type == "time_series" and \
                   ("start_time" in graph_df.columns and "metric_name" in graph_df.columns):

                    graph_df["start_time"] = pd.to_datetime(graph_df["start_time"], utc=True)

                    col1, col2 = st.columns(2)
                    smoothing = col1.toggle("Smoothed", value=True, key=f"smooth_{i}")
                    grouping = col2.radio("Group By", ["Hourly", "Daily"], horizontal=True, key=f"group_{i}")

                    df = graph_df.copy()
                    if grouping == "Daily":
                        df = (
                            df.set_index("start_time")
                              .groupby(["metric_name", pd.Grouper(freq="D")])["value"]
                              .mean()
                              .reset_index()
                        )
                    else:
                        df = (
                            df.set_index("start_time")
                              .groupby(["metric_name", pd.Grouper(freq="h")])["value"]
                              .mean()
                              .reset_index()
                        )

                    if smoothing:
                        df["value"] = df.groupby("metric_name")["value"]\
                                      .transform(lambda x: x.rolling(5, min_periods=1).mean())

                    fig = px.line(
                        df,
                        x="start_time",
                        y="value",
                        color="metric_name",
                        title="Metric Trends",
                        markers=True
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Value",
                        height=500,
                        margin=dict(t=50, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
                else:
                    # Respect the original figure (bar/pie/box/heatmap/scatter)
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")

            if entry.get("insight"):
                st.markdown(f"**💡 Insight:**\n\n{entry['insight']}")

# User input
user_query = st.chat_input("💬 Ask about building metrics...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    parsed = detect_intent_and_entities(user_query)
    print("[DEBUG] Parsed:", parsed)
    print("[DEBUG] User query:", user_query)

    parsed["original_query"] = user_query  # always carry original query forward

    # Always initialize these so they exist even if graph generation fails
    fig = None
    data_df = pd.DataFrame()
    chart_type = None

    response = {
        "role": "assistant",
        "intent": parsed.get("intent"),
        "metric": parsed.get("metric_name"),
        "graph": None,
        "graph_data": None,
        "chart_type": None,
        "insight": None,
        "content": user_query
    }

    # Generate the graph (and data for it)
    try:
        fig, data_df, chart_type = generate_graph_from_query(parsed)
    except ValueError:
        # Backward-compat if anything returns 2-tuple
        try:
            fig, data_df = generate_graph_from_query(parsed)
            chart_type = "time_series"
        except Exception as e:
            print(f"[DEBUG] Graph generation error: {e}")

    # --- Always call the insight function (even if df is empty) so fallback rebuild can run ---
    insight_text = generate_insight_from_data(
        data_df,
        user_query,
        start=parsed.get("start_date"),
        end=parsed.get("end_date"),
        metric_hint=(parsed.get("metric_name") or (parsed.get("metric_names") or [None])[0])
    )
    print("[DEBUG] Insight preview:", (insight_text or "")[:180])

    # ---> Accuracy evaluation (prints [EVAL] lines in terminal)
    fact_score = None
    try:
        # Prefer the same df used for the chart; if it's empty, evaluator returns None.
        eval_df = data_df if (isinstance(data_df, pd.DataFrame) and not data_df.empty) else pd.DataFrame()
        fact_score = log_and_score(user_query, eval_df, insight_text)
        print(f"[EVAL] Accuracy score for this query: {fact_score if fact_score is not None else 'n/a'}")
    except Exception as e:
        print(f"[EVAL] Scoring error: {e}")

    
    response["graph"] = fig if fig is not None else None
    response["graph_data"] = data_df if data_df is not None else pd.DataFrame()
    response["chart_type"] = chart_type
    response["insight"] = insight_text or "⚠️ No insight returned."
   
    if parsed.get("intent") not in ("graph", "insight"):
        response["insight"] = response["insight"] or "💡 Try asking for a graph or insight about a metric in a room."

    st.session_state.chat_history.append(response)
    st.rerun()
