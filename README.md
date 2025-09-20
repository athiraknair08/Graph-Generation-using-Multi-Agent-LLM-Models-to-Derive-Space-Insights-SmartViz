# Graph-Generation-using-Multi-Agent-LLM-Models-to-Derive-Space-Insights-SmartViz
This repository contains the implementation of a multi-agent chatbot system developed as part of my MSc Dissertation project at Aston University. The system enables natural language queries to be translated into SQL, generate appropriate graphs/visualizations, and produce contextual insights using Large Language Models (LLMs).

The project integrates PostgreSQL, LangChain, Ollama (Mistral), and Plotly into a seamless pipeline designed for smart building analytics.

*****Features*****

-> Query Agent – Extracts intent and entities from user input and generates SQL queries.

-> Graph Agent – Executes SQL queries on PostgreSQL, dynamically selects chart types, and generates interactive visualizations (line, bar, scatter, heatmap, box, pie).

-> Insight Agent – Uses LLMs (Ollama Mistral) to interpret graphs and generate natural language insights.

-> Semantic Matching – Sentence-transformers for robust room/metric matching.

-> RAG (Retrieval-Augmented Generation) – Grounds insights in database outputs to reduce hallucination.

-> Evaluation Framework – Integrated with TruLens and DeepEval for assessing correctness, relevance, and factuality.

-> Web Interface – Built with Streamlit for an interactive chatbot experience.
```
SMARTVIZ_CHATBOT_PROJECT/
│── agents/
│ ├── query_agent.py # Query Agent: intent & entity recognition
│ ├── graph_and_insight_agents.py # Graph Agent (SQL + visualization) + Insight Agent
│── data/
│ ├── metrics_app_merged_cleaned.csv
│ ├── metrics_app_hierarchy_.csv
│ ├── metrics_app_timeaggregated_.csv
│── notebooks/
│ ├── eda.ipynb # Exploratory Data Analysis + cleaning + preprocessing
│── app.py # Streamlit app entrypoint
│── chatbot_evaluator.py # Accuracy evaluation
│── db.py # Database connection setup
│── create_merged_table.py # Load + merge data into Postgres
│── cleaned_merged_data.csv # Preprocessed dataset
│── default.sqlite # Local fallback DB
│── .env # Environment variables
│── venv/ # Virtual environment
```
*****Tech Stack*****
```
Programming: Python

Frameworks: LangChain, Streamlit

Database: PostgreSQL (pgAdmin 4), SQLAlchemy ORM

Machine Learning / NLP:

Ollama (Mistral) – local LLM

Sentence-Transformers (all-MiniLM-L6-v2) – embeddings for semantic matching

Dateparser – natural language date parsing

Visualization: Plotly, Matplotlib (fallback)

Evaluation: TruLens, DeepEval

Utilities: Pandas, NumPy, Regex
```
*****Example Queries*****

Simple Query
“Hottest room right now?” → Bar chart + insight

Complex Query
“Rooms with continuous high CO₂ ordered by worst offenders” → Heatmap + ranked insight

Date-Range Query
“Plot bar graph of top rooms with average occupancy between Jan–Mar 2024” → Aggregated bar chart + insight


*****Setup Instructions*****

Clone the repo
```
git clone https://github.com/<your-username>/smartviz-llm-graph.git
cd smartviz-llm-graph


Install dependencies

pip install -r requirements.txt


Setup PostgreSQL

Create a database (e.g., smartviz_db)

Import metrics_app_merged_cleaned.csv into a table (metrics_app_merged).

Run Ollama (Mistral) locally

ollama pull mistral


Launch the chatbot

streamlit run app.py
```
*****Screenshots*****

(Add images of chatbot interface, sample graphs, and system architecture here.)

*****Dissertation Context*****

This project was developed as part of the MSc Data Science dissertation at Aston University, supervised by Vishal Sharma.

It demonstrates an end-to-end multi-agent system for turning natural language queries into actionable visual and textual insights, tailored to smart building datasets.

*****Future Work*****

Extend evaluation with domain expert user studies.

Apply system to other domains (healthcare, finance, transport).

Integrate predictive analytics and anomaly detection.

Enhance real-time dashboarding capabilities.

*****License*****

This project is proprietary and all rights are reserved.
No part of this codebase may be used, copied, modified, or distributed without explicit permission from the author.
