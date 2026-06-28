import streamlit as st
import pandas as pd

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ML Algorithm Comparison Framework",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
.main{
    background-color:#F8F9FA;
}
h1{
    color:#0A4D68;
}
.metric-card{
    background:#FFFFFF;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD RESULTS & DEFINE BEST MODEL (Updated Path)
# ---------------------------------------------------

# Fixed path to look inside the 'data' directory
results = pd.read_csv("data/results.csv")

best = results.sort_values(
    by="ROC-AUC",
    ascending=False
).iloc[0]

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🤖 ML Algorithm Comparison Framework")

st.markdown(
"""
Compare six supervised Machine Learning algorithms on the
Customer Churn dataset using standardized evaluation metrics.
"""
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Algorithms",
    "6"
)

col2.metric(
    "Records",
    "7043"
)

col3.metric(
    "Features",
    "20"
)

col4.metric(
    "Best Model",
    best["Model"]
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🤖 Dashboard")
page = st.sidebar.radio(
    "Go To",
    [
        "Dataset",
        "Results",
        "Visualizations",
        "Classification Reports",
        "Model Ranking"
    ]
)

st.sidebar.divider()
st.sidebar.success("Dataset : Customer Churn")
st.sidebar.info("Records : 7043")
st.sidebar.info("Features : 20")
st.sidebar.info("Target : Churn")

# ---------------------------------------------------
# HOME / DATASET
# ---------------------------------------------------

if page == "Dataset":

    st.header("Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Records",
        "7043"
    )

    col2.metric(
        "Features",
        "20"
    )

    col3.metric(
        "Models",
        "6"
    )

    col4.metric(
        "Best Model",
        best["Model"]
    )

    st.divider()

    st.subheader("Algorithms Compared")
    st.write("""
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes
""")

    st.subheader("Evaluation Metrics")
    st.write("""
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
""")

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------

elif page == "Results":

    st.header("Benchmark Results")

    display = results.copy()

    for col in display.columns[1:]:
        display[col] = (display[col] * 100).round(2)

    st.dataframe(
        display,
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Results",
        display.to_csv(index=False),
        file_name="results.csv"
    )
    
# ---------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------

elif page == "Visualizations":

    st.header("Model Performance Visualizations")

    tab1, tab2, tab3 = st.tabs([
        "Comparison Charts",
        "Confusion Matrix",
        "Feature Importance"
    ])

    with tab1:
        st.subheader("Accuracy Comparison")
        st.image("assets/accuracy_comparison.png", use_container_width=True)

        st.subheader("ROC-AUC Comparison")
        st.image("assets/roc_auc_comparison.png", use_container_width=True)

        st.subheader("Precision vs Recall")
        st.image("assets/precision_recall.png", use_container_width=True)

        st.subheader("ROC Curve")
        st.image("assets/roc_curve_all_models.png", use_container_width=True)

    with tab2:
        st.subheader("Confusion Matrix")
        st.image("assets/confusion_matrix.png", use_container_width=True)

    with tab3:
        st.subheader("Feature Importance")
        st.image("assets/feature_importance.png", use_container_width=True)
        
# ---------------------------------------------------
# CLASSIFICATION REPORTS
# ---------------------------------------------------

elif page == "Classification Reports":

    st.header("Classification Reports")

    reports = {
        "Logistic Regression": "classification_reports/logistic_regression.txt",
        "Decision Tree": "classification_reports/decision_tree.txt",
        "Random Forest": "classification_reports/random_forest.txt",
        "KNN": "classification_reports/knn.txt",
        "SVM": "classification_reports/svm.txt",
        "Naive Bayes": "classification_reports/naive_bayes.txt"
    }

    choice = st.selectbox(
        "Select Model",
        list(reports.keys())
    )

    with open(reports[choice], "r") as f:
        report = f.read()

    st.text_area(
        "Classification Report",
        report,
        height=350
    )

# ---------------------------------------------------
# MODEL RANKING
# ---------------------------------------------------

elif page == "Model Ranking":

    st.header("Model Ranking")

    ranking = results.sort_values(
        by="ROC-AUC",
        ascending=False
    ).reset_index(drop=True)

    ranking.index += 1

    st.dataframe(
        ranking,
        use_container_width=True
    )

    st.success(
        f"🏆 Best Model : {best['Model']}"
    )

    m_col1, m_col2 = st.columns(2)
    
    m_col1.metric(
        "ROC-AUC",
        round(best["ROC-AUC"], 3)
    )

    m_col2.metric(
        "Accuracy",
        f"{best['Accuracy']*100:.2f}%"
    )
    
st.divider()

st.caption(
"""
Developed using Python, Scikit-learn and Streamlit.
Machine Learning Algorithm Comparison Framework
"""
)