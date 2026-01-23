import streamlit as st
import pandas as pd
import joblib

st.title("Exam Score Prediction App")

# Load trained model
model = joblib.load("final_elasticnet_model.pkl")

st.write("Enter student details below:")

RMSE = 3.25  # use your real RMSE from test results

Hours_Studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
Attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=90)

Parental_Involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
Access_to_Resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
Extracurricular_Activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
Sleep_Hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)

Previous_Scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=80)
Motivation_Level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
Internet_Access = st.selectbox("Internet Access", ["Yes", "No"])

Tutoring_Sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=20, value=2)
Family_Income = st.selectbox("Family Income", ["Low", "Medium", "High"])
Teacher_Quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])

School_Type = st.selectbox("School Type", ["Public", "Private"])
Peer_Influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])

Physical_Activity = st.number_input("Physical Activity", min_value=0, max_value=20, value=3)
Learning_Disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])

Parental_Education_Level = st.selectbox(
    "Parental Education Level",
    ["High School", "College", "Postgraduate"]
)
Distance_from_Home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

Gender = st.selectbox("Gender", ["Male", "Female"])

# Build input dataframe (always ready)
input_data = pd.DataFrame([{
    "Hours_Studied": Hours_Studied,
    "Attendance": Attendance,
    "Parental_Involvement": Parental_Involvement,
    "Access_to_Resources": Access_to_Resources,
    "Extracurricular_Activities": Extracurricular_Activities,
    "Sleep_Hours": Sleep_Hours,
    "Previous_Scores": Previous_Scores,
    "Motivation_Level": Motivation_Level,
    "Internet_Access": Internet_Access,
    "Tutoring_Sessions": Tutoring_Sessions,
    "Family_Income": Family_Income,
    "Teacher_Quality": Teacher_Quality,
    "School_Type": School_Type,
    "Peer_Influence": Peer_Influence,
    "Physical_Activity": Physical_Activity,
    "Learning_Disabilities": Learning_Disabilities,
    "Parental_Education_Level": Parental_Education_Level,
    "Distance_from_Home": Distance_from_Home,
    "Gender": Gender
}])

if st.button("Predict Exam Score"):
    pred = model.predict(input_data)[0]
    st.success(f"Predicted Exam Score: {pred:.2f}")

    # Estimated error range
    st.info(f"Estimated error range: ±{RMSE:.2f} points")

    # Confidence score (simple)
    confidence_score = max(0.0, min(1.0, 1 - (RMSE / 10)))

    if confidence_score >= 0.75:
        level = "High"
    elif confidence_score >= 0.5:
        level = "Moderate"
    else:
        level = "Low"

    st.write(f"Model Confidence: **{level}**")
    st.progress(confidence_score)
