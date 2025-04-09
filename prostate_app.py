import streamlit as st
import pandas as pd
import joblib
import requests

# Load model and encoders with error handling
try:
    model = joblib.load("prostate_cancer_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# Get Hugging Face API token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    st.error("Hugging Face token not found in secrets!")
    st.stop()

st.title("🧠 Prostate Cancer Predictor")
st.write("Enter the patient's clinical measurements below:")

# User input sliders
radius = st.slider("Radius", 5, 30, 15)
texture = st.slider("Texture", 5, 30, 18)
perimeter = st.slider("Perimeter", 50, 200, 105)
area = st.slider("Area", 100, 2000, 700)
smoothness = st.slider("Smoothness", 0.05, 0.2, 0.11)
compactness = st.slider("Compactness", 0.01, 0.5, 0.17)
symmetry = st.slider("Symmetry", 0.05, 0.4, 0.23)
fractal_dimension = st.slider("Fractal Dimension", 0.04, 0.1, 0.06)

if st.button("🔍 Predict and Explain"):
    # Step 1: Build input data
    new_patient = {
        "radius": radius,
        "texture": texture,
        "perimeter": perimeter,
        "area": area,
        "smoothness": smoothness,
        "compactness": compactness,
        "symmetry": symmetry,
        "fractal_dimension": fractal_dimension
    }
    input_df = pd.DataFrame([new_patient])

    # Step 2: Predict using your trained model
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()
        
    try:
        probability = model.predict_proba(input_df)[0][1] * 100
    except AttributeError:
        probability = 0  # Fallback if predict_proba is not supported
    except Exception as e:
        st.error(f"Error calculating probability: {e}")
        st.stop()
    
    try:
        label = label_encoders["diagnosis_result"].inverse_transform([prediction])[0]
    except Exception as e:
        st.error(f"Error decoding label: {e}")
        st.stop()

    st.subheader(f"Prediction: {label} ({probability:.1f}% chance of cancer)")

    # Step 3: Create a natural language prompt
    prompt = f"""
A patient presents with the following clinical indicators:
- Radius: {radius}
- Texture: {texture}
- Perimeter: {perimeter}
- Area: {area}
- Smoothness: {smoothness}
- Compactness: {compactness}
- Symmetry: {symmetry}
- Fractal Dimension: {fractal_dimension}

Based on a trained model, the probability of this patient having prostate cancer is {probability:.1f}%.

Please explain this result in simple terms for a medical student or concerned patient.
""".strip()

    # Step 4: Ask Hugging Face model to explain
    with st.spinner("🧠 Asking AI to explain the result..."):
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
                json={"inputs": prompt}
            )
            response.raise_for_status()  # Raise error for bad HTTP status
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                explanation = result[0]["generated_text"]
            else:
                explanation = result.get("error", "Unexpected response format")
        except Exception as e:
            explanation = f"Error during API call: {e}"

    st.markdown("### AI Explanation")
    st.write(explanation)
