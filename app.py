
import os
import io
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageOps


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROJECT PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "CNN": os.path.join(BASE_DIR, "cnn_results", "best_cnn.keras"),

    "ResNet50": os.path.join(
        BASE_DIR, "resnet50_improved_results",
        "best_resnet50_finetuned.keras"
    ),

    "VGG16": os.path.join(
        BASE_DIR, "vgg16_lightweight_results",
        "best_vgg16.keras"
    ),

    "EfficientNetB0": os.path.join(
        BASE_DIR, "efficientnetb0_corrected_results",
        "best_efficientnetb0.keras"
    ),

    "SVM": os.path.join(
        BASE_DIR, "svm_results", "svm_model.pkl"
    ),

    "Random Forest": os.path.join(
        BASE_DIR, "random_forest_results",
        "random_forest_model.pkl"
    )
}

SCALER_PATH = os.path.join(
    BASE_DIR, "processed_data", "feature_scaler.pkl"
)

# MODEL METRICS

MODEL_METRICS = {
    "CNN": {
        "Accuracy": 77.35,
        "Precision": 74.36,
        "Recall": 97.42,
        "F1 Score": 84.34
    },

    "ResNet50": {
        "Accuracy": 62.62,
        "Precision": 62.62,
        "Recall": 100.00,
        "F1 Score": 77.01
    },

    "VGG16": {
        "Accuracy": 78.48,
        "Precision": 75.00,
        "Recall": 98.45,
        "F1 Score": 85.14
    },

    "EfficientNetB0": {
        "Accuracy": 86.89,
        "Precision": 84.62,
        "Recall": 96.64,
        "F1 Score": 90.23
    },

    "SVM": {
        "Accuracy": 76.70,
        "Precision": 73.59,
        "Recall": 97.93,
        "F1 Score": 84.04
    },

    "Random Forest": {
        "Accuracy": 81.55,
        "Precision": 78.14,
        "Recall": 97.93,
        "F1 Score": 86.93
    }
}

# PREMIUM DARK ANIMATED THEME

st.markdown("""
<style>

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:
        radial-gradient(ellipse at 85% 85%,
        rgba(76,29,149,0.25), transparent 45%),
        radial-gradient(ellipse at 10% 10%,
        rgba(8,145,178,0.15), transparent 40%),
        #080d1b !important;
    color: #e2e8f0 !important;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* HEADER */

header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: #080d1b !important;
    color: #e2e8f0 !important;
}

[data-testid="stHeader"] {
    border-bottom: 1px solid #263b60 !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        160deg, #080d1b, #111c33, #172554
    ) !important;
    border-right: 1px solid #263b60;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* HEADINGS */

h1 {
    background: linear-gradient(
        90deg, #22d3ee, #818cf8, #c084fc, #22d3ee
    );
    background-size: 300% auto;
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 850 !important;
    animation: neonFlow 5s linear infinite;
}

h2, h3 {
    color: #c4b5fd !important;
}

p, label {
    color: #cbd5e1 !important;
}

/* METRICS */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #152341, #10182c);
    border: 1px solid #334e75;
    border-radius: 17px;
    padding: 20px;
    transition: 0.3s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #22d3ee;
    box-shadow: 0 0 25px rgba(34,211,238,0.25);
    transform: translateY(-5px);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #67e8f9 !important;
    font-weight: 850 !important;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(
        100deg, #0891b2, #4f46e5, #7c3aed
    ) !important;
    background-size: 200% 200%;
    color: white !important;
    border: 1px solid #67e8f9;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 750;
    transition: all 0.3s ease;
    animation: buttonGlow 4s ease infinite;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 25px rgba(34,211,238,0.4);
}

/* UPLOADER */

[data-testid="stFileUploader"] {
    background: #111c33 !important;
    border: 2px dashed #22d3ee !important;
    border-radius: 18px;
    padding: 15px;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

/* SELECTBOX */

[data-baseweb="select"] > div {
    background: #111c33 !important;
    border-color: #334e75 !important;
    color: #e2e8f0 !important;
}

/* IMAGES */

[data-testid="stImage"] img {
    border-radius: 17px;
    border: 1px solid #22d3ee;
    box-shadow: 0 0 20px rgba(34,211,238,0.15);
}

/* ALERTS */

[data-testid="stAlert"] {
    background: #152341 !important;
    border: 1px solid #334e75;
    border-radius: 13px;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #334e75;
    border-radius: 14px;
}

/* DIVIDER */

hr {
    border-color: #263b60 !important;
}

/* HERO */

.hero {
    background: linear-gradient(
        120deg, #101b35, #172554, #312e81, #101b35
    );
    background-size: 300% 300%;
    animation: gradientMove 8s ease infinite;
    border: 1px solid #334b78;
    border-radius: 22px;
    padding: 45px 25px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 0 30px rgba(34,211,238,0.12);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 12px;
}

.hero p {
    color: #cbd5e1 !important;
    font-size: 17px;
}

.tag {
    display: inline-block;
    color: #67e8f9;
    background: rgba(34,211,238,0.10);
    border: 1px solid rgba(34,211,238,0.35);
    padding: 7px 16px;
    border-radius: 30px;
    margin-bottom: 16px;
    font-size: 13px;
}

/* MODEL CARDS */

.model-card {
    background: linear-gradient(145deg, #111c33, #172541);
    border: 1px solid #293e62;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 12px;
    min-height: 115px;
    transition: 0.3s ease;
}

.model-card:hover {
    transform: translateY(-5px);
    border-color: #818cf8;
    box-shadow: 0 0 18px rgba(129,140,248,0.15);
}

.model-name {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}

.model-type {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 8px;
}

.model-accuracy {
    color: #67e8f9;
    font-size: 21px;
    font-weight: 800;
    margin-top: 10px;
}

/* FOOTER */

.footer-box {
    background: #111c33;
    border: 1px solid #293e62;
    border-radius: 13px;
    padding: 18px;
    color: #94a3b8;
    text-align: center;
    margin-top: 25px;
    font-size: 13px;
}

/* ANIMATIONS */

@keyframes neonFlow {
    0% {background-position: 0% center;}
    100% {background-position: 300% center;}
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

@keyframes buttonGlow {
    0%, 100% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
}

/* SCROLLBAR */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #080d1b;
}

::-webkit-scrollbar-thumb {
    background: #334e75;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #22d3ee;
}

</style>
""", unsafe_allow_html=True)

# MODEL LOADING

@st.cache_resource
def load_keras_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


@st.cache_resource
def load_ml_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


@st.cache_resource
def load_scaler():

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    return joblib.load(SCALER_PATH)

# IMAGE PREPROCESSING

def prepare_image(image, image_size):

    image = image.convert("RGB")
    image = image.resize((image_size, image_size))

    image_array = np.array(
        image, dtype=np.float32
    ) / 255.0

    return np.expand_dims(image_array, axis=0)


def prepare_ml_features(image):

    image = image.convert("L")
    image = image.resize((64, 64))

    image_array = np.array(
        image, dtype=np.float32
    ) / 255.0

    features = image_array.flatten().reshape(1, -1)

    scaler = load_scaler()

    return scaler.transform(features)

# DEEP LEARNING PREDICTION

def predict_deep_learning(model_name, image):

    model_path = MODEL_PATHS[model_name]
    model = load_keras_model(model_path)

    image_size = 224 if model_name == "CNN" else 128

    processed_image = prepare_image(
        image, image_size
    )

    prediction = np.asarray(
        model.predict(processed_image, verbose=0)
    )

    if prediction.size == 1:

        pneumonia_probability = float(
            prediction.reshape(-1)[0]
        )

    elif prediction.shape[-1] == 2:

        pneumonia_probability = float(
            prediction[0][1]
        )

    else:

        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    pneumonia_probability = float(
        np.clip(pneumonia_probability, 0.0, 1.0)
    )

    result = (
        "PNEUMONIA"
        if pneumonia_probability >= 0.5
        else "NORMAL"
    )

    confidence = max(
        pneumonia_probability,
        1 - pneumonia_probability
    ) * 100

    return result, confidence, pneumonia_probability

# MACHINE LEARNING PREDICTION

def predict_machine_learning(model_name, image):

    model = load_ml_model(MODEL_PATHS[model_name])

    features = prepare_ml_features(image)

    prediction = model.predict(features)

    predicted_class = prediction[0]

    result = (
        "PNEUMONIA"
        if predicted_class == 1
        else "NORMAL"
    )

    confidence = None
    pneumonia_probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        if 1 in classes:
            pneumonia_probability = float(
                probabilities[classes.index(1)]
            )
        else:
            pneumonia_probability = 0.0

        confidence = float(np.max(probabilities)) * 100

    return result, confidence, pneumonia_probability

# SIDEBAR NAVIGATION


PAGES = [
    "Dashboard",
    "X-Ray Prediction",
    "Model Comparison",
    "About Project"
]

# Handle dashboard quick-navigation buttons
if "nav_target" in st.session_state:
    st.session_state["navigation"] = st.session_state.pop(
        "nav_target"
    )

st.sidebar.markdown(
    "<h2 style='color:#67e8f9 !important;'>🩺 Pneumonia AI</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Deep Learning Based Chest X-Ray Analysis"
)

page = st.sidebar.radio(
    "Navigation",
    PAGES,
    key="navigation"
)

# DASHBOARD

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="tag">
            ARTIFICIAL INTELLIGENCE • DEEP LEARNING
        </div>
        <h1>🫁 Pneumonia Detection AI</h1>
        <p>
            Intelligent Chest X-Ray Analysis Using
            Deep Learning & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Models", "06")
    col2.metric("Best Recorded Accuracy", "86.89%")
    col3.metric("Classification Classes", "02")

    st.divider()

    st.subheader("🤖 AI Model Collection")

    models = [
        ("CNN", "Deep Learning", "77.35%"),
        ("ResNet50", "Transfer Learning", "62.62%"),
        ("VGG16", "Transfer Learning", "78.48%"),
        ("EfficientNetB0", "Transfer Learning", "86.89%"),
        ("SVM", "Machine Learning", "76.70%"),
        ("Random Forest", "Machine Learning", "81.55%")
    ]

    for start in range(0, len(models), 3):

        columns = st.columns(3)

        for col, model in zip(
            columns, models[start:start + 3]
        ):

            name, model_type, accuracy = model

            with col:

                st.markdown(f"""
                <div class="model-card">
                    <div class="model-name">⚡ {name}</div>
                    <div class="model-type">{model_type}</div>
                    <div class="model-accuracy">{accuracy}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    st.subheader("🚀 Quick Actions")

    action1, action2 = st.columns(2)

    with action1:

        if st.button(
            "🩻 Open X-Ray Prediction",
            use_container_width=True
        ):

            st.session_state["nav_target"] = "X-Ray Prediction"
            st.rerun()

    with action2:

        if st.button(
            "📈 Compare AI Models",
            use_container_width=True
        ):

            st.session_state["nav_target"] = "Model Comparison"
            st.rerun()

    st.markdown("""
    <div class="footer-box">
        <b>About This Project</b><br><br>
        An academic AI project for pneumonia classification
        using chest X-ray images.<br><br>
        This system is intended for educational and research
        purposes only, not for medical diagnosis.<br><br>
        © 2026 | Pneumonia Detection AI
    </div>
    """, unsafe_allow_html=True)

# X-RAY PREDICTION

elif page == "X-Ray Prediction":

    st.title("🩻 X-Ray Prediction")

    st.write(
        "Upload a chest X-ray image and select an AI model."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Upload X-Ray")

        uploaded_file = st.file_uploader(
            "Choose an X-Ray image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="xray_upload"
        )

        uploaded_image = None

        if uploaded_file is not None:

            try:

                image_bytes = uploaded_file.getvalue()

                uploaded_image = Image.open(
                    io.BytesIO(image_bytes)
                )

                uploaded_image = ImageOps.exif_transpose(
                    uploaded_image
                ).convert("RGB")

                st.success("Image uploaded successfully!")

            except Exception as e:

                st.error("Image could not be loaded.")
                st.exception(e)

        selected_model = st.selectbox(
            "Select AI Model",
            [
                "EfficientNetB0",
                "CNN",
                "VGG16",
                "ResNet50",
                "SVM",
                "Random Forest"
            ]
        )

        analyze_button = st.button(
            "🔍 Analyze X-Ray",
            use_container_width=True
        )

    with col2:

        st.subheader("Image Preview")

        if uploaded_image is not None:

            st.image(
                uploaded_image,
                caption="Uploaded Chest X-Ray",
                use_container_width=True
            )

        else:

            st.info(
                "Upload an X-Ray image to see its preview."
            )

    st.divider()

    if analyze_button:

        if uploaded_image is None:

            st.error("Please upload an X-Ray image first.")

        else:

            try:

                with st.spinner(
                    f"Analyzing image using {selected_model}..."
                ):

                    if selected_model in [
                        "CNN",
                        "ResNet50",
                        "VGG16",
                        "EfficientNetB0"
                    ]:

                        result, confidence, probability = (
                            predict_deep_learning(
                                selected_model,
                                uploaded_image
                            )
                        )

                    else:

                        result, confidence, probability = (
                            predict_machine_learning(
                                selected_model,
                                uploaded_image
                            )
                        )

                st.subheader("Prediction Result")

                if result == "PNEUMONIA":

                    st.error("⚠️ Prediction: PNEUMONIA")

                else:

                    st.success("✅ Prediction: NORMAL")

                result_col1, result_col2 = st.columns(2)

                result_col1.metric(
                    "Selected Model",
                    selected_model
                )

                if confidence is not None:

                    result_col2.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                if probability is not None:

                    st.write("Pneumonia Probability")

                    st.progress(
                        float(np.clip(probability, 0, 1))
                    )

                    st.write(
                        f"{probability * 100:.2f}%"
                    )

            except Exception as e:

                st.error(
                    "Prediction failed. Check model files "
                    "and preprocessing compatibility."
                )

                st.exception(e)

    st.warning(
        "This application is for educational and research "
        "purposes only. It is not a medical diagnosis tool."
    )

# MODEL COMPARISON

elif page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    st.write(
        "Comparison of previously recorded test-set results."
    )

    metrics_df = pd.DataFrame(
        MODEL_METRICS
    ).T

    st.subheader("Performance Metrics")

    st.dataframe(
        metrics_df.style.format("{:.2f}%"),
        use_container_width=True
    )

    st.divider()

    st.subheader("Accuracy Comparison")

    accuracy_df = metrics_df[
        ["Accuracy"]
    ].sort_values(
        "Accuracy",
        ascending=False
    )

    st.bar_chart(accuracy_df)

    st.subheader("Precision Comparison")

    st.bar_chart(
        metrics_df[["Precision"]].sort_values(
            "Precision", ascending=False
        )
    )

    st.subheader("Recall Comparison")

    st.bar_chart(
        metrics_df[["Recall"]].sort_values(
            "Recall", ascending=False
        )
    )

    st.subheader("F1 Score Comparison")

    st.bar_chart(
        metrics_df[["F1 Score"]].sort_values(
            "F1 Score", ascending=False
        )
    )

    best_model = metrics_df["Accuracy"].idxmax()

    st.success(
        f"Highest recorded accuracy: {best_model} "
        f"({metrics_df.loc[best_model, 'Accuracy']:.2f}%)"
    )

    st.warning(
        "ResNet50 previously predicted all test images as "
        "PNEUMONIA. Its NORMAL-class recognition needs improvement."
    )

    st.info(
        "These are historical test results, not the accuracy "
        "or confidence of an individual uploaded X-ray."
    )

# ABOUT PROJECT

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
    ### Project Title

    **Pneumonia Detection Using Chest X-Ray Images**

    ### Project Overview

    This academic project explores machine learning and
    deep learning techniques for classifying chest X-ray
    images into NORMAL and PNEUMONIA categories.

    ### Implemented Models

    **Deep Learning**
    - CNN
    - ResNet50
    - VGG16
    - EfficientNetB0

    **Machine Learning**
    - Support Vector Machine (SVM)
    - Random Forest

    ### Classification Classes

    - NORMAL
    - PNEUMONIA

    ### Application Features

    - Chest X-ray image upload
    - AI-based image prediction
    - Model selection
    - Prediction confidence where supported
    - Model performance comparison
    - Interactive dashboard

    ### Technologies

    - Python
    - TensorFlow / Keras
    - Scikit-learn
    - Streamlit
    - NumPy
    - Pandas
    - Pillow
    """)

    st.divider()

    st.warning(
        "This project is developed for academic and research "
        "purposes. Predictions must not be used as a substitute "
        "for professional medical advice or diagnosis."
    )

# SIDEBAR FOOTER

st.sidebar.divider()

st.sidebar.caption(
    "🩺 Pneumonia Detection AI"
)

st.sidebar.caption(
    "Academic Deep Learning Project | 2026"
)
