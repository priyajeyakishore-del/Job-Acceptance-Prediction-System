
import joblib
import pandas as pd

from app.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_LABELS
)

from app.feature_engineering import prepare_model_input


# ============================================================
# 1. LOAD SAVED MODEL AND PREPROCESSING PIPELINE
# ============================================================

def load_model_and_preprocessor():
    """
    Load the saved final machine learning model
    and preprocessing pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


# ============================================================
# 2. PREPROCESS CANDIDATE INPUT
# ============================================================

def preprocess_candidate(candidate_data, preprocessor):
    """
    Prepare candidate input and transform it
    using the already-trained preprocessing pipeline.

    Important:
    Use transform() only.
    Never use fit() or fit_transform() on new candidates.
    """

    # Generate all 38 model input features
    model_input = prepare_model_input(candidate_data)

    # Apply the saved preprocessing pipeline
    processed_input = preprocessor.transform(model_input)

    # Convert to DataFrame for easier inspection
    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
        processed_input,
        columns=feature_names,
        index=model_input.index
    )

    return processed_df


# ============================================================
# 3. GENERATE CANDIDATE PREDICTION
# ============================================================

def predict_candidate(candidate_data):
    """
    Predict whether a candidate will be placed or not placed.

    Returns:
        Dictionary containing prediction and probabilities.
    """

    # Load saved model and preprocessor
    model, preprocessor = load_model_and_preprocessor()

    # Transform candidate input
    processed_input = preprocess_candidate(
        candidate_data,
        preprocessor
    )

    # Check transformed feature count
    expected_features = getattr(
        model, "n_features_in_", processed_input.shape[1]
    )

    if processed_input.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: expected {expected_features}, "
            f"received {processed_input.shape[1]}"
        )

    # Generate prediction
    prediction = int(model.predict(processed_input)[0])

    # Convert prediction into readable label
    prediction_label = TARGET_LABELS[prediction]

    # Generate class probabilities when supported
    probabilities = {}

    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(
            processed_input
        )[0]

        for class_label, probability in zip(
            model.classes_, probability_values
        ):
            class_label = int(class_label)

            probabilities[TARGET_LABELS[class_label]] = round(
                float(probability) * 100, 2
            )

    # Return prediction results
    result = {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probabilities": probabilities,
        "processed_features": processed_input.shape[1]
    }

    return result
