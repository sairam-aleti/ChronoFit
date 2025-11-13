import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
import re
import os
import threading
from typing import Dict, List, Any, Tuple
from mongodb_handler import get_mongodb_handler

# --- CONFIGURATION ---
# API key is now loaded from Streamlit secrets (see .streamlit/secrets.toml)
# If secrets not available, fall back to environment variable or show error
try:
    USDA_API_KEY = st.secrets.get("USDA_API_KEY", "")
except:
    USDA_API_KEY = os.getenv("USDA_API_KEY", "")

# --- MODEL ARTIFACTS ---
MODEL_PATH = 'mvva_model_v2.joblib'
PREPROCESSOR_PATH = 'mvva_preprocessor_v2.joblib'
GOAL_CLASSIFIER_PATH = 'mvva_goal_classifier_v2.joblib'
GOAL_ENCODER_PATH = 'mvva_goal_encoder_v2.joblib'

# --- API ENDPOINT CONSTANTS (Phase 1) ---
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
IND_NUTRIENT_BASE_URL = "https://indnutrientsapi.tech/api/nutrients"

# --- EXERCISE DATABASE ---
EXERCISE_DATABASE = {
    'Strength': {
        'Chest': [
            {'name': 'Bench Press', 'base_weight_ratio': 0.75, 'reps_range': (6, 8)},
            {'name': 'Incline Dumbbell Press', 'base_weight_ratio': 0.35, 'reps_range': (8, 10)},
            {'name': 'Push-ups', 'base_weight_ratio': 0.0, 'reps_range': (12, 15)},
            {'name': 'Cable Flyes', 'base_weight_ratio': 0.15, 'reps_range': (10, 12)}
        ],
        'Back': [
            {'name': 'Deadlift', 'base_weight_ratio': 1.0, 'reps_range': (5, 6)},
            {'name': 'Barbell Row', 'base_weight_ratio': 0.6, 'reps_range': (6, 8)},
            {'name': 'Pull-ups', 'base_weight_ratio': 0.0, 'reps_range': (8, 12)},
            {'name': 'Lat Pulldown', 'base_weight_ratio': 0.4, 'reps_range': (8, 10)}
        ],
        'Shoulders': [
            {'name': 'Overhead Press', 'base_weight_ratio': 0.5, 'reps_range': (6, 8)},
            {'name': 'Lateral Raises', 'base_weight_ratio': 0.1, 'reps_range': (10, 12)},
            {'name': 'Rear Delt Flyes', 'base_weight_ratio': 0.08, 'reps_range': (12, 15)}
        ],
        'Legs': [
            {'name': 'Squat', 'base_weight_ratio': 0.9, 'reps_range': (6, 8)},
            {'name': 'Leg Press', 'base_weight_ratio': 1.2, 'reps_range': (8, 10)},
            {'name': 'Romanian Deadlift', 'base_weight_ratio': 0.7, 'reps_range': (8, 10)},
            {'name': 'Leg Curls', 'base_weight_ratio': 0.25, 'reps_range': (10, 12)}
        ],
        'Arms': [
            {'name': 'Barbell Curl', 'base_weight_ratio': 0.2, 'reps_range': (8, 10)},
            {'name': 'Tricep Dips', 'base_weight_ratio': 0.0, 'reps_range': (10, 12)},
            {'name': 'Hammer Curls', 'base_weight_ratio': 0.15, 'reps_range': (10, 12)}
        ]
    },
    'Endurance': {
        'Full Body': [
            {'name': 'Circuit Training', 'base_weight_ratio': 0.0, 'reps_range': (15, 20)},
            {'name': 'Bodyweight Squats', 'base_weight_ratio': 0.0, 'reps_range': (20, 25)},
            {'name': 'Burpees', 'base_weight_ratio': 0.0, 'reps_range': (10, 15)},
            {'name': 'Mountain Climbers', 'base_weight_ratio': 0.0, 'reps_range': (30, 40)}
        ]
    },
    'Maintenance': {
        'Full Body': [
            {'name': 'Moderate Weight Training', 'base_weight_ratio': 0.4, 'reps_range': (10, 12)},
            {'name': 'Bodyweight Exercises', 'base_weight_ratio': 0.0, 'reps_range': (12, 15)},
            {'name': 'Light Resistance', 'base_weight_ratio': 0.2, 'reps_range': (12, 15)}
        ]
    },
    'Yoga': {
        'Full Body': [
            {'name': 'Downward Dog', 'base_weight_ratio': 0.0, 'reps_range': (1, 5)},
            {'name': 'Warrior Pose', 'base_weight_ratio': 0.0, 'reps_range': (1, 5)},
            {'name': 'Cat-Cow Stretch', 'base_weight_ratio': 0.0, 'reps_range': (5, 10)},
            {'name': 'Child\'s Pose', 'base_weight_ratio': 0.0, 'reps_range': (1, 3)},
            {'name': 'Pigeon Pose', 'base_weight_ratio': 0.0, 'reps_range': (1, 3)},
            {'name': 'Mountain Pose', 'base_weight_ratio': 0.0, 'reps_range': (1, 5)},
            {'name': 'Tree Pose', 'base_weight_ratio': 0.0, 'reps_range': (1, 3)}
        ]
    }
}

# Muscle group priority based on goal
MUSCLE_GROUP_PRIORITY = {
    'Strength': ['Chest', 'Back', 'Legs', 'Shoulders', 'Arms'],
    'Endurance': ['Full Body'],
    'Maintenance': ['Full Body']
}

# --- CORE FUNCTIONS ---

@st.cache_resource
def load_ml_artifacts():
    """Loads the trained model, preprocessor, goal classifier, and encoder. Waits for complete load."""
    try:
        # Check if all model files exist before attempting to load
        if not all(os.path.exists(path) for path in [MODEL_PATH, PREPROCESSOR_PATH, GOAL_CLASSIFIER_PATH, GOAL_ENCODER_PATH]):
            missing = [path for path in [MODEL_PATH, PREPROCESSOR_PATH, GOAL_CLASSIFIER_PATH, GOAL_ENCODER_PATH] if not os.path.exists(path)]
            raise FileNotFoundError(f"Missing model files: {missing}")
        
        # Load all models (complete load, no partial)
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        goal_classifier = joblib.load(GOAL_CLASSIFIER_PATH)
        goal_encoder = joblib.load(GOAL_ENCODER_PATH)
        
        # Verify all models loaded successfully (not None or corrupted)
        if any(x is None for x in [model, preprocessor, goal_classifier, goal_encoder]):
            raise ValueError("One or more models failed to load completely")
        
        return model, preprocessor, goal_classifier, goal_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading ML artifacts: {e}")
        st.error("Please ensure all model files are in the correct directory and run the notebook to generate them.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading ML artifacts: {e}")
        return None, None, None, None

def check_and_retrain_model():
    """
    Checks if new feedback is available in MongoDB and retrains the model if 3+ new feedbacks exist.
    Runs automatically on app startup - completely silent, no UI messages.
    """
    try:
        # Get MongoDB handler
        mongo_handler = get_mongodb_handler()
        
        if not mongo_handler.is_connected():
            # MongoDB not available, skip retraining
            return
        
        # Get feedback counts
        last_retrain_count = mongo_handler.get_last_retrain_count()
        current_count = mongo_handler.get_feedback_count()
        
        # Trigger retrain if 3+ new feedbacks
        if current_count >= last_retrain_count + 3:
            try:
                # Get all feedback from MongoDB
                feedback_df = mongo_handler.get_all_feedback()
                
                if feedback_df.empty or len(feedback_df) < 3:
                    return
                
                # Silent retraining - no UI output
                retrain_and_update_models(feedback_df, current_count)
                
                # Update metadata in MongoDB
                mongo_handler.update_last_retrain_count(current_count)
                
                # Clear model cache to load updated models
                st.cache_resource.clear()
                
            except Exception as e:
                # Silent fail - don't interrupt app with retraining errors
                pass
    except Exception as e:
        # Silent fail if MongoDB not available
        pass

def retrain_and_update_models(feedback_df, current_feedback_count):
    """Internal function to retrain models with user feedback from MongoDB. Runs silently."""
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    np.random.seed(42)
    NUM_RECORDS = 20000
    
    # Synthetic data generation (abbreviated)
    AGE = np.concatenate([
        np.random.normal(28, 6, int(NUM_RECORDS * 0.6)).clip(18, 40).astype(int),
        np.random.normal(48, 8, int(NUM_RECORDS * 0.4)).clip(40, 70).astype(int)
    ])
    np.random.shuffle(AGE)
    AGE = AGE[:NUM_RECORDS]
    
    SEX = np.random.choice(['M', 'F'], size=NUM_RECORDS, p=[0.6, 0.4])
    WEIGHT_KG = np.random.normal(78, 14, size=NUM_RECORDS).clip(45, 130).round(1)
    
    SLEEP_HRS = np.concatenate([
        np.random.normal(7.3, 0.8, int(NUM_RECORDS * 0.75)),
        np.random.normal(5.5, 0.6, int(NUM_RECORDS * 0.25))
    ]).clip(4.0, 10.0).round(1)
    np.random.shuffle(SLEEP_HRS)
    
    RHR_BASE = 62 + (WEIGHT_KG - 75) * 0.15
    RHR_BPM = (RHR_BASE + np.random.normal(0, 6, size=NUM_RECORDS)).clip(45, 95).round(0)
    
    SORENESS = np.concatenate([
        np.random.choice([1, 2], size=int(NUM_RECORDS * 0.7), p=[0.7, 0.3]),
        np.random.choice([3, 4, 5], size=int(NUM_RECORDS * 0.3), p=[0.4, 0.4, 0.2])
    ])
    np.random.shuffle(SORENESS)
    
    MENTAL_STRESS = np.concatenate([
        np.random.choice([1, 2], size=int(NUM_RECORDS * 0.5), p=[0.6, 0.4]),
        np.random.choice([3, 4, 5], size=int(NUM_RECORDS * 0.5), p=[0.3, 0.4, 0.3])
    ])
    np.random.shuffle(MENTAL_STRESS)
    
    CALORIES = np.random.normal(2200, 400, NUM_RECORDS).clip(1200, 4000).astype(int)
    PROTEIN_G = np.random.normal(120, 30, NUM_RECORDS).clip(40, 250).astype(int)
    CARBS_G = np.random.normal(250, 60, NUM_RECORDS).clip(100, 500).astype(int)
    NUTR_CONF_SCORE = np.random.uniform(0.5, 1.0, NUM_RECORDS)
    
    GOAL_LABELS = ['Strength', 'Endurance', 'Maintenance', 'Yoga']
    GOAL = np.random.choice(GOAL_LABELS, size=NUM_RECORDS, p=[0.4, 0.25, 0.25, 0.1])
    
    df = pd.DataFrame({
        'Age': AGE, 'Sex': SEX, 'Weight_kg': WEIGHT_KG,
        'SLEEP_HRS': SLEEP_HRS, 'RHR_BPM': RHR_BPM,
        'SORENESS': SORENESS, 'MENTAL_STRESS': MENTAL_STRESS,
        'CALORIES_IN': CALORIES, 'PROTEIN_G': PROTEIN_G,
        'CARBS_G': CARBS_G, 'NUTR_CONF_SCORE': NUTR_CONF_SCORE,
        'GOAL': GOAL
    })
    
    # Process user feedback from MongoDB
    synthetic_feedback_data = []
    
    if not feedback_df.empty:
        for idx, row in feedback_df.iterrows():
            completion_pct = row.get('workout_completion_pct', 100)
            recovery_feeling = row.get('recovery_feeling', 3)
            
            if completion_pct >= 90 and recovery_feeling >= 4:
                duration_adjustment = 1.10
                intensity_adjustment = 1.08
            elif completion_pct >= 80:
                duration_adjustment = 1.05
                intensity_adjustment = 1.03
            elif completion_pct >= 70:
                duration_adjustment = 1.0
                intensity_adjustment = 1.0
            elif completion_pct >= 50:
                duration_adjustment = 0.95
                intensity_adjustment = 0.95
            else:
                duration_adjustment = 0.85
                intensity_adjustment = 0.85
            
            if recovery_feeling == 1:
                duration_adjustment *= 0.90
                intensity_adjustment *= 0.90
            elif recovery_feeling == 5:
                duration_adjustment *= 1.05
                intensity_adjustment *= 1.05
            
            feedback_record = {
                'Age': int(row.get('age', 30)),
                'Sex': row.get('sex', 'M'),
                'Weight_kg': float(row.get('weight_kg', 75)),
                'SLEEP_HRS': float(row.get('sleep_hrs', 7)),
                'RHR_BPM': int(row.get('rhr_bpm', 60)),
                'SORENESS': int(row.get('soreness_before', 2)),
                'MENTAL_STRESS': int(row.get('mental_stress', 2)),
                'CALORIES_IN': int(row.get('calories_in', 2000)),
                'PROTEIN_G': int(row.get('protein_g', 100)),
                'CARBS_G': int(row.get('carbs_g', 250)),
                'NUTR_CONF_SCORE': 0.9,
                'GOAL': row.get('predicted_goal', 'Strength'),
                'Optimal_Duration_Min': row.get('recommended_duration', 45) * duration_adjustment,
                'Optimal_Intensity_RPE': row.get('recommended_intensity', 5.5) * intensity_adjustment
            }
            synthetic_feedback_data.append(feedback_record)
        
        if synthetic_feedback_data:
            feedback_synthetic_df = pd.DataFrame(synthetic_feedback_data)
            feedback_weighted = pd.concat([feedback_synthetic_df] * 2, ignore_index=True)
            df = pd.concat([df, feedback_weighted], ignore_index=True)
    
    # Generate targets
    base_duration = 45.0
    base_intensity_rpe = 5.5
    
    sleep_modifier = (df['SLEEP_HRS'] - 7) * 2
    rhr_modifier = (df['RHR_BPM'] - 60) * -0.1
    soreness_modifier = (df['SORENESS'] - 2.5) * -3
    stress_modifier = (df['MENTAL_STRESS'] - 3) * -2
    carbs_modifier = ((df['CARBS_G'] - 250) / 250) * 5
    protein_modifier = ((df['PROTEIN_G'] - 120) / 120) * 3
    
    age_modifier = np.zeros(len(df))
    age_modifier[df['Age'] < 30] = 3
    age_modifier[(df['Age'] >= 30) & (df['Age'] < 45)] = 0
    age_modifier[df['Age'] >= 45] = -3
    
    weight_modifier = ((df['Weight_kg'] - 75) / 75) * 2
    
    goal_duration_modifier = np.where(df['GOAL'] == 'Strength', 10, 
                                      np.where(df['GOAL'] == 'Endurance', 5, 
                                              np.where(df['GOAL'] == 'Yoga', -10, 0)))
    goal_intensity_modifier = np.where(df['GOAL'] == 'Strength', 2, 
                                       np.where(df['GOAL'] == 'Endurance', 1, 
                                               np.where(df['GOAL'] == 'Yoga', -3, 0)))
    
    confidence_modifier = ((df['NUTR_CONF_SCORE'] - 0.75) / 0.25) * 5
    
    duration = (base_duration + sleep_modifier + rhr_modifier + soreness_modifier + 
                stress_modifier + carbs_modifier + protein_modifier + age_modifier + 
                weight_modifier + goal_duration_modifier + confidence_modifier)
    intensity = (base_intensity_rpe + (sleep_modifier * 0.2) + (soreness_modifier * 0.2) + 
                 (stress_modifier * 0.15) + (protein_modifier * 0.15) + (age_modifier * 0.1) + 
                 (goal_intensity_modifier * 0.3) + (confidence_modifier * 0.1))
    
    duration = np.clip(duration, 20, 120)
    intensity = np.clip(intensity, 2, 9)
    
    # Prepare training data
    X_train_df = df[['Age', 'Sex', 'Weight_kg', 'SLEEP_HRS', 'RHR_BPM', 'SORENESS', 
                     'MENTAL_STRESS', 'CALORIES_IN', 'PROTEIN_G', 'CARBS_G', 'NUTR_CONF_SCORE']].copy()
    Y_goal = df['GOAL'].values
    
    # Preprocessing
    categorical_features = ['Sex']
    numerical_features = ['Age', 'Weight_kg', 'SLEEP_HRS', 'RHR_BPM', 'SORENESS', 
                          'MENTAL_STRESS', 'CALORIES_IN', 'PROTEIN_G', 'CARBS_G', 'NUTR_CONF_SCORE']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X_train_df)
    
    # Train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_processed, Y_goal):
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]
        Y_train_goal, Y_test_goal = Y_goal[train_idx], Y_goal[test_idx]
    
    # Handle pandas series vs numpy arrays
    if hasattr(duration, 'iloc'):
        Y_train_duration = duration.iloc[train_idx].values
        Y_test_duration = duration.iloc[test_idx].values
        Y_train_intensity = intensity.iloc[train_idx].values
        Y_test_intensity = intensity.iloc[test_idx].values
    else:
        Y_train_duration = duration[train_idx]
        Y_test_duration = duration[test_idx]
        Y_train_intensity = intensity[train_idx]
        Y_test_intensity = intensity[test_idx]
    
    # Train duration/intensity model
    base_rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, min_samples_split=5)
    multi_model = MultiOutputRegressor(base_rf)
    multi_model.fit(X_train, np.column_stack([Y_train_duration, Y_train_intensity]))
    
    # Train goal classifier
    goal_encoder = LabelEncoder()
    Y_train_goal_encoded = goal_encoder.fit_transform(Y_train_goal)
    goal_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    goal_classifier.fit(X_train, Y_train_goal_encoded)
    
    # Save all models
    joblib.dump(multi_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(goal_classifier, GOAL_CLASSIFIER_PATH)
    joblib.dump(goal_encoder, GOAL_ENCODER_PATH)

def parse_serving_size(food_input: str) -> Tuple[str, float, str]:
    """
    Parses serving size from food input string.
    Examples: "3 cups of rice" -> ("rice", 3.0, "3 cups"), "2 bananas" -> ("banana", 2.0, "2 medium bananas")
    Returns: (cleaned_food_name, quantity_multiplier, display_quantity_str)
    """
    
    # Standard conversion: 1 cup = 100g
    CUP_TO_GRAMS = 100
    
    food_input = food_input.strip().lower()
    original_input = food_input
    display_quantity = "1 cup"  # Default display
    
    # Pattern to match numbers followed by unit words
    patterns = [
        (r'(\d+\.?\d*)\s*(?:cups?)\s*(?:of\s+)?(.+)', 1, "cups"),  # "3 cups of rice"
        (r'(\d+\.?\d*)\s*(?:grams?|g|gm)\s*(?:of\s+)?(.+)', 1/CUP_TO_GRAMS, "grams"),  # "250 grams of rice"
        (r'(\d+\.?\d*)\s*(?:kg|kilograms?)\s*(?:of\s+)?(.+)', 10, "kg"),  # "1 kg of rice"
        (r'(\d+\.?\d*)\s*(?:tablespoons?|tbsp)\s*(?:of\s+)?(.+)', 1/16, "tbsp"),  # "2 tbsp of oil"
        (r'(\d+\.?\d*)\s*(?:teaspoons?|tsp)\s*(?:of\s+)?(.+)', 1/48, "tsp"),  # "1 tsp of salt"
        (r'(\d+\.?\d*)\s*(?:pieces?|pcs?)\s*(?:of\s+)?(.+)', 1/10, "pieces"),  # "2 pieces of bread"
        (r'(\d+\.?\d*)\s*(?:medium|small|large)?\s*(?:bananas?|apples?|oranges?|eggs?|breads?)\b', 1, "items"),  # "2 medium bananas"
    ]
    
    quantity = 1.0
    cleaned_name = food_input
    
    for pattern_info in patterns:
        pattern = pattern_info[0]
        multiplier = pattern_info[1]
        unit_name = pattern_info[2] if len(pattern_info) > 2 else "units"
        
        match = re.search(pattern, food_input, re.IGNORECASE)
        if match:
            quantity_num = float(match.group(1))
            quantity = quantity_num * multiplier
            
            # Extract food name
            if len(match.groups()) >= 2:
                cleaned_name = match.group(2).strip()
            else:
                cleaned_name = re.sub(pattern, '', food_input, flags=re.IGNORECASE).strip()
            
            # Format display quantity
            if unit_name == "cups":
                display_quantity = f"{quantity_num:.1f} cup{'s' if quantity_num != 1 else ''} ({quantity*100:.0f}g)"
            elif unit_name == "items":
                display_quantity = f"{quantity_num:.0f} {'medium' if 'medium' in original_input else 'standard'} {'items' if quantity_num != 1 else 'item'}"
            elif unit_name == "pieces":
                display_quantity = f"{quantity_num:.0f} piece{'s' if quantity_num != 1 else ''} ({quantity*100:.0f}g)"
            else:
                display_quantity = f"{quantity_num:.1f} {unit_name}"
            
            # Clean up extra words
            cleaned_name = re.sub(r'\b(?:of|the|a|an)\b', '', cleaned_name, flags=re.IGNORECASE).strip()
            break
    
    # If no pattern matched, try to extract just a number at the start
    if quantity == 1.0:
        match = re.match(r'^(\d+\.?\d*)\s+(.+)', food_input)
        if match:
            quantity_num = float(match.group(1))
            quantity = quantity_num
            cleaned_name = match.group(2).strip()
            display_quantity = f"{quantity_num:.0f} serving{'s' if quantity_num != 1 else ''}"
    
    return cleaned_name, quantity, display_quantity

def fetch_multi_source_nutrients(food_name: str, usda_key: str) -> Dict[str, float]:
    """
    PHASE 1: Attempts to find nutrient data from multiple sources with common food database fallback.
    Parses serving sizes (e.g., "3 cups of rice") and multiplies nutrition values accordingly.
    Returns: Dict of standardized macros and confidence score.
    """
    # Common foods database (per 100g) - for when APIs fail
    COMMON_FOODS = {
        'oats': {'cal': 389, 'protein': 17, 'carbs': 66},
        'milk': {'cal': 61, 'protein': 3.2, 'carbs': 4.8},
        'rice': {'cal': 130, 'protein': 2.7, 'carbs': 28},
        'chicken': {'cal': 165, 'protein': 31, 'carbs': 0},
        'egg': {'cal': 155, 'protein': 13, 'carbs': 1.1},
        'banana': {'cal': 89, 'protein': 1.1, 'carbs': 23},
        'apple': {'cal': 52, 'protein': 0.3, 'carbs': 14},
        'bread': {'cal': 265, 'protein': 9, 'carbs': 49},
        'pasta': {'cal': 371, 'protein': 13, 'carbs': 75},
        'peanut': {'cal': 567, 'protein': 26, 'carbs': 16},
        'almond': {'cal': 579, 'protein': 21, 'carbs': 22},
        'fish': {'cal': 100, 'protein': 20, 'carbs': 0},
        'beef': {'cal': 250, 'protein': 26, 'carbs': 0},
        'broccoli': {'cal': 34, 'protein': 2.8, 'carbs': 7},
        'spinach': {'cal': 23, 'protein': 2.9, 'carbs': 3.6},
        'yogurt': {'cal': 59, 'protein': 10, 'carbs': 3.3},
        'cheese': {'cal': 402, 'protein': 25, 'carbs': 1.3},
        'honey': {'cal': 304, 'protein': 0.3, 'carbs': 82},
        'oil': {'cal': 884, 'protein': 0, 'carbs': 0},
        'butter': {'cal': 717, 'protein': 0.9, 'carbs': 0.1},
    }
    
    # Parse serving size from input
    cleaned_food_name, quantity_multiplier, display_quantity = parse_serving_size(food_name)
    
    # Store original input for display
    original_input = food_name.strip()
    
    # 1. Fallback if key is missing or input is empty
    if not cleaned_food_name:
        st.warning("Please enter a food item name.")
        return {'CALORIES_IN': 2000, 'PROTEIN_G': 100, 'CARBS_G': 250, 'NUTR_CONF_SCORE': 0.5}
    
    if not usda_key:
        st.warning("USDA API key not configured. Using default values.")
        return {'CALORIES_IN': 2000, 'PROTEIN_G': 100, 'CARBS_G': 250, 'NUTR_CONF_SCORE': 0.5}
    
    # Base serving: 1 cup = 100g (our standard)
    BASE_SERVING_CUPS = 1.0
    BASE_SERVING_GRAMS = 100.0
    
    # Calculate total serving size in cups, then convert to grams
    total_cups = BASE_SERVING_CUPS * quantity_multiplier
    total_grams = total_cups * BASE_SERVING_GRAMS
    
    # --- FIRST: Check Common Foods Database ---
    food_lower = cleaned_food_name.lower().strip()
    for food_key, nutrition in COMMON_FOODS.items():
        if food_key in food_lower or food_lower in food_key:
            # Found in common foods
            serving_factor = total_grams / 100.0
            return {
                'CALORIES_IN': max(0, nutrition['cal'] * serving_factor),
                'PROTEIN_G': max(0, nutrition['protein'] * serving_factor),
                'CARBS_G': max(0, nutrition['carbs'] * serving_factor),
                'NUTR_CONF_SCORE': 0.92
            }
    
    # --- Attempt 1: Indian API Lookup ---
    try:
        ind_params = {'query': cleaned_food_name}
        ind_response = requests.get(IND_NUTRIENT_BASE_URL, params=ind_params, timeout=3)
        
        if ind_response.status_code == 200 and ind_response.json():
            ind_data = ind_response.json()
            if isinstance(ind_data, list) and ind_data:
                item = ind_data[0]
                
                # Get base nutrition per 100g (assuming API returns per 100g)
                base_calories = float(item.get('kcal', item.get('calories', 0)))
                base_protein = float(item.get('protein_g', item.get('protein', 0)))
                base_carbs = float(item.get('carb_g', item.get('carbs', 0)))
                
                # Multiply by serving size (total_grams / 100g)
                serving_factor = total_grams / 100.0
                
                return {
                    'CALORIES_IN': max(0, base_calories * serving_factor),
                    'PROTEIN_G': max(0, base_protein * serving_factor),
                    'CARBS_G': max(0, base_carbs * serving_factor),
                    'NUTR_CONF_SCORE': 0.95
                }
    except requests.exceptions.RequestException:
        pass
    
    # --- Attempt 2: USDA API Lookup ---
    usda_params = {
        'api_key': usda_key,
        'query': cleaned_food_name,
        'pageSize': 1,
        'dataType': ['SR Legacy', 'Foundation']
    }

    try:
        usda_response = requests.get(USDA_BASE_URL, params=usda_params, timeout=3)
        usda_response.raise_for_status() 
        data = usda_response.json()
        
        if data.get('foods'):
            food_item = data['foods'][0]
            # Safe extraction with error handling
            try:
                nutrients = {n.get('nutrientName', ''): n.get('value', 0) for n in food_item.get('foodNutrients', []) if n.get('nutrientName')}
            except (TypeError, KeyError):
                nutrients = {}
            
            # USDA typically returns per 100g, so multiply by serving size
            serving_factor = total_grams / 100.0
            
            # Return with safe defaults
            return {
                'CALORIES_IN': max(0, float(nutrients.get('Energy', 0) or 0) * serving_factor), 
                'PROTEIN_G': max(0, float(nutrients.get('Protein', 0) or 0) * serving_factor),
                'CARBS_G': max(0, float(nutrients.get('Carbohydrate, by difference', 0) or 0) * serving_factor),
                'NUTR_CONF_SCORE': 0.85

            }

    except requests.exceptions.RequestException:
        pass

    # --- Final Fallback ---
    st.warning("⚠️ Food not found in databases. Using default values.")
    return {'CALORIES_IN': 2000, 'PROTEIN_G': 100, 'CARBS_G': 250, 'NUTR_CONF_SCORE': 0.5}

def preprocess_and_predict(input_df: pd.DataFrame, model, preprocessor, goal_classifier, goal_encoder) -> Dict[str, Any]:
    """PHASE 2: Preprocessing and running the MVAA model for prediction. Also predicts goal."""
    try:
        # GOAL is no longer an input - it's predicted!
        original_features = ['age', 'sex', 'weight_kg', 'sleep_hrs', 'rhr_bpm', 'soreness', 
                             'mental_stress', 'calories_in', 'protein_g', 'carbs_g', 'nutrition_confidence']
        X_unprocessed = input_df[original_features]
        X_processed = preprocessor.transform(X_unprocessed)
        
        # Predict duration and intensity
        prediction_array = model.predict(X_processed)
        results_df = pd.DataFrame(prediction_array, columns=['Optimal_Duration_Min', 'Optimal_Intensity_RPE'])
        
        # Predict goal
        goal_encoded = goal_classifier.predict(X_processed)
        predicted_goal = goal_encoder.inverse_transform(goal_encoded)[0]
        
        return {
            'duration': results_df['Optimal_Duration_Min'].iloc[0],
            'intensity': results_df['Optimal_Intensity_RPE'].iloc[0],
            'goal': predicted_goal
        }
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return {'duration': None, 'intensity': None, 'goal': None}

def generate_exercise_recommendations(
    goal: str, 
    intensity: float, 
    duration: float,
    user_weight: float,
    soreness: int,
    carbs_g: float,
    sleep_hrs: float,
    stress: int
) -> Dict[str, Any]:
    """
    Generates specific exercise recommendations based on model output and user profile.
    Returns: Dict with muscle groups, exercises, reps, weights, and cardio recommendation.
    """
    recommendations = {
        'primary_muscle_group': None,
        'exercises': [],
        'cardio_recommended': False,
        'cardio_details': None,
        'mental_training': None
    }
    
    # Determine primary muscle group based on goal
    if goal == 'Strength':
        # Rotate through muscle groups based on intensity (simplified logic)
        muscle_groups = MUSCLE_GROUP_PRIORITY['Strength']
        # Use intensity to determine which muscle group (cycling logic)
        group_idx = int(intensity) % len(muscle_groups)
        recommendations['primary_muscle_group'] = muscle_groups[group_idx]
    elif goal == 'Yoga':
        # Yoga doesn't use muscle groups, it uses poses
        recommendations['primary_muscle_group'] = 'Full Body'
    else:
        recommendations['primary_muscle_group'] = 'Full Body'
    
    # Get exercises for the selected muscle group
    muscle_group = recommendations['primary_muscle_group']
    if goal in EXERCISE_DATABASE and muscle_group in EXERCISE_DATABASE[goal]:
        exercises = EXERCISE_DATABASE[goal][muscle_group]
        
        # Select 2-3 exercises based on duration and intensity
        num_exercises = min(3, max(2, int(duration / 20)))
        selected_exercises = exercises[:num_exercises]
        
        # Calculate optimal weight and reps for each exercise
        for exercise in selected_exercises:
            # Adjust weight based on intensity and readiness
            readiness_factor = 1.0
            if soreness >= 4:
                readiness_factor *= 0.7  # Reduce weight if very sore
            elif soreness <= 2:
                readiness_factor *= 1.1  # Can push harder if not sore
            
            if sleep_hrs < 6:
                readiness_factor *= 0.85  # Reduce if sleep deprived
            
            # Calculate optimal weight
            if exercise['base_weight_ratio'] > 0:
                base_weight = user_weight * exercise['base_weight_ratio']
                # Adjust for intensity (RPE 2-9 maps to 50%-90% of base)
                intensity_multiplier = 0.5 + (intensity / 10) * 0.4
                optimal_weight = base_weight * intensity_multiplier * readiness_factor
                optimal_weight = max(5, round(optimal_weight / 5) * 5)  # Round to nearest 5kg
            else:
                optimal_weight = None  # Bodyweight exercise
            
            # Calculate reps based on intensity and goal
            min_reps, max_reps = exercise['reps_range']
            if intensity <= 4:
                reps = max_reps + 5  # Higher reps for low intensity
            elif intensity >= 8:
                reps = min_reps - 2  # Lower reps for high intensity
            else:
                reps = (min_reps + max_reps) // 2
            
            reps = max(5, min(25, reps))  # Clamp between 5-25
            
            recommendations['exercises'].append({
                'name': exercise['name'],
                'weight_kg': optimal_weight,
                'reps': reps,
                'sets': 3 if intensity >= 6 else 2
            })
    
    # High stress = recommend yoga/mental training
    if stress >= 4:
        recommendations['mental_training'] = {
            'type': 'Mindfulness Yoga',
            'duration_min': 15,
            'reason': f'High mental stress ({stress}/5) detected. Yoga helps reduce cortisol and improves mental clarity.',
            'tips': 'Focus on breathing, stretching, and meditation. Even 10-15 minutes can significantly reduce stress.'
        }
    
    # Cardio recommendation based on carb intake and goal
    if carbs_g > 300:  # High carb threshold
        recommendations['cardio_recommended'] = True
        
        # Choose cardio type based on intensity and stress
        if intensity >= 8:
            cardio_type = 'HIIT Treadmill'
            duration_min = min(20, max(10, int(duration * 0.3)))
        elif intensity >= 5:
            cardio_type = 'Staircase Climber'
            duration_min = min(25, max(15, int(duration * 0.35)))
        else:
            cardio_type = 'Steady Cardio (Treadmill/Elliptical)'
            duration_min = min(30, max(20, int(duration * 0.4)))
        
        recommendations['cardio_details'] = {
            'type': cardio_type,
            'duration_min': duration_min,
            'reason': f'High carbohydrate intake ({carbs_g:.0f}g) - optimal fuel for cardiovascular exercise',
            'options': ['Treadmill', 'Stationary Bike', 'Staircase Climber', 'Rowing Machine']
        }
    
    return recommendations

# --- STREAMLIT UI LAYOUT ---

st.set_page_config(
    layout="wide", 
    page_title="ChronoFit MVAA - Personalized Workout AI",
    page_icon="🏋️"
)

# Check and retrain model if new feedback is available (runs silently at startup)
check_and_retrain_model()

# Initialize session state to track if app has been loaded
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

# Show loading screen only on first load
if not st.session_state.app_loaded:
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        st.markdown("""
            <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.7; transform: scale(0.95); }
            }
            
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes progressBar {
                0% { width: 0%; }
                100% { width: 100%; }
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            @keyframes gradientShift {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            .initial-loading-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 50%, #2d2d2d 100%);
                background-size: 400% 400%;
                animation: gradientShift 3s ease infinite;
                z-index: 9999;
            }
            
            .loader {
                width: 80px;
                height: 80px;
                border: 6px solid rgba(255, 69, 0, 0.2);
                border-top: 6px solid #FF4500;
                border-radius: 50%;
                animation: spin 1.2s linear infinite;
                margin-bottom: 30px;
                box-shadow: 0 0 20px rgba(255, 69, 0, 0.6);
            }
            
            .loading-text {
                color: white;
                font-size: 32px;
                font-weight: 800;
                text-align: center;
                animation: pulse 2s ease-in-out infinite;
                margin-bottom: 15px;
                text-shadow: 0 0 15px rgba(255, 69, 0, 0.6),
                             0 4px 10px rgba(0, 0, 0, 0.8),
                             2px 2px 0 #DC143C;
                color: #FFD700;
            }
            
            .loading-subtext {
                color: rgba(255, 215, 0, 0.95);
                font-size: 16px;
                text-align: center;
                animation: slideUp 0.8s ease-out;
                max-width: 600px;
                padding: 0 20px;
                line-height: 1.6;
                margin-bottom: 30px;
                text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
            }
            
            .progress-container-initial {
                width: 300px;
                height: 6px;
                background: rgba(255, 69, 0, 0.2);
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 20px;
                border: 2px solid #FF4500;
            }
            
            .progress-bar-initial {
                height: 100%;
                background: linear-gradient(90deg, #FF4500 0%, #FFD700 100%);
                border-radius: 10px;
                animation: progressBar 2.5s ease-in-out;
                box-shadow: 0 0 15px rgba(255, 69, 0, 0.8);
            }
            </style>
            
            <div class="initial-loading-container">
                <div class="loader"></div>
                <div class="loading-text">🏋️ ChronoFit MVAA</div>
                <div class="loading-subtext">AI-Powered Personalized Workout Recommendations</div>
                <div class="progress-container-initial">
                    <div class="progress-bar-initial"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
    
    loading_placeholder.empty()
    st.session_state.app_loaded = True

# Professional Sports/Gym Theme CSS with fierce styling
st.markdown("""
    <style>
    /* Prevent page jumping when scrollbar appears/disappears on focus */
    html {
        overflow-y: scroll;
        scrollbar-gutter: stable;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes scaleUp {
        from {
            transform: scale(0.8);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-100%);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fieryGlow {
        0%, 100% {
            text-shadow: 0 0 10px rgba(255, 69, 0, 0.6),
                         0 4px 15px rgba(0, 0, 0, 0.8);
        }
        50% {
            text-shadow: 0 0 15px rgba(255, 69, 0, 0.7),
                         0 4px 15px rgba(0, 0, 0, 0.8);
        }
    }
    
    @keyframes pumpWeight {
        0%, 100% {
            transform: translateY(0px) scale(1);
        }
        25% {
            transform: translateY(-8px) scale(1.02);
        }
        50% {
            transform: translateY(0px) scale(1);
        }
        75% {
            transform: translateY(-8px) scale(1.02);
        }
    }
    
    @keyframes muscleFlexPulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(255, 69, 0, 0.4);
        }
        50% {
            box-shadow: 0 0 0 10px rgba(255, 69, 0, 0);
        }
    }
    
    @keyframes boltStrike {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.7;
            transform: scale(1.1) rotate(5deg);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 50%, #2d2d2d 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 0;
    }
    
    /* FIERCE Gym Header Styles */
    .header-container {
        background: linear-gradient(135deg, #FF4500 0%, #DC143C 40%, #FF8C00 70%, #FFD700 100%);
        background-size: 300% 300%;
        animation: slideDown 1s ease-out, gradientShift 4s ease infinite;
        padding: 60px 20px;
        border-radius: 0 0 40px 40px;
        margin-bottom: 40px;
        box-shadow: 0 0 50px rgba(255, 69, 0, 0.6),
                    0 20px 60px rgba(220, 20, 60, 0.5),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
        border: 3px solid #FF6347;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            linear-gradient(45deg, transparent 40%, rgba(255, 255, 255, 0.05) 50%, transparent 60%);
    }
    
    .header-container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        animation: pulse 3s ease-in-out infinite;
    }
    
    .header-title {
        color: white;
        text-align: center;
        font-size: 64px;
        font-weight: 900;
        margin: 0;
        animation: fieryGlow 2.5s ease-in-out infinite;
        position: relative;
        z-index: 1;
        text-shadow: 
            0 0 10px rgba(255, 69, 0, 0.6),
            0 4px 15px rgba(0, 0, 0, 0.8),
            2px 2px 0 #DC143C;
        letter-spacing: 2px;
        text-transform: uppercase;
        transform: perspective(500px) rotateX(1deg);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.98);
        text-align: center;
        font-size: 20px;
        margin-top: 15px;
        animation: fadeInUp 1s ease-out 0.3s both;
        position: relative;
        z-index: 1;
        letter-spacing: 3px;
        font-weight: 700;
        text-transform: uppercase;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
    }
    
    /* Section Headers - Sports Style */
    .section-header {
        font-size: 1.8rem;
        color: #FFD700;
        margin: 30px 0 20px 0;
        font-weight: 900;
        animation: slideInLeft 0.8s ease-out;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8),
                     2px 2px 0 #FF4500,
                     0 0 15px rgba(255, 69, 0, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input Section - Dark Gym Theme */
    .input-section {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 35px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(255, 69, 0, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: fadeInUp 0.8s ease-out;
        border: 2px solid #FF6347;
        border-top: 4px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .input-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 69, 0, 0.05) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .input-section h3 {
        color: #FFD700;
        margin-bottom: 25px;
        font-size: 26px;
        font-weight: 900;
        position: relative;
        z-index: 1;
        animation: slideInLeft 0.8s ease-out;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button Styles - Aggressive Gym Style */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF4500 0%, #DC143C 50%, #FF8C00 100%);
        background-size: 200% 200%;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #FFD700;
        box-shadow: 0 0 30px rgba(255, 69, 0, 0.5),
                    0 8px 16px rgba(220, 20, 60, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: slideInUp 0.8s ease-out 0.3s both;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
        font-weight: 900;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.4);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 0 50px rgba(255, 69, 0, 0.8),
                    0 15px 30px rgba(220, 20, 60, 0.6);
        background: linear-gradient(135deg, #FF6347 0%, #FF1493 50%, #FFD700 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    /* Metric Cards - Gym Themed */
    .metric-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 35px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(255, 69, 0, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin: 15px 0;
        animation: bounceIn 0.8s ease-out;
        border: 3px solid #FF4500;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 69, 0, 0.15),
            transparent
        );
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 0 40px rgba(255, 69, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: #FFD700;
    }
    
    .metric-card h3 {
        color: #FFD700;
        text-transform: uppercase;
        font-weight: 900;
        letter-spacing: 1px;
    }
    
    .metric-card h1 {
        color: #FF4500;
        text-shadow: 0 0 15px rgba(255, 69, 0, 0.6);
    }
    
    /* Goal Badge - Fierce Styles */
    .goal-badge-high {
        background: linear-gradient(135deg, #FF4500 0%, #DC143C 40%, #FF8C00 100%);
        background-size: 300% 300%;
        color: white;
        padding: 50px 30px;
        border-radius: 20px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 0 50px rgba(255, 69, 0, 0.6),
                    0 20px 60px rgba(220, 20, 60, 0.5);
        animation: scaleUp 0.8s ease-out, gradientShift 3s ease infinite;
        border: 3px solid #FFD700;
        position: relative;
        transform: perspective(800px) rotateX(1deg) rotateY(-1deg);
    }
    
    .goal-badge-high::after {
        content: '💪';
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 32px;
        animation: boltStrike 1.5s ease-in-out infinite;
    }
    
    .goal-strength {
        background: linear-gradient(135deg, #FF4500 0%, #DC143C 100%);
    }
    
    .goal-endurance {
        background: linear-gradient(135deg, #FF8C00 0%, #FFD700 100%);
    }
    
    .goal-maintenance {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%);
        border-color: #FF4500;
    }
    
    .goal-yoga {
        background: linear-gradient(135deg, #6495ED 0%, #87CEEB 100%);
        border-color: #4169E1;
    }
    
    /* Exercise Card - Sports Theme */
    .exercise-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 6px solid #FF4500;
        margin: 20px 0;
        box-shadow: 0 5px 20px rgba(255, 69, 0, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: slideInLeft 0.8s ease-out;
        border-bottom: 3px solid #FFD700;
    }
    
    .exercise-card:hover {
        transform: translateX(10px) translateY(-5px);
        box-shadow: 0 0 30px rgba(255, 69, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        border-left-color: #FFD700;
        background: linear-gradient(135deg, #333333 0%, #1a1a1a 100%);
    }
    
    .exercise-name {
        font-size: 1.6rem;
        font-weight: 900;
        color: #FFD700;
        margin-bottom: 15px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .exercise-details {
        font-size: 1.1rem;
        color: #FFFFFF;
        margin: 10px 0;
        line-height: 1.8;
        font-weight: 600;
    }
    
    .exercise-note {
        font-size: 0.95rem;
        color: #FFA500;
        margin-top: 15px;
        font-style: italic;
        border-top: 2px dashed #FF4500;
        padding-top: 15px;
    }
    
    /* Readiness Notes - Color Coded */
    .workout-notes-container {
        background: linear-gradient(135deg, #2a1a1a 0%, #1a0d0d 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #FF4500;
        margin: 15px 0;
        animation: fadeInUp 0.8s ease-out;
        box-shadow: 0 5px 15px rgba(255, 69, 0, 0.15);
        border-bottom: 3px solid #FF6347;
        color: #FFFFFF;
    }
    
    .success-note {
        background: linear-gradient(135deg, #1a2a1a 0%, #0d1a0d 100%);
        border-left-color: #228B22;
        border-bottom-color: #32CD32;
        box-shadow: 0 5px 15px rgba(34, 139, 34, 0.15);
    }
    
    .info-note {
        background: linear-gradient(135deg, #1a2a3a 0%, #0d1a2a 100%);
        border-left-color: #FF8C00;
        border-bottom-color: #FFD700;
        box-shadow: 0 5px 15px rgba(255, 140, 0, 0.15);
    }
    
    /* Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 40px;
        border-radius: 15px;
        margin: 30px 0;
        box-shadow: 0 8px 32px rgba(255, 69, 0, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-left: 6px solid #FF4500;
        animation: slideInLeft 0.8s ease-out;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        border-bottom: 3px solid #FFD700;
    }
    
    .recommendation-box:hover {
        transform: translateX(15px) scale(1.02);
        box-shadow: 0 0 40px rgba(255, 69, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .recommendation-box h4 {
        margin-bottom: 25px;
        font-size: 26px;
        font-weight: 900;
        color: #FFD700;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
    }
    
    .recommendation-box p {
        color: #FFFFFF;
        line-height: 2;
        font-size: 17px;
        margin: 12px 0;
    }
    
    /* Success Animation */
    @keyframes successPulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.05);
            opacity: 0.9;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .success-container {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 50%, #00AA00 100%);
        background-size: 300% 300%;
        color: white;
        padding: 50px 30px;
        border-radius: 15px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 0 50px rgba(34, 205, 50, 0.5),
                    0 20px 60px rgba(34, 139, 34, 0.4);
        animation: bounceIn 0.8s ease-out, gradientShift 3s ease infinite;
        border: 3px solid #FFD700;
        position: relative;
        text-transform: uppercase;
        font-weight: 900;
        letter-spacing: 2px;
    }
    
    .success-container h2 {
        font-size: 36px;
        font-weight: 900;
        margin: 10px 0;
        letter-spacing: 3px;
        text-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    
    .success-container p {
        font-size: 18px;
        margin-top: 15px;
        letter-spacing: 1px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #FFD700;
        padding: 50px 20px;
        margin-top: 80px;
        border-top: 3px solid #FF4500;
        animation: fadeInUp 1s ease-out 0.8s both;
        background: linear-gradient(135deg, rgba(255, 69, 0, 0.1) 0%, rgba(220, 20, 60, 0.1) 100%);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(255, 69, 0, 0.1);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .footer p {
        margin: 12px 0;
        font-size: 15px;
    }
    
    /* Responsive Design - Enhanced for Mobile */
    @media (max-width: 768px) {
        .header-title {
            font-size: 42px;
            letter-spacing: 1px;
            margin: 0;
        }
        
        .header-subtitle {
            font-size: 14px;
            margin-top: 10px;
        }
        
        .header-container {
            padding: 30px 15px;
            margin-bottom: 20px;
            border-radius: 20px;
        }
        
        .metric-card {
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
        }
        
        .metric-card h3 {
            font-size: 14px;
        }
        
        .metric-card h1 {
            font-size: 24px;
        }
        
        .input-section {
            padding: 20px;
            margin: 15px 0;
            border-radius: 12px;
        }
        
        .input-section h3 {
            font-size: 20px;
            margin-bottom: 15px;
        }
        
        .section-header {
            font-size: 1.4rem;
            margin: 20px 0 15px 0;
        }
        
        .stButton>button {
            font-size: 14px;
            padding: 15px;
            border-radius: 10px;
        }
        
        .goal-badge-high {
            padding: 30px 20px;
            border-radius: 15px;
            margin: 20px 0;
        }
        
        .goal-badge-high::after {
            font-size: 24px;
            top: 8px;
            right: 8px;
        }
        
        .exercise-card {
            padding: 20px;
            margin: 15px 0;
            border-radius: 12px;
        }
        
        .exercise-name {
            font-size: 1.3rem;
            margin-bottom: 10px;
        }
        
        .exercise-details {
            font-size: 1rem;
            margin: 8px 0;
        }
        
        .recommendation-box {
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
        }
        
        .recommendation-box h4 {
            font-size: 20px;
            margin-bottom: 15px;
        }
        
        .recommendation-box p {
            font-size: 15px;
            line-height: 1.6;
        }
        
        .footer {
            padding: 30px 15px;
            margin-top: 40px;
            border-radius: 10px;
        }
        
        .footer p {
            font-size: 13px;
            margin: 8px 0;
        }
        
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            font-size: 16px !important;
            padding: 10px 12px !important;
            border-radius: 8px !important;
        }
        
        .workout-notes-container {
            padding: 15px;
            border-left: 4px solid #FF4500;
            margin: 12px 0;
            border-radius: 10px;
        }
    }
    
    /* Small Mobile Devices (< 480px) */
    @media (max-width: 480px) {
        .header-title {
            font-size: 28px;
            letter-spacing: 0px;
            text-shadow: 
                0 0 8px rgba(255, 69, 0, 0.6),
                0 2px 10px rgba(0, 0, 0, 0.8),
                2px 2px 0 #DC143C;
        }
        
        .header-subtitle {
            font-size: 12px;
            margin-top: 8px;
        }
        
        .header-container {
            padding: 20px 10px;
            border-radius: 15px;
            margin-bottom: 15px;
            border: 2px solid #FF6347;
        }
        
        .header-container::before {
            display: none;
        }
        
        .metric-card {
            padding: 15px;
            margin: 8px 0;
            border-radius: 8px;
        }
        
        .metric-card h3 {
            font-size: 12px;
            margin-bottom: 5px;
        }
        
        .metric-card h1 {
            font-size: 20px;
        }
        
        .input-section {
            padding: 15px;
            margin: 12px 0;
            border-radius: 10px;
            border: 2px solid #FF6347;
            border-top: 3px solid #FFD700;
        }
        
        .input-section h3 {
            font-size: 16px;
            margin-bottom: 12px;
        }
        
        .section-header {
            font-size: 1.2rem;
            margin: 15px 0 12px 0;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8),
                         2px 2px 0 #FF4500;
        }
        
        .stButton>button {
            font-size: 13px;
            padding: 12px;
            border-radius: 8px;
            border: 2px solid #FFD700;
        }
        
        .goal-badge-high {
            padding: 25px 15px;
            border-radius: 12px;
            margin: 15px 0;
            border: 2px solid #FFD700;
        }
        
        .goal-badge-high h2 {
            font-size: 20px;
        }
        
        .goal-badge-high h3 {
            font-size: 14px;
        }
        
        .goal-badge-high::after {
            font-size: 20px;
            top: 5px;
            right: 5px;
        }
        
        .exercise-card {
            padding: 15px;
            margin: 12px 0;
            border-radius: 8px;
            border-left: 4px solid #FF4500;
            border-bottom: 2px solid #FFD700;
        }
        
        .exercise-name {
            font-size: 1.1rem;
            margin-bottom: 8px;
        }
        
        .exercise-details {
            font-size: 0.9rem;
            margin: 6px 0;
            line-height: 1.6;
        }
        
        .recommendation-box {
            padding: 18px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #FF4500;
            border-bottom: 2px solid #FFD700;
        }
        
        .recommendation-box h4 {
            font-size: 16px;
            margin-bottom: 12px;
        }
        
        .recommendation-box p {
            font-size: 13px;
            line-height: 1.5;
            margin: 8px 0;
        }
        
        .footer {
            padding: 20px 10px;
            margin-top: 30px;
            border-radius: 8px;
            border: 2px solid #FF4500;
        }
        
        .footer p {
            font-size: 11px;
            margin: 6px 0;
        }
        
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            font-size: 14px !important;
            padding: 8px 10px !important;
            border-radius: 6px !important;
            min-height: 40px;
        }
        
        .workout-notes-container {
            padding: 12px;
            border-left: 3px solid #FF4500;
            margin: 10px 0;
            border-radius: 8px;
            border-bottom: 2px solid #FF6347;
            font-size: 13px;
        }
        
        /* Ensure labels are readable on mobile */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            word-break: break-word;
        }
    }
    
    /* Extra Small Devices (< 360px) */
    @media (max-width: 360px) {
        .header-title {
            font-size: 24px;
        }
        
        .header-container {
            padding: 15px 8px;
        }
        
        .section-header {
            font-size: 1.1rem;
        }
        
        .metric-card h1 {
            font-size: 18px;
        }
    }
    
    /* Landscape Mode Adjustments */
    @media (max-height: 500px) {
        .header-container {
            padding: 20px 15px;
            margin-bottom: 15px;
        }
        
        .header-title {
            font-size: 32px;
            margin: 0;
        }
        
        .header-subtitle {
            font-size: 12px;
            margin-top: 5px;
        }
    }
    
    
    /* Hide number input increment/decrement buttons - MULTIPLE APPROACHES */
    /* Approach 1: WebKit browsers (Chrome, Safari, Edge) */
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        appearance: none !important;
        margin: 0 !important;
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
    }
    
    /* Approach 2: Firefox */
    input[type=number] {
        -moz-appearance: textfield !important;
    }
    
    /* Approach 3: Streamlit number input wrapper */
    .stNumberInput {
        appearance: none !important;
        -webkit-appearance: none !important;
        -moz-appearance: textfield !important;
    }
    
    /* === FIX NUMBER INPUT STYLING === */
    
    /* Make labels bigger and better spaced */
    .stNumberInput > label,
    .stSelectbox > label {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 8px !important;
        display: block !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.6) !important;
    }
    
    /* Reduce input box height and padding */
    .stNumberInput input[type=number],
    .stSelectbox select {
        height: 36px !important;
        padding: 6px 12px !important;
        font-size: 14px !important;
        border-radius: 6px !important;
        border: 2px solid #FF4500 !important;
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
    }
    
    /* Reduce overall container height */
    .stNumberInput,
    .stSelectbox {
        margin-bottom: 12px !important;
    }
    
    /* Remove ALL +/- buttons with extreme CSS */
    input[type="number"] {
        -moz-appearance: textfield !important;
        appearance: textfield !important;
        -webkit-appearance: textfield !important;
    }
    
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-calendar-picker-indicator {
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        pointer-events: none !important;
    }
    
    /* Firefox specific */
    input[type="number"]:disabled {
        background-color: #2a2a2a !important;
    }
    
    /* Target Streamlit components specifically */
    div.stNumberInput input {
        -webkit-appearance: textfield !important;
        -moz-appearance: textfield !important;
        appearance: textfield !important;
    }
    
    /* Remove up/down arrows completely */
    .stNumberInput input::-webkit-outer-spin-button {
        display: none !important;
    }
    
    .stNumberInput input::-webkit-inner-spin-button {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORTANT: Load models FIRST with loading indicator before rendering UI
# ============================================================================

# Create placeholder for loading screen
loading_placeholder = st.empty()

# Show loading screen while models load
with loading_placeholder.container():
    st.markdown("""
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh; background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 50%, #2d2d2d 100%);">
            <div style="width: 80px; height: 80px; border: 6px solid rgba(255, 69, 0, 0.2); border-top: 6px solid #FF4500; border-radius: 50%; animation: spin 1.2s linear infinite; margin-bottom: 30px; box-shadow: 0 0 20px rgba(255, 69, 0, 0.6);"></div>
            <p style="color: white; font-size: 24px; font-weight: 800; text-align: center; animation: pulse 2s ease-in-out infinite;">Loading ChronoFit AI Models...</p>
            <p style="color: #aaa; font-size: 14px; margin-top: 20px;">This may take 10-15 seconds on first load</p>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.7; }
                }
            </style>
        </div>
    """, unsafe_allow_html=True)

# Load the artifacts (with full wait - no partial loading)
mvva_model, preprocessor, goal_classifier, goal_encoder = load_ml_artifacts()

# Check if models loaded successfully
if mvva_model is None or preprocessor is None or goal_classifier is None or goal_encoder is None:
    st.error("Failed to load AI models. Please check the model files and restart the app.")
    st.stop()

# Models loaded successfully - clear the loading screen
loading_placeholder.empty()

# Header Section
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏋️ CHRONOFIT MVAA</h1>
        <p class="header-subtitle">AI-Powered Personalized Workout Recommendations Platform</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- User Input Forms ---
st.markdown('<h2 class="section-header">👤 Your Profile & Readiness</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="input-section" style="margin-bottom: 0;">
            <h3>🏋️ Body Information</h3>
    """, unsafe_allow_html=True)
    user_age = st.text_input("Age (years)", value="30", help="Enter your age")
    user_sex = st.selectbox("Sex", options=['M', 'F'])
    user_weight = st.text_input("Weight (kg)", value="75", help="Enter your weight")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="input-section" style="margin-bottom: 0;">
            <h3>❤️ Daily Readiness Factors</h3>
    """, unsafe_allow_html=True)
    user_sleep = st.text_input("Sleep Hours (last night)", value="7", help="Hours of sleep")
    user_rhr = st.text_input("Resting Heart Rate (BPM)", value="60", help="Your resting heart rate")
    user_soreness = st.text_input("Physical Soreness (1-5)", value="2", help="1=none, 5=very sore")
    user_stress = st.text_input("Mental Stress Level (1-5)", value="2", help="1=low, 5=high")
    st.markdown("</div>", unsafe_allow_html=True)

# Convert text inputs to numbers
try:
    user_age = int(user_age)
    user_weight = float(user_weight)
    user_sleep = float(user_sleep)
    user_rhr = int(user_rhr)
    user_soreness = int(user_soreness)
    user_stress = int(user_stress)
except ValueError:
    st.error("Please enter valid numbers for all fields")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-header">🍽️ Nutritional Intake</h2>', unsafe_allow_html=True)

nutr_option = st.radio(
    "How would you like to enter your food data?",
    ('🔍 Automated Lookup (Multi-Source)', '📝 Manual Estimate'),
    horizontal=True
)

nutr_features = {'CALORIES_IN': 2000, 'PROTEIN_G': 100, 'CARBS_G': 250, 'NUTR_CONF_SCORE': 0.7}
lookup_clicked = False

if nutr_option == '🔍 Automated Lookup (Multi-Source)':
    st.info("💡 **Smart Parsing**: Enter quantities naturally - '3 cups rice', '2 medium bananas', '250g chicken', etc. Standard: 1 cup = 100g")
    food_name = st.text_input(
        "Enter food item with quantity",
        value="",
        placeholder="Examples: '3 cups rice', '2 medium bananas', '250g chicken breast', '2 pieces roti'",
        help="Enter the food name with quantity. Our AI parser understands natural language."
    )
    
    if st.button("🔍 Lookup Nutrients", type="primary", use_container_width=True):
        if not food_name.strip():
            st.error("Please enter a food item name.")
        else:
            lookup_clicked = True
            with st.spinner(f"🔎 Searching databases for '{food_name}'..."):
                nutr_features = fetch_multi_source_nutrients(food_name, USDA_API_KEY)
                
                # Display results in a nice grid
                col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                with col_n1:
                    st.metric("🔥 Calories", f"{nutr_features['CALORIES_IN']:.0f} kcal")
                with col_n2:
                    st.metric("💪 Protein", f"{nutr_features['PROTEIN_G']:.0f}g")
                with col_n3:
                    st.metric("🌾 Carbs", f"{nutr_features['CARBS_G']:.0f}g")
                with col_n4:
                    confidence_pct = nutr_features['NUTR_CONF_SCORE'] * 100
                    st.metric("✅ Confidence", f"{confidence_pct:.0f}%")

    if not lookup_clicked:
        pass

elif nutr_option == '📝 Manual Estimate':
    st.info("💡 Enter your estimated daily nutritional intake")
    
    col_man1, col_man2, col_man3 = st.columns(3)
    
    with col_man1:
        manual_calories = st.number_input("Total Calories", min_value=0, value=2500)
    with col_man2:
        manual_protein = st.number_input("Total Protein (g)", min_value=0, value=120)
    with col_man3:
        manual_carbs = st.number_input("Total Carbs (g)", min_value=0, value=350)
    
    nutr_features = {
        'CALORIES_IN': manual_calories,
        'PROTEIN_G': manual_protein,
        'CARBS_G': manual_carbs,
        'NUTR_CONF_SCORE': 0.55
    }

st.markdown("<br>", unsafe_allow_html=True)

# --- Assemble Final Input DataFrame ---
final_input_data = {
    'age': [user_age], 'sex': [user_sex], 'weight_kg': [user_weight],
    'sleep_hrs': [user_sleep], 'rhr_bpm': [user_rhr], 'soreness': [user_soreness],
    'mental_stress': [user_stress],
    'calories_in': [nutr_features['CALORIES_IN']],
    'protein_g': [nutr_features['PROTEIN_G']],
    'carbs_g': [nutr_features['CARBS_G']],
    'nutrition_confidence': [nutr_features['NUTR_CONF_SCORE']]
}

input_df = pd.DataFrame(final_input_data)

# --- Run Prediction ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'exercise_recommendations' not in st.session_state:
    st.session_state.exercise_recommendations = None
if 'predicted_goal' not in st.session_state:
    st.session_state.predicted_goal = None

if st.button("🚀 Generate My Personalized Workout Plan", type="primary", use_container_width=True):
    with st.spinner("🤖 AI analyzing 11 readiness factors..."):
        prediction_output = preprocess_and_predict(input_df, mvva_model, preprocessor, goal_classifier, goal_encoder)
        
        if prediction_output['duration'] is not None:
            duration = prediction_output['duration']
            intensity = prediction_output['intensity']
            predicted_goal = prediction_output['goal']
            
            # Generate exercise recommendations using PREDICTED goal
            exercise_recs = generate_exercise_recommendations(
                predicted_goal, intensity, duration, user_weight, 
                user_soreness, nutr_features['CARBS_G'], user_sleep, user_stress
            )
            
            st.session_state.prediction = (duration, intensity)
            st.session_state.exercise_recommendations = exercise_recs
            st.session_state.predicted_goal = predicted_goal
            
            st.rerun()
        else:
            st.error("❌ Prediction failed. Please check your inputs.")

# Display the results
if st.session_state.prediction and st.session_state.exercise_recommendations and st.session_state.predicted_goal:
    duration, intensity = st.session_state.prediction
    exercise_recs = st.session_state.exercise_recommendations
    predicted_goal = st.session_state.predicted_goal
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Success Animation - Professional
    st.markdown(f"""
    <div class="success-container">
        <h2>✅ Your Personalized Workout Plan is Ready!</h2>
        <p>Based on 11 readiness factors analyzed by our AI engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🎯 Recommended Training Focus</h2>', unsafe_allow_html=True)
    
    # Show predicted goal prominently
    goal_class = f"goal-{predicted_goal.lower()}"
    goal_emoji = {"Strength": "💪", "Endurance": "🏃", "Maintenance": "🧘", "Yoga": "🕉️"}
    st.markdown(f"""
    <div class="goal-badge-high {goal_class}">
        <h1 style="font-size: 48px; margin: 0;">{goal_emoji[predicted_goal]} {predicted_goal} Training</h1>
        <p style="font-size: 18px; margin-top: 15px; opacity: 0.95;">
            AI Analysis: Sleep ({user_sleep}h) • Soreness ({user_soreness}/5) • Stress ({user_stress}/5) • Carbs ({nutr_features['CARBS_G']:.0f}g) • Age ({user_age}y)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea;">⏱️ Total Duration</h3>
            <h1 style="color: #667eea; font-size: 48px; margin: 10px 0;">{duration:.0f} min</h1>
            <p style="color: #7f8c8d; font-size: 14px;">Optimal workout time based on your readiness</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        intensity_color = "#27ae60" if intensity >= 7 else "#f39c12" if intensity >= 4 else "#e74c3c"
        intensity_label = "High Intensity" if intensity >= 7 else "Moderate Intensity" if intensity >= 4 else "Light Intensity"
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: {intensity_color};">
            <h3 style="color: {intensity_color};">🔥 Intensity Level</h3>
            <h1 style="color: {intensity_color}; font-size: 48px; margin: 10px 0;">{intensity:.1f}/10 RPE</h1>
            <p style="color: #7f8c8d; font-size: 14px;">{intensity_label} - Rate of Perceived Exertion</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Safety and readiness notes
    st.markdown('<h2 class="section-header">📋 Important Workout Notes</h2>', unsafe_allow_html=True)
    
    notes = []
    
    if user_soreness >= 4:
        notes.append(("⚠️ High Soreness Detected", 
                     "Consider reducing weight by 20-30% and focus on form over load. Your muscles need recovery time.",
                     "workout-notes-container"))
    if user_sleep < 6:
        notes.append(("😴 Low Sleep Warning", 
                     "Your recovery may be compromised. Consider lighter intensity today and prioritize rest tonight.",
                     "workout-notes-container"))
    if intensity <= 4:
        notes.append(("🛡️ Recovery Day", 
                     "Perfect for active recovery. Focus on technique, mobility, and form rather than intensity.",
                     "info-note workout-notes-container"))
    if intensity >= 8:
        notes.append(("🔥 High Intensity Day", 
                     "Ensure proper warm-up (10-15 min) and cool-down. Stay hydrated and maintain proper form!",
                     "workout-notes-container"))
    if user_age >= 50:
        notes.append(("👴 Age-Based Recommendations", 
                     "Your body may need more recovery time. Listen to your body, use proper form, and don't skip warm-up/cool-down.",
                     "info-note workout-notes-container"))
    
    if notes:
        for title, description, note_class in notes:
            st.markdown(f"""
            <div class="{note_class}">
                <p style="margin: 0; font-size: 1.15rem; font-weight: 700; margin-bottom: 8px;">{title}</p>
                <p style="margin: 0; font-size: 1.05rem;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-note workout-notes-container">
            <p style="margin: 0; font-size: 1.1rem; font-weight: 700;">✅ All readiness factors are optimal!</p>
            <p style="margin: 8px 0 0 0; font-size: 1.05rem;">You're in great condition for a challenging workout. Give it your best effort!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Exercise recommendations
    if exercise_recs['primary_muscle_group']:
        st.markdown(f'<h2 class="section-header">💪 Focus: {exercise_recs["primary_muscle_group"]}</h2>', unsafe_allow_html=True)
        
        if exercise_recs['exercises']:
            st.markdown('<h3 style="color: #FFD700; margin: 20px 0;">🎯 Recommended Exercises</h3>', unsafe_allow_html=True)
            
            for i, exercise in enumerate(exercise_recs['exercises'], 1):
                weight_str = f"{exercise['weight_kg']} kg" if exercise['weight_kg'] else "Bodyweight"
                st.markdown(f"""
                <div class="exercise-card">
                    <div class="exercise-name">#{i} {exercise['name']}</div>
                    <div class="exercise-details">
                        <strong>⚖️ Weight:</strong> {weight_str} 
                        <br><strong>🔄 Reps:</strong> {exercise['reps']} per set
                        <br><strong>📊 Sets:</strong> {exercise['sets']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Cardio recommendation
    if exercise_recs['cardio_recommended']:
        st.markdown('<h2 class="section-header">🏃 Cardio Recommendation</h2>', unsafe_allow_html=True)
        cardio = exercise_recs['cardio_details']
        st.markdown(f"""
        <div class="recommendation-box">
            <h4>{cardio['type']}</h4>
            <p><strong>Duration:</strong> {cardio['duration_min']} minutes</p>
            <p><strong>Reason:</strong> {cardio['reason']}</p>
            <div style="background: rgba(255, 69, 0, 0.1); border-left: 4px solid #FF4500; padding: 12px; margin-top: 12px; border-radius: 4px;">
                <p style="margin: 0;"><strong>🔄 Choose Your Machine:</strong></p>
                <p style="margin: 5px 0 0 0; font-size: 14px; color: #ecf0f1;">{', '.join(cardio['options'])}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Mental training/yoga recommendation
    if exercise_recs['mental_training']:
        st.markdown('<h2 class="section-header">🧘 Mental Training</h2>', unsafe_allow_html=True)
        yoga = exercise_recs['mental_training']
        st.markdown(f"""
        <div class="recommendation-box">
            <h4>{yoga['type']}</h4>
            <p><strong>Duration:</strong> {yoga['duration_min']} minutes</p>
            <p><strong>Why This Matters:</strong> {yoga['reason']}</p>
            <div style="background: rgba(100, 150, 200, 0.1); border-left: 4px solid #6496C8; padding: 12px; margin-top: 12px; border-radius: 4px;">
                <p style="margin: 0;"><strong>💡 Pro Tips:</strong></p>
                <p style="margin: 5px 0 0 0; font-size: 14px; color: #ecf0f1;">{yoga['tips']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- POST-WORKOUT FEEDBACK LOOP (Minimal UI) ---
    st.markdown('<h2 class="section-header">📊 Post-Workout Feedback</h2>', unsafe_allow_html=True)
    
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        workout_completion = st.text_input(
            "Workout Completion (%)",
            value="100"
        )
        actual_intensity = st.text_input(
            "Actual Intensity (1-10)",
            value=str(int(intensity))
        )
        workout_difficulty = st.selectbox(
            "Difficulty",
            options=["Too Easy", "Just Right", "Challenging", "Too Hard"]
        )
    
    with feedback_col2:
        recovery_feeling = st.text_input(
            "Recovery Feeling (1-5)",
            value="3"
        )
        soreness_now = st.text_input(
            "Expected Soreness Tomorrow (1-5)",
            value=str(user_soreness)
        )
        would_repeat = st.selectbox(
            "Would Repeat?",
            options=["Yes", "Probably", "Maybe", "No"]
        )
    
    # Convert inputs to numbers
    try:
        workout_completion = int(workout_completion)
        actual_intensity = float(actual_intensity)
        recovery_feeling = int(recovery_feeling)
        soreness_now = int(soreness_now)
    except ValueError:
        st.error("Please enter valid numbers for all feedback fields")
        st.stop()
    
    # Feedback submission button
    if st.button("Submit Feedback", type="primary", use_container_width=True):
        # Prepare feedback record
        feedback_record = {
            'timestamp': pd.Timestamp.now(),
            'age': user_age,
            'sex': user_sex,
            'weight_kg': user_weight,
            'sleep_hrs': user_sleep,
            'rhr_bpm': user_rhr,
            'soreness_before': user_soreness,
            'mental_stress': user_stress,
            'calories_in': nutr_features['CALORIES_IN'],
            'protein_g': nutr_features['PROTEIN_G'],
            'carbs_g': nutr_features['CARBS_G'],
            'predicted_goal': predicted_goal,
            'recommended_duration': duration,
            'recommended_intensity': intensity,
            'workout_completion_pct': workout_completion,
            'actual_intensity': actual_intensity,
            'difficulty_feedback': workout_difficulty,
            'recovery_feeling': recovery_feeling,
            'soreness_next_day_expected': soreness_now,
            'would_repeat': would_repeat
        }
        
        # Save to MongoDB
        try:
            mongo_handler = get_mongodb_handler()
            if mongo_handler.is_connected():
                # Save to MongoDB
                if mongo_handler.save_feedback(feedback_record):
                    st.success("Feedback submitted.")
                    
                    # OPTION A: Background thread retraining after feedback
                    # Trigger background retraining if 3+ new feedbacks accumulated
                    def background_retrain():
                        try:
                            last_retrain_count = mongo_handler.get_last_retrain_count()
                            current_count = mongo_handler.get_feedback_count()
                            
                            # Only retrain if 3+ new feedbacks since last retrain
                            if current_count >= last_retrain_count + 3:
                                feedback_df = mongo_handler.get_all_feedback()
                                if not feedback_df.empty and len(feedback_df) >= 3:
                                    retrain_and_update_models(feedback_df, current_count)
                                    mongo_handler.update_last_retrain_count(current_count)
                        except Exception as e:
                            # Silent fail - background retraining should not interrupt user
                            pass
                    
                    # Start background thread (daemon=True so it doesn't block app shutdown)
                    retrain_thread = threading.Thread(target=background_retrain, daemon=True)
                    retrain_thread.start()
                    
                else:
                    st.error("Error saving feedback to database.")
            else:
                # Fallback to CSV if MongoDB unavailable
                feedback_df = pd.DataFrame([feedback_record])
                try:
                    if os.path.exists('feedback_history.csv'):
                        existing_feedback = pd.read_csv('feedback_history.csv')
                        updated_feedback = pd.concat([existing_feedback, feedback_df], ignore_index=True)
                    else:
                        updated_feedback = feedback_df
                    
                    updated_feedback.to_csv('feedback_history.csv', index=False)
                    st.success("Feedback submitted.")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p style='font-size: 18px; font-weight: 700;'>�️ TRAIN HARD. PUSH LIMITS.</p>
        <p style='font-size: 14px;'>AI-Powered Personalized Workouts | ChronoFit MVAA</p>
        <p style='font-size: 12px; margin-top: 15px;'>Your feedback powers continuous AI improvement </p>
    </div>
    """, unsafe_allow_html=True)
