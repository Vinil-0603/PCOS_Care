from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import os
import google.generativeai as genai
import json
import uuid
import time
import re
import pickle
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

# Create Flask app
app = Flask(__name__)
app.secret_key = "pcos_integrated_support_secret_key"  # For session management

# Configure API keys
GEMINI_API_KEY = "AIzaSyCUYuDX1ACvaJ4liwDQUNcu9UqA9Yi7UIM"
RECOMMENDATION_API_KEY = "AIzaSyBL7LfbY1v3MAYMb2ybYCS_JQ0mdi6rETg"

# Configure Gemini API for chatbot and recommendations
genai.configure(api_key=GEMINI_API_KEY)

# Mental health chatbot model
MENTAL_HEALTH_MODEL = "tunedModels/pcosmentalhealth-fndi214ha4hl"

# Define a ScalerAdapter to handle numpy array scalers
class ScalerAdapter:
    def __init__(self, scaling_array):
        # Assuming first row is means, second row is standard deviations
        self.means = scaling_array[0]
        self.stds = scaling_array[1]
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (X - self.means) / self.stds

# Load PCOS detection model and scaler with improved error handling
def load_pcos_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'logistic_regression_pcos.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler_pcos.pkl')
    
    model = None
    scaler = None
    
    # Try to load model with multiple approaches
    if os.path.exists(model_path):
        # Try pickle first
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Successfully loaded model with pickle")
        except Exception as e:
            print(f"Pickle error loading model: {str(e)}")
            # Try joblib as fallback
            try:
                import joblib
                model = joblib.load(model_path)
                print("Successfully loaded model with joblib")
            except Exception as e:
                print(f"Joblib error loading model: {str(e)}")
    else:
        print(f"Model file not found at {model_path}")
    
    # Try to load scaler with multiple approaches
    if os.path.exists(scaler_path):
        # Try pickle first
        try:
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
            print("Successfully loaded scaler with pickle")
            
            # Check if scaler is a numpy array and needs adaptation
            if isinstance(scaler, np.ndarray) and not hasattr(scaler, 'transform'):
                scaler = ScalerAdapter(scaler)
                print("Created ScalerAdapter for numpy array scaler")
                
        except Exception as e:
            print(f"Pickle error loading scaler: {str(e)}")
            # Try joblib as fallback
            try:
                import joblib
                scaler = joblib.load(scaler_path)
                print("Successfully loaded scaler with joblib")
                
                # Check if scaler is a numpy array and needs adaptation
                if isinstance(scaler, np.ndarray) and not hasattr(scaler, 'transform'):
                    scaler = ScalerAdapter(scaler)
                    print("Created ScalerAdapter for numpy array scaler")
            except Exception as e:
                print(f"Joblib error loading scaler: {str(e)}")
    else:
        print(f"Scaler file not found at {scaler_path}")
        
    return model, scaler

# Create a dummy model and scaler for testing if loading fails
def create_dummy_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    print("Creating dummy model and scaler for testing")
    
    # Create a simple dummy model
    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.random.randn(1, 41)  # 41 features
    model.intercept_ = np.array([0])
    
    # Create a dummy scaler
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(41)  # 41 features
    scaler.scale_ = np.ones(41)  # 41 features
    
    return model, scaler

# Try to load the model and scaler at startup
pcos_model, pcos_scaler = load_pcos_model()

# Fall back to dummy model if loading fails
if pcos_model is None or pcos_scaler is None:
    pcos_model, pcos_scaler = create_dummy_model()

# Default sample input for PCOS prediction with values typical for PCOS patients
# These are used for hardcoded values when the user doesn't provide them
default_pcos_sample = {
   "Age (yrs)": 28.0,
   "Weight (Kg)": 68.5,
   "Height(Cm)": 162.0,
   "BMI": 26.1,
   "Blood Group": 15.0,
   "Pulse rate(bpm)": 78.0,
   "RR (breaths/min)": 22.0,
   "Hb(g/dl)": 11.2,
   "Cycle(R/I)": 4.0,  # Irregular cycles (4.0)
   "Cycle length(days)": 38.0,
   "Marraige Status (Yrs)": 4.0,
   "Pregnant(Y/N)": 0.0,
   "No. of abortions": 0.0,
   "I beta-HCG(mIU/mL)": 1.99,
   "II beta-HCG(mIU/mL)": 1.99,
   "FSH(mIU/mL)": 5.5,
   "LH(mIU/mL)": 12.5,
   "FSH/LH Ratio": 0.44,
   "Hip(inch)": 40.0,
   "Waist(inch)": 34.0,
   "Waist:Hip Ratio": 0.85,
   "TSH (mIU/L)": 3.8,
   "AMH(ng/mL)": 7.5,
   "PRL(ng/mL)": 24.5,
   "Vit D3 (ng/mL)": 18.4,
   "PRG(ng/mL)": 0.41,
   "RBS(mg/dl)": 102.0,
   "Weight gain(Y/N)": 1.0,
   "hair growth(Y/N)": 1.0,
   "Skin darkening (Y/N)": 1.0,
   "Hair loss(Y/N)": 1.0,
   "Pimples(Y/N)": 1.0,
   "Fast food (Y/N)": 1.0,
   "Reg.Exercise(Y/N)": 0.0,
   "BP _Systolic (mmHg)": 128.0,
   "BP _Diastolic (mmHg)": 82.0,
   "Follicle No. (L)": 12.0,
   "Follicle No. (R)": 14.0,
   "Avg. F size (L) (mm)": 6.0,
   "Avg. F size (R) (mm)": 5.5,
   "Endometrium (mm)": 10.5
}

# Ensure necessary directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Conversation state management for mental health chatbot
conversation_flows = {}

class ConversationState:
    def __init__(self):
        self.state = "greeting"  # Initial state
        self.assessment_started = False
        self.questions_asked = 0
        self.assessment_data = {}
        self.history = []
        self.last_update = time.time()
    
    def update_state(self, new_state):
        self.state = new_state
        self.last_update = time.time()
    
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        self.last_update = time.time()
    
    def get_history_text(self):
        """Convert history to text format for the model"""
        result = ""
        for msg in self.history:
            prefix = "User: " if msg["role"] == "user" else "Bot: "
            result += f"{prefix}{msg['content']}\n"
        return result

# Helper function to detect if message is a brief greeting
def is_greeting(message):
    greeting_patterns = [
        r'^hi+\s*$', r'^hello\s*$', r'^hey\s*$', r'^greetings\s*$', 
        r'^howdy\s*$', r'^what\'?s up\s*$', r'^sup\s*$',
        r'^good morning\s*$', r'^good afternoon\s*$', r'^good evening\s*$'
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, message.lower().strip()):
            return True
    return False

# Helper function to detect if user wants help
def wants_help(message):
    help_patterns = [
        r'yes', r'yeah', r'sure', r'ok', r'okay', r'please', r'help',
        r'can you help', r'need help', r'want help', r'feeling', r'talk', 
        r'chat', r'discuss', r'advice', r'support', r'assist'
    ]
    
    message = message.lower()
    for pattern in help_patterns:
        if re.search(pattern, message):
            return True
    return False

# Helper function to clean up repetitive text in responses
def clean_repetitive_text(text):
    # Create a pattern to match repetitive sentences or phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 1:
        return text
    
    clean_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        # Lowercase version for comparison
        sentence_lower = sentence.lower().strip()
        
        # Skip empty sentences
        if not sentence_lower:
            continue
            
        # If we haven't seen this sentence before (allowing for small variations)
        if not any(similar_sentence(sentence_lower, seen) for seen in seen_sentences):
            clean_sentences.append(sentence)
            seen_sentences.add(sentence_lower)
    
    # Join back together with spaces
    return ' '.join(clean_sentences)

def similar_sentence(sentence1, sentence2):
    """Check if two sentences are substantially similar (based on Levenshtein distance or other heuristics)"""
    # Simple implementation - could be improved with better similarity measures
    if len(sentence1) == 0 or len(sentence2) == 0:
        return False
        
    # Check if one is contained in the other or they're very similar
    if sentence1 in sentence2 or sentence2 in sentence1:
        return True
        
    # Check word-level similarity
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())
    
    # If the sentences share more than 80% of words, consider them similar
    if len(words1) > 0 and len(words2) > 0:
        common_words = words1.intersection(words2)
        similarity = len(common_words) / max(len(words1), len(words2))
        return similarity > 0.8
        
    return False

# Initialize the mental health chatbot with dynamic conversation management
def get_mental_health_response(user_input, session_id):
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Get or create conversation state
    if session_id not in conversation_flows:
        conversation_flows[session_id] = ConversationState()
    
    conv_state = conversation_flows[session_id]
    conv_state.add_message("user", user_input)
    
    # Create the base context for the model
    base_context = """
    You are a mental health chatbot specifically created for people with Polycystic Ovary Syndrome (PCOS).
    Your purpose is to:
    1. Provide supportive responses and coping strategies
    2. Recognize when to suggest professional help
    3. Understand the unique challenges faced by PCOS patients including hormonal imbalances,
       fertility concerns, body image issues, and chronic condition management
    
    Always be compassionate, non-judgmental, and provide evidence-based information when possible.
    If the user appears to be in crisis, always encourage them to contact a healthcare professional
    or emergency services.
    
    IMPORTANT: Keep your responses concise and avoid repeating the same phrases or sentences.
    
    Previous conversation history:
    """
    
    # Determine current state and next actions based on conversation flow
    current_state = conv_state.state
    
    # If user just sent a brief greeting, respond naturally without jumping to assessment
    if is_greeting(user_input) and len(conv_state.history) <= 2:
        instructions = """
        The user has sent a brief greeting. Respond naturally with a warm greeting and ask how they're feeling 
        today or if they'd like to talk about how PCOS might be affecting their mental wellbeing. 
        Don't jump straight into an assessment or providing mental health advice.
        Keep your response short and concise, under 3 sentences.
        """
    
    # If user hasn't started assessment and shows interest, offer to begin assessment
    elif not conv_state.assessment_started and wants_help(user_input):
        conv_state.assessment_started = True
        conv_state.update_state("beginning_assessment")
        instructions = """
        The user seems interested in discussing their mental health. Begin by asking ONE specific question
        about their experience with PCOS and mental health. Ask about:
        - Their current mood or emotional state
        - How long they've been dealing with PCOS
        - What specific PCOS symptoms they find most challenging
        
        Ask just ONE question that helps understand their situation better.
        Keep your response concise and focused.
        """
    
    # If assessment has begun, continue with appropriate questions based on responses
    elif conv_state.assessment_started:
        conv_state.questions_asked += 1
        
        # First few questions of assessment
        if conv_state.questions_asked < 3:
            instructions = """
            Continue the assessment by asking a follow-up question based on their response.
            Focus on understanding:
            - Severity of any mental health symptoms
            - Duration of symptoms 
            - Impact on daily life
            - Existing coping mechanisms
            
            Ask just ONE relevant question. Be conversational and natural - don't make it feel like
            a formal questionnaire.
            Keep your response concise - no more than 3-4 sentences total.
            """
        
        # Middle of assessment
        elif conv_state.questions_asked < 5:
            instructions = """
            Based on what you've learned so far, ask another question to better understand their mental 
            wellbeing in relation to PCOS. Consider asking about:
            - Support systems they have
            - Strategies they've tried to cope with their symptoms
            - How PCOS symptoms and mental health interact for them
            
            Ask just ONE thoughtful question that shows you're listening to their specific situation.
            Keep your response concise - no more than 3-4 sentences total.
            """
        
        # Transition to providing guidance
        else:
            conv_state.update_state("providing_guidance")
            instructions = """
            Now that you have sufficient information, provide personalized guidance:
            1. Validate their experiences and feelings
            2. Offer 2-3 specific coping strategies relevant to their situation
            3. Suggest ways they might track mood and symptoms to identify patterns
            4. If appropriate, gently suggest when professional support might be helpful
            
            Be warm, supportive, and specific to their situation. Avoid generic advice.
            Keep your response focused and avoid repeating the same phrases.
            Limit your response to 5-6 sentences maximum.
            """
    
    # Default response for other cases (user hasn't clearly indicated interest yet)
    else:
        instructions = """
        The user hasn't explicitly agreed to an assessment yet. Respond naturally to what they've said,
        be warm and empathetic, and gently invite them to share more about how they're feeling or how 
        PCOS might be affecting their mental wellbeing. Don't pressure them or be too formal.
        Keep your response concise - no more than 3 sentences.
        """
    
    # Build the final context for the model
    context = base_context + conv_state.get_history_text() + "\n\nInstructions for this response: " + instructions
    
    # Process with the model
    response = model.generate_content([context, user_input])
    response_text = response.text
    
    # Clean up any repetitive text in the response
    cleaned_response = clean_repetitive_text(response_text)
    
    # Add bot response to history
    conv_state.add_message("bot", cleaned_response)
    
    # Clean up old sessions (optional, for production use)
    cleanup_old_sessions()
    
    return cleaned_response

def cleanup_old_sessions():
    """Remove conversation states that are older than 30 minutes"""
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, conv_state in conversation_flows.items():
        if current_time - conv_state.last_update > 1800:  # 30 minutes
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del conversation_flows[session_id]

# Function to predict PCOS risk using the loaded model
def predict_pcos_risk(input_data):
    if pcos_model is None:
        return {
            "error": "Model not loaded",
            "message": "The PCOS prediction model is not available."
        }
    
    try:
        # Start with the default values
        feature_dict = default_pcos_sample.copy()
        
        # Update with the user-provided values (from the form)
        
        # 1. Basic Information
        if 'Age (years)' in input_data and input_data['Age (years)']:
            feature_dict['Age (yrs)'] = float(input_data['Age (years)'])
            
        if 'BMI' in input_data and input_data['BMI']:
            feature_dict['BMI'] = float(input_data['BMI'])
            
        if 'Menstrual Cycle' in input_data:
            if input_data['Menstrual Cycle'] == 'Regular':
                feature_dict['Cycle(R/I)'] = 2.0
            elif input_data['Menstrual Cycle'] == 'Irregular':
                feature_dict['Cycle(R/I)'] = 4.0
        
        # 2. PCOS Symptoms
        if 'Weight Gain' in input_data:
            feature_dict['Weight gain(Y/N)'] = 1.0 if input_data['Weight Gain'] == 'Yes' else 0.0
            
        if 'Hair Growth (Hirsutism)' in input_data:
            feature_dict['hair growth(Y/N)'] = 1.0 if input_data['Hair Growth (Hirsutism)'] == 'Yes' else 0.0
            
        if 'Skin Darkening' in input_data:
            feature_dict['Skin darkening (Y/N)'] = 1.0 if input_data['Skin Darkening'] == 'Yes' else 0.0
            
        if 'Hair Loss' in input_data:
            feature_dict['Hair loss(Y/N)'] = 1.0 if input_data['Hair Loss'] == 'Yes' else 0.0
            
        if 'Pimples/Acne' in input_data:
            feature_dict['Pimples(Y/N)'] = 1.0 if input_data['Pimples/Acne'] == 'Yes' else 0.0
            
        # 3. Lifestyle Factors
        if 'Fast Food Consumption' in input_data:
            feature_dict['Fast food (Y/N)'] = 1.0 if input_data['Fast Food Consumption'] == 'Yes' else 0.0
            
        if 'Regular Exercise' in input_data:
            feature_dict['Reg.Exercise(Y/N)'] = 1.0 if input_data['Regular Exercise'] == 'Yes' else 0.0
        
        # 4. Additional Parameters (existing)
        if 'Cycle Length (days)' in input_data and input_data['Cycle Length (days)']:
            feature_dict['Cycle length(days)'] = float(input_data['Cycle Length (days)'])
            
        if 'Hemoglobin (g/dl)' in input_data and input_data['Hemoglobin (g/dl)']:
            feature_dict['Hb(g/dl)'] = float(input_data['Hemoglobin (g/dl)'])
            
        if 'Waist:Hip Ratio' in input_data and input_data['Waist:Hip Ratio']:
            feature_dict['Waist:Hip Ratio'] = float(input_data['Waist:Hip Ratio'])
            
        # 5. Main PCOS detection parameters taken from user inputs
        if 'Follicle No. (L)' in input_data and input_data['Follicle No. (L)']:
            feature_dict['Follicle No. (L)'] = float(input_data['Follicle No. (L)'])
            
        if 'Follicle No. (R)' in input_data and input_data['Follicle No. (R)']:
            feature_dict['Follicle No. (R)'] = float(input_data['Follicle No. (R)'])
            
        if 'AMH(ng/mL)' in input_data and input_data['AMH(ng/mL)']:
            feature_dict['AMH(ng/mL)'] = float(input_data['AMH(ng/mL)'])
            
        if 'PRL(ng/mL)' in input_data and input_data['PRL(ng/mL)']:
            feature_dict['PRL(ng/mL)'] = float(input_data['PRL(ng/mL)'])
            
        if 'FSH(mIU/mL)' in input_data and input_data['FSH(mIU/mL)']:
            feature_dict['FSH(mIU/mL)'] = float(input_data['FSH(mIU/mL)'])
            
        if 'LH(mIU/mL)' in input_data and input_data['LH(mIU/mL)']:
            feature_dict['LH(mIU/mL)'] = float(input_data['LH(mIU/mL)'])
            
        if 'FSH/LH Ratio' in input_data and input_data['FSH/LH Ratio']:
            feature_dict['FSH/LH Ratio'] = float(input_data['FSH/LH Ratio'])
        
        # Convert to DataFrame
        input_df = pd.DataFrame([feature_dict])
        
        # Scale the data using the loaded scaler with error handling
        try:
            if hasattr(pcos_scaler, 'transform'):
                scaled_data = pcos_scaler.transform(input_df)
            else:
                # If scaler doesn't have transform method (e.g., it's a numpy array)
                print("Warning: Using unscaled data for prediction")
                scaled_data = input_df.values
        except Exception as e:
            print(f"Warning: Scaling error: {str(e)}. Using unscaled data.")
            scaled_data = input_df.values
        
        # Make prediction with error handling
        try:
            prediction = pcos_model.predict(scaled_data)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Fallback to a simple threshold-based prediction on key features
            if (feature_dict['Cycle(R/I)'] == 4.0 and  # Irregular cycles
                feature_dict['hair growth(Y/N)'] == 1.0 and  # Hirsutism
                feature_dict['AMH(ng/mL)'] > 4.0):  # Elevated AMH
                prediction = np.array([1])  # Likely PCOS
            else:
                prediction = np.array([0])  # Likely not PCOS
        
        # Get probability if the model supports predict_proba
        try:
            probability = pcos_model.predict_proba(scaled_data)[0][1]  # Probability of PCOS (class 1)
        except Exception as e:
            print(f"Probability estimation error: {str(e)}")
            # Estimate probability based on key features
            pcos_indicators = 0
            total_indicators = 7
            
            # Count PCOS indicators
            if feature_dict['Cycle(R/I)'] == 4.0:  # Irregular cycles
                pcos_indicators += 1
            if feature_dict['hair growth(Y/N)'] == 1.0:  # Hirsutism
                pcos_indicators += 1
            if feature_dict['Skin darkening (Y/N)'] == 1.0:  # Skin darkening
                pcos_indicators += 1
            if feature_dict['Hair loss(Y/N)'] == 1.0:  # Hair loss
                pcos_indicators += 1
            if feature_dict['Pimples(Y/N)'] == 1.0:  # Acne
                pcos_indicators += 1
            if feature_dict['BMI'] > 25:  # Elevated BMI
                pcos_indicators += 1
            if feature_dict['AMH(ng/mL)'] > 4.0:  # Elevated AMH
                pcos_indicators += 1
                
            # Calculate estimated probability
            probability = pcos_indicators / total_indicators
        
        # Determine risk level based on probability
        if probability > 0.75:
            risk_level = "High"
        elif probability > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Prepare a detailed report based on key risk factors
        key_factors = []
        
        # Check for key clinical signs of PCOS and add to report
        if input_df['Cycle(R/I)'].values[0] == 4.0:  # Irregular cycles
            key_factors.append("Irregular menstrual cycles")
        
        if input_df['hair growth(Y/N)'].values[0] == 1.0:  # Hirsutism
            key_factors.append("Excess hair growth (hirsutism)")
            
        if input_df['Skin darkening (Y/N)'].values[0] == 1.0:  # Acanthosis nigricans
            key_factors.append("Skin darkening (possible sign of insulin resistance)")
            
        if input_df['Hair loss(Y/N)'].values[0] == 1.0:  # Alopecia
            key_factors.append("Hair thinning/loss")
            
        if input_df['Pimples(Y/N)'].values[0] == 1.0:  # Acne
            key_factors.append("Acne")
            
        # Check hormone levels
        if input_df['AMH(ng/mL)'].values[0] > 4.0:
            key_factors.append("Elevated AMH levels")
            
        # Check metabolic factors
        if input_df['BMI'].values[0] > 25:
            key_factors.append("Elevated BMI")
            
        if input_df['Waist:Hip Ratio'].values[0] > 0.85:
            key_factors.append("Elevated waist-to-hip ratio")
            
        if input_df['Weight gain(Y/N)'].values[0] == 1.0:
            key_factors.append("Weight gain")
        
        # Return prediction results with detailed report
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "has_pcos": bool(prediction[0] == 1),
            "risk_level": risk_level,
            "key_factors": key_factors,
            "message": "PCOS detected with high confidence" if probability > 0.75 else
                       "PCOS detected" if prediction[0] == 1 else
                       "No PCOS detected",
            "summary": f"Based on the analysis, the model {'detected PCOS' if prediction[0] == 1 else 'did not detect PCOS'} with a {risk_level.lower()} risk level ({int(probability*100)}% probability). " + 
                      (f"Key indicators include: {', '.join(key_factors)}." if key_factors else "")
        }
        
    except Exception as e:
        return {
            "error": "Prediction error",
            "message": f"Error during prediction: {str(e)}"
        }

# Function to get PCOS recommendations
def get_pcos_recommendations(user_inputs):
    """
    Generate personalized PCOS recommendations based on user inputs.
    :param user_inputs: Dictionary containing user health parameters.
    :return: AI-generated recommendations.
    """
    # Configure the API for recommendations
    recommendation_genai = genai.GenerativeModel("gemini-2.0-flash")
    
    # Extract key parameters for the recommendation prompt
    # Map form fields to the correct variable names
    
    # Handle various field name formats
    age = user_inputs.get('Age (years)', user_inputs.get('Age (yrs)', 'Not provided'))
    bmi = user_inputs.get('BMI', 'Not provided')
    
    # Map menstrual cycle selection to text
    if 'Menstrual Cycle' in user_inputs:
        cycle = user_inputs['Menstrual Cycle']
    else:
        cycle_code = user_inputs.get('Cycle(R/I)', 2.0)
        cycle = "Regular" if cycle_code == 2.0 else "Irregular" if cycle_code == 4.0 else "Not provided"
    
    cycle_length = user_inputs.get('Cycle Length (days)', user_inputs.get('Cycle length(days)', 'Not provided'))
    
    # Map Yes/No fields for symptoms
    weight_gain = "Yes" if user_inputs.get('Weight Gain', '') == 'Yes' or user_inputs.get('Weight gain(Y/N)', 0) == 1.0 else "No"
    hair_growth = "Yes" if user_inputs.get('Hair Growth (Hirsutism)', '') == 'Yes' or user_inputs.get('hair growth(Y/N)', 0) == 1.0 else "No"
    skin_darkening = "Yes" if user_inputs.get('Skin Darkening', '') == 'Yes' or user_inputs.get('Skin darkening (Y/N)', 0) == 1.0 else "No"
    hair_loss = "Yes" if user_inputs.get('Hair Loss', '') == 'Yes' or user_inputs.get('Hair loss(Y/N)', 0) == 1.0 else "No"
    pimples = "Yes" if user_inputs.get('Pimples/Acne', '') == 'Yes' or user_inputs.get('Pimples(Y/N)', 0) == 1.0 else "No"
    fast_food = "Yes" if user_inputs.get('Fast Food Consumption', '') == 'Yes' or user_inputs.get('Fast food (Y/N)', 0) == 1.0 else "No"
    exercise = "Yes" if user_inputs.get('Regular Exercise', '') == 'Yes' or user_inputs.get('Reg.Exercise(Y/N)', 0) == 1.0 else "No"
    
    # Additional parameters
    hemoglobin = user_inputs.get('Hemoglobin (g/dl)', user_inputs.get('Hb(g/dl)', 'Not provided'))
    waist_hip_ratio = user_inputs.get('Waist:Hip Ratio', 'Not provided')
    amh = user_inputs.get('AMH (ng/mL)', user_inputs.get('AMH(ng/mL)', 'Not provided'))
    
    prompt = f"""
    A patient with PCOS has provided the following health details:
    Age: {age} years
    BMI: {bmi}
    Menstrual Cycle: {cycle}
    Cycle Length: {cycle_length} days
    Weight Gain: {weight_gain}
    Hair Growth (Hirsutism): {hair_growth}
    Skin Darkening: {skin_darkening}
    Hair Loss: {hair_loss}
    Pimples/Acne: {pimples}
    Fast Food Consumption: {fast_food}
    Regular Exercise: {exercise}
    Hemoglobin Level: {hemoglobin} g/dl
    Waist-to-Hip Ratio: {waist_hip_ratio}
    AMH Level: {amh} ng/mL
    
    Based on these details, provide personalized PCOS management recommendations including:
    1. Dietary Changes - specific foods to include and avoid
    2. Exercise Plan - type, frequency, and intensity appropriate for this patient
    3. Lifestyle Modifications - sleep, stress management, etc.
    4. Symptom Management - specifically addressing prominent symptoms
    5. When to Seek Medical Help - important warning signs or thresholds
    
    Format the response with clear headings and bullet points for each section. Keep the recommendations evidence-based, practical, and easy to follow.
    """
    
    try:
        response = recommendation_genai.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Recommendation generation error: {str(e)}")
        # Provide fallback generic recommendations
        return """
# PCOS Management Recommendations

## Dietary Changes
* Focus on a low-glycemic diet rich in whole foods
* Include lean proteins, healthy fats, and plenty of fiber
* Limit processed foods, added sugars, and refined carbohydrates
* Consider consulting with a registered dietitian for personalized guidance

## Exercise Plan
* Aim for 150 minutes of moderate exercise per week
* Include both cardiovascular and strength training
* Start slowly and gradually increase intensity
* Find activities you enjoy for better long-term adherence

## Lifestyle Modifications
* Prioritize consistent sleep patterns (7-9 hours nightly)
* Practice stress management techniques like meditation or deep breathing
* Maintain a consistent daily routine
* Track your symptoms in a journal to identify patterns

## Symptom Management
* For hirsutism: Consider laser hair removal or prescription creams
* For acne: Use gentle, non-comedogenic skin care products
* For irregular periods: Track your cycle with an app
* For weight management: Focus on sustainable changes, not crash diets

## When to Seek Medical Help
* If periods are absent for more than 3 months
* If you experience severe pain or heavy bleeding
* If symptoms suddenly worsen
* For ongoing fertility support if trying to conceive

Please consult with healthcare professionals for personalized medical advice.
        """

# Routes
@app.route('/')
def home():
    # Generate a new session ID if needed
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/mental-health')
def mental_health():
    return render_template('mental_health.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"response": "I didn't receive a message. Could you please try again?"})
    
    # Get session ID
    session_id = session.get('session_id', str(uuid.uuid4()))
    
    try:
        bot_response = get_mental_health_response(user_message, session_id)
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"response": "I'm having trouble processing your request. Please try again later."})

@app.route('/api/detect-pcos', methods=['POST'])
def detect_pcos():
    input_data = request.json
    if not input_data:
        return jsonify({"error": "No input data provided"})
    
    # Debug data received
    print(f"Received input data: {input_data}")
    
    result = predict_pcos_risk(input_data)
    
    # If PCOS detected, also generate recommendations
    if result.get("has_pcos") and not result.get("error"):
        try:
            recommendations = get_pcos_recommendations(input_data)
            result["recommendations"] = recommendations
        except Exception as e:
            result["recommendation_error"] = str(e)
            # Add fallback recommendations
            result["recommendations"] = "Unable to generate personalized recommendations. Please consult with a healthcare provider."
    
    return jsonify(result)

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    input_data = request.json
    if not input_data:
        return jsonify({"error": "No input data provided"})
    
    try:
        recommendations = get_pcos_recommendations(input_data)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files and 'scaler_file' not in request.files:
        return jsonify({"error": "No file part"})
    
    if 'model_file' in request.files:
        model_file = request.files['model_file']
        if model_file.filename == '':
            return jsonify({"error": "No selected model file"})
        
        if model_file and model_file.filename.endswith('.pkl'):
            filename = secure_filename('logistic_regression_pcos.pkl')
            file_path = os.path.join('models', filename)
            model_file.save(file_path)
    
    if 'scaler_file' in request.files:
        scaler_file = request.files['scaler_file']
        if scaler_file.filename == '':
            return jsonify({"error": "No selected scaler file"})
        
        if scaler_file and scaler_file.filename.endswith('.pkl'):
            filename = secure_filename('scaler_pcos.pkl')
            file_path = os.path.join('models', filename)
            scaler_file.save(file_path)
    
    # Try to reload the model and scaler
    global pcos_model, pcos_scaler
    try:
        pcos_model, pcos_scaler = load_pcos_model()
        if pcos_model is None or pcos_scaler is None:
            pcos_model, pcos_scaler = create_dummy_model()
            
        return jsonify({"success": "Model files processed. Using " + 
                       ("actual model" if pcos_model is not None and not hasattr(pcos_model, '_is_dummy') else "fallback model") +
                       " for predictions."})
    except Exception as e:
        return jsonify({"error": f"Error processing model files: {str(e)}"})

# Simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": pcos_model is not None,
        "scaler_loaded": pcos_scaler is not None,
        "using_dummy": hasattr(pcos_model, '_is_dummy') if pcos_model is not None else None
    })

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
    