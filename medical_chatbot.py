# medical_chatbot.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import json
import random
import re
import warnings
warnings.filterwarnings("ignore")
import nltk
import os

# Set custom NLTK data path
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download required datasets
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

class MedicalChatbot:
    def __init__(self):
        self.name = "MediBot"
        self.disclaimer_shown = False
        
        # Download necessary NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            try:
                nltk.download('punkt')
                nltk.download('wordnet')
            except Exception as e:
                print(f"Error downloading NLTK data: {e}")
                raise
        
        self.lemmatizer = WordNetLemmatizer()
        
        # Load medical intents
        try:
            with open('medical_intents.json', 'r') as file:
                self.intents = json.load(file)
        except FileNotFoundError:
            print("medical_intents.json not found. Creating a new one...")
            self.intents = self._create_basic_intents()
            with open('medical_intents.json', 'w') as file:
                json.dump(self.intents, file, indent=4)
        
        # Load or train model
        try:
            self.model = tf.keras.models.load_model('medical_chatbot_model')
            
            # Load words and classes
            with open('words.json', 'r') as file:
                self.words = json.load(file)
            with open('classes.json', 'r') as file:
                self.classes = json.load(file)
                
            print("Model and data loaded successfully")
        except (OSError, FileNotFoundError):
            print("Model or required files not found. Training a new model...")
            self._train_model()
    
    def _create_basic_intents(self):
        """Create basic medical intents if file doesn't exist"""
        return {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up", "Good day"],
                    "responses": ["Hello! I'm MediBot, a medical AI assistant. How can I help you today?", 
                                 "Hi there! I'm here to provide medical information. What can I help you with?"],
                    "context": [""]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["Bye", "See you later", "Goodbye", "Thanks for your help", "That's all"],
                    "responses": ["Take care! Remember to consult healthcare professionals for personalized medical advice.",
                                 "Goodbye! Please seek professional medical help for any serious concerns."],
                    "context": [""]
                },
                {
                    "tag": "thanks",
                    "patterns": ["Thanks", "Thank you", "That's helpful"],
                    "responses": ["You're welcome! Remember, always consult with healthcare professionals for medical advice.",
                                 "Happy to help! Remember this is general information only."],
                    "context": [""]
                },
                {
                    "tag": "emergency",
                    "patterns": ["can't breathe", "heart attack", "stroke", "unconscious", "severe bleeding", 
                                "suicide", "emergency", "chest pain", "collapsed", "seizure"],
                    "responses": ["This sounds like a medical emergency. Please call emergency services (911 in the US) immediately."],
                    "context": [""]
                },
                {
                    "tag": "headache",
                    "patterns": ["I have a headache", "My head hurts", "Migraine", "Head pain", "Throbbing head"],
                    "responses": ["Headaches can have many causes. Common treatments include rest, hydration, and over-the-counter pain relievers if appropriate. Seek medical attention for severe, sudden, or unusual headaches, especially if accompanied by fever, neck stiffness, confusion, seizures, double vision, weakness, numbness, or difficulty speaking."],
                    "context": [""]
                },
                {
                    "tag": "cold",
                    "patterns": ["I have a cold", "Runny nose", "Sneezing", "Coughing", "Sore throat", "Common cold"],
                    "responses": ["Common cold symptoms typically include runny nose, congestion, sneezing, and sometimes cough or low fever. Rest, staying hydrated, and over-the-counter cold medications may help relieve symptoms. See a doctor if symptoms are severe, last more than 10 days, or if you have difficulty breathing."],
                    "context": [""]
                },
                {
                    "tag": "allergies",
                    "patterns": ["I have allergies", "Itchy eyes", "Hay fever", "Allergic reaction", "Allergy symptoms"],
                    "responses": ["Allergy symptoms include sneezing, itchy eyes, runny nose, and sometimes rashes. Avoiding allergens and taking antihistamines may help. Seek immediate medical attention for severe allergic reactions involving breathing difficulty or swelling."],
                    "context": [""]
                },
                {
                    "tag": "disclaimer",
                    "patterns": ["Are you a doctor", "Medical advice", "Can you diagnose", "Legal disclaimer"],
                    "responses": ["IMPORTANT: I am an AI assistant providing general medical information only. I am not a doctor and cannot diagnose conditions or prescribe treatments. For medical emergencies, call emergency services immediately. Always consult qualified healthcare professionals for personal medical advice."],
                    "context": [""]
                }
            ]
        }
    
    def _train_model(self):
        """Train the NLP model for intent classification"""
        try:
            # Preprocessing
            words = []
            classes = []
            documents = []
            ignore_words = ['?', '!', '.', ',']
            
            # Loop through intents
            for intent in self.intents['intents']:
                for pattern in intent['patterns']:
                    # Tokenize and lemmatize
                    word_list = word_tokenize(pattern.lower())
                    words.extend(word_list)
                    documents.append((word_list, intent['tag']))
                    
                    # Add to classes list
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])
            
            # Lemmatize and remove duplicates
            words = [self.lemmatizer.lemmatize(word) for word in words if word not in ignore_words]
            words = sorted(list(set(words)))
            classes = sorted(list(set(classes)))
            
            # Save for later use
            with open('words.json', 'w') as file:
                json.dump(words, file)
            with open('classes.json', 'w') as file:
                json.dump(classes, file)
            
            self.words = words
            self.classes = classes
            
            # Create training data
            training = []
            output_empty = [0] * len(classes)
            
            for document in documents:
                bag = []
                word_patterns = document[0]
                word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
                
                # Create bag of words
                for word in words:
                    bag.append(1) if word in word_patterns else bag.append(0)
                
                # Output is '1' for current tag and '0' for other tags
                output_row = list(output_empty)
                output_row[classes.index(document[1])] = 1
                
                training.append([bag, output_row])
            
            # Shuffle and convert to numpy array
            random.shuffle(training)
            training = np.array(training, dtype=object)
            
            # Split data
            train_x = list(training[:,0])
            train_y = list(training[:,1])
            
            # Build model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
            ])
            
            # Compile model
            model.compile(loss='categorical_crossentropy', 
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                          metrics=['accuracy'])
            
            # Train model
            hist = model.fit(np.array(train_x), np.array(train_y), 
                             epochs=200, batch_size=5, verbose=1)
            
            # Save model
            model.save('medical_chatbot_model')
            self.model = model
            
            print("Model trained and saved")
        except Exception as e:
            print(f"Error during model training: {e}")
            raise