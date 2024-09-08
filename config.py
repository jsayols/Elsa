import os

# Configuration
API_KEY = os.getenv('ENV_VARIABLE_NAME')
SOURCE_FILE = "Files/source.txt"
TARGET_FILE = "Files/target.txt"
GLOSSARY_FILE = "Files/AmnestyTerms.csv"
OUTPUT_FILE = "ori-tra-pes.xlsx"

# Generation configuration
generation_config = {
    "temperature": None, 
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}