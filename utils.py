"""Module providing all functions."""

# Standard Library Imports
import os
import re
import time

# Third-Party Imports
import pandas as pd
import google.generativeai as genai

from config import API_KEY, SOURCE_FILE, TARGET_FILE, GLOSSARY_FILE
from config import generation_config


# Language settings
SOURCE_LANG = "English"
TARGET_LANG = "Spanish"


# --- Helper functions ---

# Read input files
def read_file(filepath):
    with open(filepath, "r") as file:
        return file.read().strip()

# Segment text into sentences
def segment_into_sentences(paragraph):
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    return re.split(sentence_endings, paragraph)

# Extract post-edited translations
def extract_translations(text, lang):
    pattern = rf'Post-edited translation \d+ \({lang}\):\s*?(.+)'
    return re.findall(pattern, text)

def load_glossary(file_path):
    df_terms = pd.read_csv(file_path)
    return pd.Series(df_terms.target_term.values, index=df_terms.source_term).to_dict()

def identify_relevant_terms(source, glossary):
    return {term: glossary[term] for term in glossary if term in source}

def format_pe_step(relevant_terms):
    if not relevant_terms:
        return "No specific glossary terms to focus on."
    formatted_terms = "\n".join([f"'{source_term}' to '{target_term}'" for source_term, target_term in relevant_terms.items()])
    return f"Translate: \n{formatted_terms}"

def create_system_message_with_source(pe_step):
    return f"""
    You are given a sentence in {SOURCE_LANG} and its translation into {TARGET_LANG}.
    Post-edit the translation in {TARGET_LANG} focusing only on the following:
    {pe_step}
    Do NOT translate the translation back to {SOURCE_LANG}.
    Output the post-edited translation in {TARGET_LANG} and nothing else.
    """

def create_system_message_without_source(pe_step):
    return f"""
    Edit the given sentence in {TARGET_LANG} focusing only on the following:
    {pe_step}
    Do NOT translate the translation back to {SOURCE_LANG}.
    Output the post-edited translation in {TARGET_LANG} and nothing else.
    """

def create_prompt(system_message, original, translation):
    prompt = [system_message]
    for i, (st, tt) in enumerate(zip(original, translation), start=1):
        prompt.append(
            f"Original {i} ({SOURCE_LANG}): {st}\n"
            f"Translation {i} ({TARGET_LANG}): {tt}\n"
            f"Post-edited translation {i} ({TARGET_LANG}):\n"
        )
    return prompt

def create_prompt_without_source(system_message, translation):
    prompt = [system_message]
    for i, tt in enumerate(translation, start=1):
        prompt.append(
            f"Translation {i} ({TARGET_LANG}): {tt}\n"
            f"Post-edited translation {i} ({TARGET_LANG}):\n"
        )
    return prompt

# Print prompt data
def print_prompt_data(data):
    for entry in data:
        print(entry.strip())
        print("\n---\n")

# Clean text
def clean_text(text):
    """
    Cleans the input text by removing double asterisks and normalizing whitespace.
    Parameters:
        text (str): The string to be cleaned.
    Returns:
        str: The cleaned text.
    """
    # Remove double asterisks
    text = re.sub(r'\*\*', '', text)
    # Normalize whitespace (remove extra spaces and trim)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Save DataFrame to Excel file
def save_df_to_excel(df, base_filename='ori-tra-pes.xlsx'):
    """
    Save the DataFrame to an Excel file with a unique filename. 
    If a file with this name already exists, a counter will be appended to the base name to generate a unique filename.
    df: DataFrame to be saved. / base_filename: The base filename for the Excel file.
    """
    file_root, file_ext = os.path.splitext(base_filename)
    # Create a new filename if the file already exists
    counter = 2
    new_filename = base_filename
    while os.path.exists(new_filename):
        new_filename = f"{file_root}_{counter}{file_ext}"
        counter += 1
    # Save the DataFrame to the new Excel file
    df.to_excel(new_filename, index=False)

# Post-editing steps definition
def MTPE(temp_accuracy=1.7, temp_terms=0.0, temp_fluency=1.5, temp_style=0.5, temp_inclusive=0.0):
    # Start the timer
    start_time = time.time()
    # Configure and initialize the LLM
    genai.configure(api_key=API_KEY)
    llm = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are a professional linguist specializing in post-editing machine-translated content."
    )
    # Prompting user for file details inside the MTPE function

    #SOURCE_FILE = input("\nHello, my name is Elsa.\nI am going to help you post-edit the translation you provide me.\nPlease enter the name of the source file (e.g., source.txt): ")
    #TARGET_FILE = input("\nNow please enter the name of the target file (e.g., target.txt): ")
       
    # Load source and target text
    source = read_file(SOURCE_FILE)
    target = read_file(TARGET_FILE)
    original = segment_into_sentences(source)
    translation = segment_into_sentences(target)
    print("\nSource and target files loaded successfully.")
    
    # Ensure the number of segments in the original and translation match
    if len(original) != len(translation):
        raise ValueError("The number of segments in the original and translation must be the same.")
    
    # Creates DataFrame with original and translation
    
    df = pd.DataFrame({
        'Original': original,
        'Translation': translation
    })
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    
    print("\nPost-editing of the machine translation in process...\n")
   
    # Accuracy post-editing
    accuracy = read_file("Files/accuracy.txt")
    system_message_accuracy = create_system_message_with_source(accuracy)
    prompt_accuracy = create_prompt(system_message_accuracy, original, translation)
    response_accuracy = llm.generate_content(prompt_accuracy, generation_config={'temperature': temp_accuracy})
    matches_accuracy = extract_translations(response_accuracy.text, TARGET_LANG)
    df['Post-edited accuracy'] = matches_accuracy
    
    # Terminology post-editing
    glossary = load_glossary(GLOSSARY_FILE)
    relevant_terms = identify_relevant_terms(source, glossary)
    terms = format_pe_step(relevant_terms)
    system_message_terms = create_system_message_with_source(terms)
    prompt_terms = create_prompt(system_message_terms, original, matches_accuracy)
    response_terms = llm.generate_content(prompt_terms, generation_config={'temperature': temp_terms})
    matches_terms = extract_translations(response_terms.text, TARGET_LANG)
    df['Post-edited terms'] = matches_terms
    
    # Fluency post-editing
    fluency = read_file('Files/fluency.txt')
    system_message_fluency = create_system_message_with_source(fluency)
    prompt_fluency = create_prompt(system_message_fluency, original, matches_terms)
    response_fluency = llm.generate_content(prompt_fluency, generation_config={'temperature': temp_fluency})
    matches_fluency = extract_translations(response_fluency.text, TARGET_LANG)
    df['Post-edited fluency'] = matches_fluency
    
    # Style post-editing
    style = read_file('Files/style.txt')
    system_message_style = create_system_message_without_source(style)
    prompt_style = create_prompt_without_source(system_message_style, matches_fluency)
    response_style = llm.generate_content(prompt_style, generation_config={'temperature': temp_style})
    matches_style = extract_translations(response_style.text, TARGET_LANG)
    df['Post-edited style'] = matches_style
    
    # Inclusive language post-editing
    inclusive = read_file('Files/inclusive.txt')
    system_message_inclusive = create_system_message_without_source(inclusive)
    prompt_inclusive = create_prompt_without_source(system_message_inclusive, matches_style)
    response_inclusive = llm.generate_content(prompt_inclusive, generation_config={'temperature': temp_inclusive})
    matches_inclusive = extract_translations(response_inclusive.text, TARGET_LANG)
    df['Post-edited inclusive'] = matches_inclusive
    
    # Record the end time and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nPost-editing completed! Elapsed time: {elapsed_time:.2f} seconds")
    
    # Save to excel file
    save_df_to_excel(df)
    
    # Final post-edited translation in a single paragraph
    final_post_edited_translation = ' '.join(matches_inclusive)
    
    # Clean the post-edited translation
    clean_post_edited_translation = clean_text(final_post_edited_translation)
    
    # Save to TXT file
    output_text_file = 'post-edited_translation.txt'
    
    # Check if the file already exists
    if os.path.exists(output_text_file):
        # Initialize counter
        counter = 2
        # Loop until we find a non-existing filename
        while os.path.exists(f'post-edited_translation_{counter}.txt'):
            counter += 1
        # Update filename with the new version number
        output_text_file = f'post-edited_translation_{counter}.txt'
    
    # Save the final post-edited translation to the file
    with open(output_text_file, 'w', encoding='utf-8') as file:
        file.write(clean_post_edited_translation)
    print(f"\nAll post-editing results saved to an Excel file and final post-edited translation saved as {output_text_file}.\n")

    # Return the DataFrame for display
    return df