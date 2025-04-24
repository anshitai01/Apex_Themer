# utils.py
"""
General utility functions for the AI Themer application.
Includes data loading, JSON parsing, batching, and API key validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # Needed if validation checks specific model types

# --- Logging Configuration (Can be set up in main_app.py, but ensure logger is accessible) ---
# Example setup (can be refined in main_app.py)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def batch_responses(responses, batch_size):
    """Split responses into batches."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        logging.warning(f"Invalid batch_size '{batch_size}', defaulting to 1.")
        batch_size = 1
    if not responses:
        return []
    return [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]

def clean_and_parse_json(json_text, expected_type=list):
    """
    Clean and parse JSON text, expecting a specific type (list or dict).

    Handles common issues like markdown code blocks, incorrect escapes,
    and slight variations in API output structure.

    Args:
        json_text (str): The raw text potentially containing JSON.
        expected_type (type): The expected Python type (list or dict).

    Returns:
        The parsed JSON object (list or dict) or None if parsing fails
        or the type is incorrect.
    """
    if not json_text:
        logging.warning("Received empty JSON text to parse.")
        return None
    try:
        # Remove markdown code block fences (```json ... ``` or ``` ... ```)
        cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', json_text.strip(), flags=re.MULTILINE | re.DOTALL).strip()

        # Sometimes the word 'json' might still be at the start after cleaning fences
        if cleaned_text.lower().startswith('json'):
             cleaned_text = cleaned_text[4:].lstrip()

        # Attempt standard parsing first
        try:
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError as e_inner:
             # Try replacing common problematic escapes if basic parsing fails
             logging.warning(f"Initial JSON parsing failed ({e_inner}), attempting to fix common escapes.")
             # More robust escape handling might be needed depending on API behavior
             cleaned_text_escaped = cleaned_text.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
             # Add more replacements if other escape issues are found
             try:
                 parsed_json = json.loads(cleaned_text_escaped)
                 logging.info("Successfully parsed after fixing common escapes.")
             except json.JSONDecodeError as e_escaped:
                 logging.error(f"JSON parsing failed even after escape fixing: {e_escaped}")
                 raise e_escaped # Reraise the error after attempting fix

        # Validate the type
        if isinstance(parsed_json, expected_type):
            return parsed_json
        # Handle list case where API might wrap the list in a dict
        elif expected_type is list and isinstance(parsed_json, dict):
            logging.warning("API returned a dictionary, expected a list. Trying common keys.")
            common_list_keys = ['themes', 'results', 'data', 'assignments', 'descriptions', 'items']
            for key in common_list_keys:
                 if key in parsed_json and isinstance(parsed_json[key], list):
                     logging.info(f"Found list under key '{key}'. Returning nested list.")
                     return parsed_json[key]
            logging.warning("Could not find a nested list under common keys in the dictionary. Returning None.")
            return None
        # Handle dict case where API might return a list containing one dict
        elif expected_type is dict and isinstance(parsed_json, list):
             if len(parsed_json) == 1 and isinstance(parsed_json[0], dict):
                 logging.warning("API returned a list with one dictionary, expected a dictionary. Returning the first element.")
                 return parsed_json[0]
             else:
                logging.error(f"Expected dict, got list with {len(parsed_json)} elements. Cannot reliably convert.")
                return None
        # Type mismatch
        else:
            logging.error(f"Parsed JSON type mismatch: Expected {expected_type}, got {type(parsed_json)}")
            st.error(f"Unexpected JSON structure received from AI. Expected {expected_type}, got {type(parsed_json)}.")
            return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response as JSON: {e}")
        # Only show problematic text in debug/dev or if explicitly enabled
        # st.text_area("Problematic Text:", cleaned_text, height=100)
        logging.error(f"JSONDecodeError: {e}. Problematic text snippet: {cleaned_text[:500]}...")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during JSON parsing: {e}")
        # st.text_area("Problematic Text:", cleaned_text, height=100)
        logging.error(f"Unexpected JSON parsing error: {e}. Problematic text snippet: {cleaned_text[:500]}...")
        return None


def load_data_from_file(uploaded_file):
    """
    Loads data from an uploaded CSV or Excel file into a pandas DataFrame.

    Handles basic cleaning (dropping empty rows/columns) and common
    CSV encoding issues (UTF-8, fallback to latin1). Reads all columns
    as strings initially to avoid type inference issues.

    Args:
        uploaded_file: The file object uploaded via st.file_uploader.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame or None if loading fails.
    """
    df = None
    if uploaded_file is None:
        return None
    try:
        file_name = uploaded_file.name
        logging.info(f"Attempting to load file: {file_name}")
        # Use st.info for user feedback in the main app, keep logging internal
        # st.info(f"Reading file: {file_name}...")

        if file_name.endswith('.csv'):
            try: # Try UTF-8 first
                uploaded_file.seek(0) # Reset file pointer
                df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False) # Read all as string, keep blanks as ''
                logging.info(f"Successfully read CSV '{file_name}' with UTF-8.")
            except UnicodeDecodeError: # Fallback to latin1
                logging.warning(f"UTF-8 decoding failed for '{file_name}', trying latin1.")
                # st.info("UTF-8 failed, trying alternate encoding (latin1)...") # User feedback in main app
                try:
                    uploaded_file.seek(0) # Reset file pointer again
                    df = pd.read_csv(uploaded_file, encoding='latin1', dtype=str, keep_default_na=False)
                    logging.info(f"Successfully read CSV '{file_name}' with latin1.")
                except Exception as e_latin:
                     logging.error(f"Failed to read CSV '{file_name}' with latin1: {e_latin}")
                     st.error(f"Error reading CSV (tried UTF-8 and latin1): {e_latin}") # Show error to user
                     return None
            except Exception as e_csv:
                 logging.error(f"General error reading CSV '{file_name}': {e_csv}")
                 st.error(f"Error reading CSV: {e_csv}") # Show error to user
                 return None
        elif file_name.endswith(('.xls', '.xlsx')):
            try:
                 uploaded_file.seek(0) # Reset file pointer
                 df = pd.read_excel(uploaded_file, dtype=str, keep_default_na=False) # Read all as string
                 logging.info(f"Successfully read Excel file '{file_name}'.")
            except Exception as e_excel:
                 logging.error(f"Error reading Excel file '{file_name}': {e_excel}")
                 st.error(f"Error reading Excel file: {e_excel}") # Show error to user
                 return None
        else:
            # This should be caught by the uploader's type filter, but good practice
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            logging.warning(f"Unsupported file format attempted: {file_name}")
            return None

        # --- Post-load Cleaning ---
        if df is not None:
             original_shape = df.shape
             # Drop rows/columns that are *completely* empty (all NA/None/empty strings)
             df.replace("", np.nan, inplace=True) # Treat empty strings as NaN for dropping
             df.dropna(axis=0, how='all', inplace=True)
             df.dropna(axis=1, how='all', inplace=True)
             df.fillna('', inplace=True) # Convert remaining NaN back to empty strings

             cleaned_shape = df.shape
             logging.info(f"Cleaned DataFrame shape: {cleaned_shape} (Original: {original_shape})")

             if df.empty:
                 st.warning("The file appears empty or contains only empty rows/columns after cleaning.")
                 logging.warning(f"File '{file_name}' resulted in an empty DataFrame after cleaning.")
             # else:
                 # st.success("File loaded and cleaned.") # User feedback in main app
             return df
        else:
             # Errors should have been caught and returned None earlier
             st.error("Failed to create DataFrame from file for unknown reasons.")
             logging.error(f"DataFrame creation failed unexpectedly for {file_name}")
             return None

    except Exception as e:
        # Catch-all for unexpected issues during file processing
        st.error(f"An critical error occurred during file loading: {e}")
        logging.exception(f"Critical file loading error for {getattr(uploaded_file, 'name', 'N/A')}")
        return None

def validate_api_key(api_key_to_validate):
    """
    Validates the Gemini API key by trying to list available models.

    Args:
        api_key_to_validate (str): The API key string.

    Returns:
        tuple: (bool, str) indicating (is_valid, message)
    """
    if not api_key_to_validate:
        return False, "API key is empty."
    try:
        genai.configure(api_key=api_key_to_validate)
        # Listing models is a lightweight way to check authentication
        models = genai.list_models()
        # Optional: More specific check if needed
        if not any('models/gemini' in m.name for m in models):
            logging.warning(f"Could not find standard 'gemini' models for key validation (found: {[m.name for m in models]}). Key might be for a specific version or service.")
            # Decide if this is a failure or just a warning. Let's treat it as success for now.
            # return False, "Could not find standard Gemini models. Check key permissions/type."
        logging.info("API Key validated successfully via list_models().")
        return True, "API Key Validated"
    except Exception as e:
        logging.error(f"API Key validation failed: {e}")
        error_msg = str(e).lower()
        # Check for common error messages
        if "api key not valid" in error_msg or "invalid api key" in error_msg:
            return False, "Invalid API Key provided."
        elif "permission denied" in error_msg:
             return False, "Permission denied. Check API key permissions."
        elif "quota" in error_msg:
             return False, "Quota exceeded or billing issue. Check Google Cloud Console."
        else:
            # General connection or unexpected error
            return False, f"API Connection Error ({type(e).__name__}). Check network or key details."