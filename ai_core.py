# ai_core.py
"""
Core functions for interacting with the Google Generative AI API.
Includes theme generation, description generation, assignment, and Q&A.
"""

import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json
import logging
import time # Potentially for retries or delays if needed

# Import from local modules
from config import SAFETY_SETTINGS
from utils import clean_and_parse_json

# --- Model Initialization Helper ---
# This helps avoid repeated configuration if multiple calls use the same key/model type
# Note: Streamlit's execution model means this might still re-run often.
# Consider caching the model object itself if performance is critical and key doesn't change.
_MODEL_CACHE = {}

def get_generative_model(model_name="gemini-1.5-pro-latest", api_key=None):
    """Initializes and returns a GenerativeModel instance."""
    # Simple caching based on model name and key presence
    cache_key = (model_name, bool(api_key))
    if cache_key in _MODEL_CACHE:
        # Basic check if configuration seems okay (more robust checks might be needed)
        # If the key changes, the cache should ideally be invalidated.
        # Streamlit's rerun mechanism often handles this implicitly if api_key comes from state.
        # logging.debug(f"Using cached model for {model_name}")
        return _MODEL_CACHE[cache_key]

    if not api_key:
        st.error("API Key is missing. Cannot initialize AI model.")
        logging.error("Attempted to get model without API key.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        _MODEL_CACHE[cache_key] = model # Store in cache
        logging.info(f"Successfully initialized GenerativeModel: {model_name}")
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini Model ({model_name}): {e}")
        logging.error(f"Gemini Model initialization failed for {model_name}: {e}")
        return None


# --- Theme Generation ---
def generate_themes_with_llm(survey_question, responses_sample, api_key, generation_config):
    """
    Sends a prompt to the LLM to generate themes and sub-themes.

    Args:
        survey_question (str): The survey question text.
        responses_sample (list): A sample list of response strings.
        api_key (str): The Google API key.
        generation_config (GenerationConfig): Configuration for the generation.

    Returns:
        list or None: A list of dictionaries [{'theme': '...', 'sub_themes': ['...', ...]}, ...]
                      or None if a critical error occurs. Returns empty list if generation
                      succeeds but finds no themes.
    """
    logging.info(f"Attempting to generate themes from {len(responses_sample)} sample responses.")
    model = get_generative_model("gemini-1.5-pro-latest", api_key) # Use powerful model
    if not model:
        return None # Error already shown by get_generative_model

    theme_gen_prompt = f"""
Analyze the following survey responses provided for the question: "{survey_question}"

Identify the main recurring themes and specific sub-themes within each main theme based *strictly* on the provided responses.
- Aim for 5-10 distinct main themes unless the data strongly suggests otherwise.
- Each main theme should represent a significant, distinct topic mentioned in the responses.
- Sub-themes should capture specific facets, opinions, or examples within a main theme. Aim for 2-5 specific sub-themes per main theme.
- Provide concise, descriptive labels (1-4 words max) for each theme and sub-theme. Labels should be easily understandable and accurately reflect the content.
- A response should ideally belong to one main theme and one sub-theme.

Structure the output ONLY as a valid JSON list of objects. Each object must have:
1. A 'theme' key with the main theme label (string).
2. A 'sub_themes' key with a list of sub-theme labels (list of strings). Ensure sub-themes are relevant and specific to the main theme.

Example Output Format:
[
  {{
    "theme": "Ease of Use",
    "sub_themes": ["Intuitive Interface", "Clear Instructions", "Quick Setup"]
  }},
  {{
    "theme": "Customer Support",
    "sub_themes": ["Responsiveness", "Helpfulness", "Knowledge"]
  }},
  {{
    "theme": "Pricing & Value",
    "sub_themes": ["Cost", "Discounts", "Competitor Pricing", "Overall Value"]
  }}
]

Survey Responses Sample (Analyze these for themes):
{json.dumps(responses_sample, indent=2)}

---
Output ONLY the JSON list:
"""
    try:
        logging.info(f"Sending Theme Generation request...")
        result = model.generate_content(
            theme_gen_prompt,
            generation_config=generation_config,
            safety_settings=SAFETY_SETTINGS
        )

        # Safely access response text and handle potential blocking
        response_text = getattr(result, 'text', None)
        if response_text is None:
            block_reason = "Unknown"
            try:
                # Access prompt_feedback safely
                if hasattr(result, 'prompt_feedback') and result.prompt_feedback:
                    block_reason = result.prompt_feedback.block_reason.name
                elif hasattr(result, 'candidates') and result.candidates: # Check candidates if prompt_feedback missing
                    first_candidate = result.candidates[0]
                    if first_candidate.finish_reason.name != 'STOP':
                        block_reason = f"Finish Reason: {first_candidate.finish_reason.name}"
                        # Potentially check safety ratings here too if needed
            except Exception as feedback_err:
                 logging.warning(f"Could not determine block reason: {feedback_err}")

            st.error(f"Theme Generation failed: API returned no text content. Block Reason: {block_reason}")
            logging.error(f"Theme Gen API returned empty/blocked response. Reason: {block_reason}")
            return None

        # Use the robust parser from utils
        parsed_themes = clean_and_parse_json(response_text, expected_type=list)

        if parsed_themes is None: # Parsing failed or wrong type
            st.error("Theme Generation failed: Could not parse a valid JSON list from the AI response.")
            logging.error("Theme Gen parsing failed or returned wrong type.")
            # Optionally log the problematic response_text here for debugging
            # logging.debug(f"Problematic response text for theme gen: {response_text[:500]}")
            return None
        if not parsed_themes:
             st.warning("Theme Generation resulted in an empty list of themes. The AI might not have found significant patterns in the sample.")
             logging.info("Theme Gen successful but resulted in an empty list.")
             return [] # Return empty list, not None

        # --- Basic structure validation and cleaning ---
        validated_themes = []
        seen_themes = set()
        for item in parsed_themes:
            if isinstance(item, dict) and 'theme' in item and 'sub_themes' in item and isinstance(item['sub_themes'], list):
                theme_label = str(item['theme']).strip()
                # Skip empty or duplicate main themes
                if not theme_label or theme_label in seen_themes:
                    logging.warning(f"Skipping empty or duplicate theme label: '{theme_label}'")
                    continue
                seen_themes.add(theme_label)

                # Ensure sub-themes are strings and filter empty/duplicates within the theme
                valid_sub_themes = []
                seen_sub_themes = set()
                for sub in item['sub_themes']:
                    sub_label = str(sub).strip()
                    # Keep only non-empty, unique sub-theme labels for this theme
                    if sub_label and sub_label not in seen_sub_themes:
                        valid_sub_themes.append(sub_label)
                        seen_sub_themes.add(sub_label)

                # Decision: Keep theme even if it has no valid sub-themes after cleaning? Yes.
                item['theme'] = theme_label # Use stripped label
                item['sub_themes'] = valid_sub_themes # Use cleaned list
                validated_themes.append(item)
            else:
                logging.warning(f"Skipping invalid theme structure item during validation: {item}")

        if not validated_themes:
             st.error("Theme Generation failed: No valid theme structures found after validating the AI response.")
             logging.error("Theme Gen post-validation resulted in zero valid themes.")
             return None # Indicate failure if nothing valid remains

        logging.info(f"Successfully generated and validated {len(validated_themes)} themes.")
        return validated_themes

    except Exception as e:
        st.error(f"An unexpected error occurred during theme generation: {e}")
        logging.exception("Theme generation process failed unexpectedly.")
        return None


# --- Theme Description Generation ---
def generate_theme_descriptions_llm(theme_structure, survey_question, responses_sample, api_key, generation_config):
    """
    Generates descriptions for each theme based on its structure and sample responses.

    Args:
        theme_structure (list): The list of theme dictionaries generated previously.
        survey_question (str): The survey question text.
        responses_sample (list): Sample responses used for theme generation (for context).
        api_key (str): The Google API key.
        generation_config (GenerationConfig): Base configuration (temperature might be adjusted).

    Returns:
        dict or None: A dictionary mapping theme labels to their generated descriptions.
                      Returns None if a critical error occurs or no valid descriptions are generated.
    """
    logging.info("Attempting to generate theme descriptions...")
    if not theme_structure or not isinstance(theme_structure, list):
        logging.warning("Cannot generate descriptions for empty or invalid theme structure.")
        return None # Return None if no themes

    model = get_generative_model("gemini-1.5-flash-latest", api_key) # Flash model is sufficient
    if not model:
        return None

    # --- Prepare theme structure for prompt ---
    theme_info_prompt = ""
    original_theme_labels = set() # Keep track of themes sent for description
    for i, theme_data in enumerate(theme_structure):
        theme_label = theme_data.get('theme')
        if not theme_label or not isinstance(theme_label, str) or not theme_label.strip():
            logging.warning(f"Skipping description generation for theme at index {i} due to missing/invalid label.")
            continue # Skip if theme label is missing/empty/invalid
        original_theme_labels.add(theme_label)
        sub_themes = theme_data.get('sub_themes', [])
        # Ensure sub-themes are strings and format nicely
        sub_themes_str = [f'"{str(sub)}"' for sub in sub_themes if isinstance(sub, str) and sub.strip()]
        theme_info_prompt += f"{i+1}. Theme: \"{theme_label}\"\n   Sub-themes: [{', '.join(sub_themes_str) if sub_themes_str else 'N/A'}]\n"

    if not theme_info_prompt: # Handle case where all themes in the input were invalid
         logging.warning("No valid themes found in structure to generate descriptions for.")
         st.warning("Could not generate descriptions as no valid themes were provided.")
         return None

    # --- Construct Description Prompt ---
    desc_gen_prompt = f"""
You are an analyst summarizing survey themes.
Based on the following theme structure derived from responses to the question "{survey_question}", write a concise 1-2 sentence description for EACH main theme listed.
The description should explain the core topic or sentiment captured by the theme and its associated sub-themes, referencing the context of the original survey question and potentially the provided sample responses. Make descriptions distinct and informative.

Theme Structure:
{theme_info_prompt}

Context (Sample Responses Used for Theme Generation - Limited Sample):
{json.dumps(responses_sample[:30], indent=2)}

Output Format:
Return ONLY a valid JSON list where each object corresponds to one theme in the input structure (maintain order if possible, but matching by 'theme' label is key). Each object must have exactly two keys:
- "theme": The exact original theme label (string).
- "description": The generated 1-2 sentence description (string). Ensure the description is not empty.

Example Output:
[
  {{
    "theme": "Ease of Use",
    "description": "This theme captures feedback related to how simple and straightforward users found the product/service to interact with, including aspects like the interface and initial setup."
  }},
  {{
    "theme": "Customer Support",
    "description": "Concerns feedback on interactions with the support team, covering their speed in responding, the quality of help provided, and their overall product knowledge."
  }}
]
---
Output ONLY the JSON list below:
"""
    try:
        logging.info(f"Sending Theme Description request for {len(original_theme_labels)} themes.")
        # Adjust config for description generation if needed (e.g., slightly lower temp)
        desc_config = GenerationConfig(
             temperature=0.4, # Slightly more focused than default theme gen
             top_k=generation_config.top_k,
             top_p=generation_config.top_p,
             max_output_tokens=generation_config.max_output_tokens # Ensure enough tokens
        )
        result = model.generate_content(
            desc_gen_prompt, generation_config=desc_config, safety_settings=SAFETY_SETTINGS
        )

        # Handle response text and blocking
        response_text = getattr(result, 'text', None)
        if response_text is None:
            block_reason = "Unknown"
            try: # Safe access
                if hasattr(result, 'prompt_feedback') and result.prompt_feedback:
                    block_reason = result.prompt_feedback.block_reason.name
                elif hasattr(result, 'candidates') and result.candidates:
                     first_candidate = result.candidates[0]
                     if first_candidate.finish_reason.name != 'STOP': block_reason = f"Finish Reason: {first_candidate.finish_reason.name}"
            except Exception: pass
            st.error(f"Theme Description generation failed: API returned no text. Block Reason: {block_reason}")
            logging.error(f"Theme Desc API returned empty/blocked response. Reason: {block_reason}")
            return None

        # Use the robust parser
        parsed_descriptions = clean_and_parse_json(response_text, expected_type=list)

        if parsed_descriptions is None:
            st.error("Theme Description generation failed: Could not parse valid JSON list from response.")
            logging.error("Theme Desc parsing failed or returned wrong type.")
            # logging.debug(f"Problematic response text for theme desc: {response_text[:500]}")
            return None

        # --- Match descriptions back to original themes ---
        validated_descs_map = {}
        themes_described = set()
        processed_count = 0

        for item in parsed_descriptions:
            if isinstance(item, dict) and 'theme' in item and 'description' in item:
                theme_label = str(item['theme']).strip()
                description = str(item['description']).strip()

                # Only add if theme exists in original structure AND description is not empty
                if theme_label in original_theme_labels and description:
                    # Avoid duplicates if LLM repeats a theme
                    if theme_label not in themes_described:
                         validated_descs_map[theme_label] = description
                         processed_count += 1
                         themes_described.add(theme_label)
                    else:
                         logging.warning(f"Duplicate description received from LLM for theme: '{theme_label}'. Using first instance.")
                else:
                     # Log mismatches or empty descriptions from LLM
                     if theme_label not in original_theme_labels:
                         logging.warning(f"LLM returned description for an unknown/unexpected theme: '{theme_label}'. Skipping.")
                     elif not description:
                          logging.warning(f"LLM returned an empty description for theme: '{theme_label}'. Skipping.")
            else:
                logging.warning(f"Skipping invalid description structure item from LLM: {item}")

        if not validated_descs_map:
             st.warning("No valid theme descriptions were generated or matched to the provided themes.")
             logging.warning("Theme Desc post-processing yielded no valid descriptions.")
             return None # Return None if completely failed to get any descriptions

        # Log if some descriptions were missing
        missing_themes = original_theme_labels - themes_described
        if missing_themes:
            logging.warning(f"Could not generate descriptions for themes: {missing_themes}")
            st.warning(f"Descriptions could not be generated for some themes: {', '.join(missing_themes)}")

        logging.info(f"Successfully generated and validated descriptions for {processed_count}/{len(original_theme_labels)} themes.")
        return validated_descs_map # Return the map for easier merging

    except Exception as e:
        st.error(f"An unexpected error occurred during theme description generation: {e}")
        logging.exception("Theme description generation failed unexpectedly.")
        return None


# --- Theme Assignment (Batch) ---
def assign_themes_with_llm_batch(survey_question, responses_batch, theme_structure, api_key, generation_config):
    """
    Assigns themes/sub-themes AND confidence scores to a batch of responses using an LLM.

    Args:
        survey_question (str): The survey question text.
        responses_batch (list): A list of response strings for this batch.
        theme_structure (list): The finalized list of theme dictionaries (with labels, sub-themes).
        api_key (str): The Google API key.
        generation_config (GenerationConfig): Base configuration (temperature is overridden).

    Returns:
        list: A list of dictionaries, one for each response in the batch.
              Each dict has {'assigned_theme': ..., 'assigned_sub_theme': ..., 'assignment_confidence': ...}.
              Errors are marked within elements (e.g., 'Error: API Blocked').
              Returns an empty list on critical setup failure before API call.
              Returns list of error dicts if API call fails mid-batch.
    """
    logging.info(f"Attempting to assign themes and confidence to batch of {len(responses_batch)} responses.")
    if not responses_batch: return [] # Handle empty batch input immediately

    model = get_generative_model("gemini-1.5-flash-latest", api_key) # Flash is suitable for assignment
    if not model:
        # Error already shown by get_generative_model
        # Return list of errors matching batch size
        return [{'assigned_theme': 'Error: Model Init Failed', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(responses_batch)

    # --- Format Theme Structure for Prompt ---
    theme_structure_string = ""
    if not theme_structure or not isinstance(theme_structure, list):
        logging.error("Theme structure is empty or invalid for assignment.")
        st.error("Cannot perform assignment: Theme structure is missing or invalid.")
        return [{'assigned_theme': 'Error: No Themes Provided', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(responses_batch)

    # Extract valid theme labels and structure for prompt and validation
    theme_map = {} # Store theme -> sub-themes mapping for validation later
    valid_theme_labels = [] # Store valid theme labels found
    for i, theme_data in enumerate(theme_structure):
        theme_label = theme_data.get('theme')
        if not theme_label or not isinstance(theme_label, str) or not theme_label.strip():
            logging.warning(f"Skipping theme at index {i} in assignment prompt due to invalid label.")
            continue # Skip themes without valid labels

        theme_label = theme_label.strip() # Use stripped label
        valid_theme_labels.append(theme_label)

        sub_themes_list = theme_data.get('sub_themes', [])
        # Ensure sub-themes are strings and filter empty ones
        valid_sub_themes = [str(sub).strip() for sub in sub_themes_list if isinstance(sub, str) and str(sub).strip()]
        theme_map[theme_label] = valid_sub_themes # Store valid sub-themes for this theme

        if not valid_sub_themes:
            sub_themes_prompt_str = "[No specific sub-themes defined]"
        else:
             # Quote sub-themes for clarity in the prompt
             sub_themes_prompt_str = ", ".join([f'"{sub}"' for sub in valid_sub_themes])

        theme_structure_string += f"{i+1}. Theme: \"{theme_label}\"\n   - Available Sub-themes: [{sub_themes_prompt_str}]\n"

    # Check if any valid themes were added to the prompt string
    if not theme_structure_string or not valid_theme_labels:
        logging.error("Theme structure contained no valid themes with labels for assignment prompt.")
        st.error("Cannot perform assignment: No valid themes found in the provided structure.")
        return [{'assigned_theme': 'Error: No Valid Themes', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(responses_batch)

    # Explicitly add Uncategorized option at the end
    uncategorized_theme_label = "Uncategorized"
    uncategorized_index = len(valid_theme_labels) + 1 # Use count of *valid* themes
    theme_structure_string += f"{uncategorized_index}. Theme: \"{uncategorized_theme_label}\"\n   - Available Sub-themes: [\"N/A\"]\n"
    all_valid_theme_options = valid_theme_labels + [uncategorized_theme_label] # Full list for validation

    # --- Construct the Assignment Prompt (with confidence) ---
    assignment_prompt = f"""
You are an expert survey analyst performing thematic coding. Your task is to categorize EACH response in the provided batch based ONLY on the predefined theme structure given below. For each response, you must assign the single best main theme, the single best sub-theme within that main theme, and a confidence score for your assignment.

**Survey Question:**
"{survey_question}"

**Predefined Theme Structure (Use ONLY these themes and sub-themes):**
{theme_structure_string}

**Instructions for EACH response in the 'Input Responses Batch':**
1.  Read the response carefully in the context of the survey question.
2.  Select the **single BEST fitting** main 'Theme' label from the predefined structure (e.g., "{valid_theme_labels[0] if valid_theme_labels else 'Example Theme'}" or "{uncategorized_theme_label}").
3.  Select the **single BEST fitting** 'Sub-theme' from the list associated with your chosen main Theme.
4.  If the chosen main theme is "{uncategorized_theme_label}", the sub-theme MUST be "N/A".
5.  If a specific numbered theme (e.g., "{valid_theme_labels[0] if valid_theme_labels else 'Example Theme'}") fits the response well, but *none* of its listed sub-themes are a perfect match for the nuance, assign the main theme label and use the sub-theme label "General". If no sub-themes were defined at all for that theme in the structure, use "N/A" as the sub-theme.
6.  Determine your confidence in the chosen theme/sub-theme assignment for the response: "High", "Medium", or "Low". "High" confidence means the response clearly fits the theme/sub-theme. "Low" confidence means it's ambiguous, borderline, or potentially fits multiple themes weakly.
7.  Focus solely on matching the response content to the provided theme definitions. Adhere strictly to the provided theme/sub-theme labels. Do not invent new themes or sub-themes.

**Input Responses Batch (Process each one sequentially and provide a corresponding JSON object for each):**
{json.dumps(responses_batch, indent=2)}

**Output Format:**
Return ONLY a valid JSON list where EACH object corresponds EXACTLY to ONE response in the input batch (maintain the original order). Each object must contain exactly THREE keys:
- "assigned_theme": The exact string label of the chosen Theme (must be one of: {json.dumps(all_valid_theme_options)}).
- "assigned_sub_theme": The exact string label of the chosen Sub-theme (e.g., a specific sub-theme from the structure, "General", or "N/A").
- "assignment_confidence": Your confidence level as a string ("High", "Medium", or "Low").

**Example Output for a batch of 2 responses:**
[
  {{
    "assigned_theme": "{valid_theme_labels[0] if valid_theme_labels else 'Example Theme'}",
    "assigned_sub_theme": "Specific Sub-theme A1",
    "assignment_confidence": "High"
  }},
  {{
    "assigned_theme": "Theme Label B",
    "assigned_sub_theme": "General",
    "assignment_confidence": "Medium"
  }}
]

---
Output ONLY the JSON list below:
"""
    try:
        logging.info(f"Sending Theme Assignment request for batch of {len(responses_batch)}.")
        # Ensure low temperature for assignment consistency, allow sufficient tokens
        assign_config = GenerationConfig(
            temperature=0.15, # Force low temperature for more deterministic assignment
            top_k=1, # Consider setting top_k=1 to force picking the most likely token
            top_p=generation_config.top_p, # Top_p might still be relevant even with low temp/top_k
            max_output_tokens=max(2048, generation_config.max_output_tokens) # Ensure enough tokens for JSON output
        )

        result = model.generate_content(
            assignment_prompt,
            generation_config=assign_config,
            safety_settings=SAFETY_SETTINGS
        )
        response_text = getattr(result, 'text', None)

        # Handle blocking or empty response
        if response_text is None:
            block_reason = "Unknown"
            try: # Safe access
                 if hasattr(result, 'prompt_feedback') and result.prompt_feedback:
                     block_reason = result.prompt_feedback.block_reason.name
                 elif hasattr(result, 'candidates') and result.candidates:
                     first_candidate = result.candidates[0]
                     if first_candidate.finish_reason.name != 'STOP': block_reason = f"Finish Reason: {first_candidate.finish_reason.name}"
            except Exception: pass
            st.error(f"Theme Assignment failed for batch: API returned no text. Block Reason: {block_reason}")
            logging.error(f"Theme Assign API returned empty/blocked response. Reason: {block_reason}")
            return [{'assigned_theme': 'Error: API Blocked', 'assigned_sub_theme': block_reason, 'assignment_confidence': 'Low'}] * len(responses_batch)

        # Use the robust parser
        parsed_assignments = clean_and_parse_json(response_text, expected_type=list)

        if parsed_assignments is None:
            st.error("Theme Assignment failed for batch: Could not parse valid JSON list from AI response.")
            logging.error("Theme Assign parsing failed or returned wrong type.")
            # logging.debug(f"Problematic response text for theme assign: {response_text[:500]}")
            return [{'assigned_theme': 'Error: Parsing Failed', 'assigned_sub_theme': 'Invalid Structure', 'assignment_confidence': 'Low'}] * len(responses_batch)

        # --- Validate Length and Structure ---
        if len(parsed_assignments) != len(responses_batch):
            st.warning(f"Theme Assignment length mismatch: Expected {len(responses_batch)} results, AI returned {len(parsed_assignments)}. Padding/truncating results.")
            logging.warning(f"Theme Assign batch length mismatch: Expected {len(responses_batch)}, got {len(parsed_assignments)}")
            # Adjust length to match input batch size: Pad with errors or truncate
            if len(parsed_assignments) < len(responses_batch):
                # Pad with errors
                assignments_adjusted = parsed_assignments + [{'assigned_theme': 'Error: Missing Result', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * (len(responses_batch) - len(parsed_assignments))
            else:
                # Truncate
                assignments_adjusted = parsed_assignments[:len(responses_batch)]
            validated_assignments = assignments_adjusted
        else:
             validated_assignments = parsed_assignments # Length matches

        # --- Final validation of individual items ---
        final_results = []
        valid_confidences = ["High", "Medium", "Low"]

        for idx, item in enumerate(validated_assignments):
            if isinstance(item, dict) and all(k in item for k in ['assigned_theme', 'assigned_sub_theme', 'assignment_confidence']):
                # Safely get values, convert to string, and strip whitespace
                assigned_theme = str(item.get('assigned_theme', '')).strip()
                assigned_sub_theme = str(item.get('assigned_sub_theme', '')).strip()
                confidence = str(item.get('assignment_confidence', 'Low')).strip()

                # Validate confidence value
                if confidence not in valid_confidences:
                     logging.warning(f"Invalid confidence value '{confidence}' at index {idx}, defaulting to Low.")
                     confidence = "Low"

                # Handle empty theme/sub-theme assignment from LLM -> default to Uncategorized
                if not assigned_theme:
                     assigned_theme = uncategorized_theme_label
                     assigned_sub_theme = "N/A"
                     confidence = "Low" # Assume low confidence if theme was missing
                     logging.warning(f"Empty assigned_theme received from LLM at index {idx}, defaulting to '{uncategorized_theme_label}'.")
                elif not assigned_sub_theme:
                     # If theme is valid but sub-theme is missing, default based on theme
                     assigned_sub_theme = "N/A" if assigned_theme == uncategorized_theme_label else "General"
                     logging.warning(f"Empty assigned_sub_theme received for theme '{assigned_theme}' at index {idx}, defaulting to '{assigned_sub_theme}'.")


                # Validate if assigned_theme is one of the allowed options
                if assigned_theme not in all_valid_theme_options:
                    logging.warning(f"LLM assigned theme '{assigned_theme}' which is not in the defined structure ({all_valid_theme_options}) at index {idx}. Forcing to '{uncategorized_theme_label}'.")
                    # Force to 'Uncategorized' if LLM hallucinates a theme label
                    assigned_theme = uncategorized_theme_label
                    assigned_sub_theme = "N/A"
                    confidence = "Low" # If theme was wrong, confidence is low

                # If theme is Uncategorized, sub-theme MUST be N/A
                if assigned_theme == uncategorized_theme_label and assigned_sub_theme != "N/A":
                    logging.warning(f"Correcting sub-theme for '{uncategorized_theme_label}' at index {idx}: was '{assigned_sub_theme}', changed to 'N/A'.")
                    assigned_sub_theme = "N/A"

                # Optional: Validate if assigned_sub_theme is valid for the assigned_theme
                # This adds complexity but improves accuracy.
                # if assigned_theme != uncategorized_theme_label and assigned_sub_theme not in ["General", "N/A"]:
                #     allowed_subs = theme_map.get(assigned_theme, [])
                #     if assigned_sub_theme not in allowed_subs:
                #         logging.warning(f"LLM assigned sub-theme '{assigned_sub_theme}' which is not defined for theme '{assigned_theme}' at index {idx}. Allowing for now, but may indicate LLM deviation.")
                        # Option: Force to 'General' or 'N/A'? Or allow user to fix? Let's allow for now.

                final_results.append({
                    'assigned_theme': assigned_theme,
                    'assigned_sub_theme': assigned_sub_theme,
                    'assignment_confidence': confidence
                })
            else:
                # Handle cases where the item is not a dict or keys are missing
                logging.warning(f"Invalid assignment item structure received from LLM at index {idx}: {item}. Replacing with error placeholder.")
                final_results.append({'assigned_theme': 'Error: Invalid Format', 'assigned_sub_theme': str(item)[:50], 'assignment_confidence': 'Low'})

        return final_results

    except Exception as e:
        st.error(f"An unexpected error occurred during theme assignment batch processing: {e}")
        logging.exception("Theme assignment batch failed unexpectedly.")
        # Return list of errors matching batch size
        return [{'assigned_theme': 'Error: Exception', 'assigned_sub_theme': str(e)[:50], 'assignment_confidence': 'Low'}] * len(responses_batch)


# --- AI Q&A Function ---
def ask_ai_about_data(survey_question, responses_list, user_question, generation_config, api_key):
    """
    Asks the AI a question about the provided survey data (raw responses).

    Args:
        survey_question (str): The survey question text.
        responses_list (list): The list of raw response strings.
        user_question (str): The question asked by the user.
        generation_config (GenerationConfig): Configuration for the generation.
        api_key (str): The Google API key.

    Returns:
        str: The AI's answer, or an error message string.
    """
    logging.info(f"Processing AI Q&A request. Question: '{user_question[:50]}...'")
    if not responses_list:
        logging.warning("AI Q&A called with no responses provided.")
        return "Error: No responses provided to ask questions about."
    if not survey_question:
        logging.warning("AI Q&A called without survey question context.")
        # return "Error: Survey question context is missing." # Or proceed without it

    model = get_generative_model("gemini-1.5-flash-latest", api_key) # Flash is usually fine for Q&A
    if not model:
        return "Error: Could not initialize AI model for Q&A." # Error msg already shown

    # --- Format the responses for the prompt context ---
    # Join responses, limiting total length to avoid exceeding model context limits
    responses_context = "\n".join([f"- {r}" for r in responses_list if isinstance(r, str) and r.strip()])
    max_context_chars = 15000 # Limit context length (adjust based on model limits and typical response sizes)
    if len(responses_context) > max_context_chars:
        responses_context = responses_context[:max_context_chars] + "\n... (responses truncated)"
        logging.warning(f"Q&A context truncated to {max_context_chars} characters.")
        st.caption(f"Note: Input responses context was truncated to {max_context_chars} characters for the AI.")

    if not responses_context.strip():
         logging.warning("AI Q&A context became empty after processing responses.")
         return "Error: No valid response content found to provide context for the question."

    # --- Construct Q&A Prompt ---
    qa_prompt = f"""
Context:
You are an AI assistant analyzing feedback for a survey.

Survey Question:
"{survey_question}"

Provided Responses (Raw Data - potentially truncated):
{responses_context}

---
Task:
Based *only* on the provided Survey Question and Responses context above, answer the following user question accurately and concisely.
- Do not invent information or make assumptions beyond the provided text.
- If the question asks for specific examples (like quotes or verbatims), retrieve them directly from the 'Provided Responses' section.
- Use markdown formatting (like lists or bolding) for clarity where appropriate.
- Avoid introductory phrases like "Based on the provided responses..." unless specifically asked to refer to the data source.

User Question:
"{user_question}"

Answer:
"""
    try:
        logging.info(f"Sending Q&A query to Gemini...")
        # Use a slightly higher temperature for Q&A if more nuanced answers are desired
        qa_config = GenerationConfig(
            temperature=0.6, # Allow slightly more variability for Q&A
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            max_output_tokens=max(1024, generation_config.max_output_tokens) # Ensure reasonable output length for answers
        )
        result = model.generate_content(
            qa_prompt,
            generation_config=qa_config,
            safety_settings=SAFETY_SETTINGS
            )
        response_text = getattr(result, 'text', None)

        if response_text is None:
            block_reason = "Unknown"
            try: # Safe access
                 if hasattr(result, 'prompt_feedback') and result.prompt_feedback:
                     block_reason = result.prompt_feedback.block_reason.name
                 elif hasattr(result, 'candidates') and result.candidates:
                     first_candidate = result.candidates[0]
                     if first_candidate.finish_reason.name != 'STOP': block_reason = f"Finish Reason: {first_candidate.finish_reason.name}"
            except Exception: pass
            st.error(f"AI Q&A failed: Response was blocked or empty. Reason: {block_reason}")
            logging.error(f"AI Q&A blocked/empty. Reason: {block_reason}")
            return f"Error: AI response was blocked or empty (Reason: {block_reason})."

        logging.info("AI Q&A response received successfully.")
        return response_text.strip() # Return the stripped text content

    except Exception as e:
        st.error(f"An error occurred during AI Q&A generation: {e}")
        logging.exception("Exception occurred during AI Q&A execution.")
        return f"Error: An exception occurred while generating the answer: {e}"