# ai_core.py (for AI Themer App)
"""
Core functions for interacting with the Google Generative AI API for the AI Themer.
Includes theme generation, description generation, assignment, and the CORRECTED Q&A function.
"""

import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold # Added HarmCategory/BlockThreshold
import json
import logging
import time

# Import from local Themer modules
from config import SAFETY_SETTINGS # Use Themer's safety settings
from utils import clean_and_parse_json # Use Themer's JSON parser

# --- Model Initialization Helper ---
# (Keep the existing helper from Themer)
_MODEL_CACHE = {}
def get_generative_model(model_name="gemini-1.5-pro-latest", api_key=None):
    """Initializes and returns a GenerativeModel instance for the Themer."""
    cache_key = (model_name, bool(api_key))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not api_key:
        st.error("API Key is missing. Cannot initialize AI model.")
        logging.error("Themer: Attempted to get model without API key.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        _MODEL_CACHE[cache_key] = model
        logging.info(f"Themer: Initialized GenerativeModel: {model_name}")
        return model
    except Exception as e:
        st.error(f"Themer: Failed to initialize Gemini Model ({model_name}): {e}")
        logging.error(f"Themer: Gemini Model init failed for {model_name}: {e}")
        return None

# --- Theme Generation Function ---
# (Keep the existing function from Themer's ai_core.py)
def generate_themes_with_llm(survey_question, responses_sample, api_key, generation_config):
    # ... (Full function code as previously defined) ...
    logging.info(f"Attempting to generate themes from {len(responses_sample)} sample responses.")
    model = get_generative_model("gemini-1.5-pro-latest", api_key)
    if not model: return None

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
[ {{"theme": "Ease of Use", "sub_themes": ["Intuitive Interface", "Clear Instructions", "Quick Setup"]}}, {{"theme": "Customer Support", "sub_themes": ["Responsiveness", "Helpfulness", "Knowledge"]}}, {{"theme": "Pricing & Value", "sub_themes": ["Cost", "Discounts", "Competitor Pricing", "Overall Value"]}} ]
Survey Responses Sample (Analyze these for themes):
{json.dumps(responses_sample, indent=2)}
---
Output ONLY the JSON list:
"""
    try:
        logging.info(f"Sending Theme Generation request...")
        result = model.generate_content(theme_gen_prompt, generation_config=generation_config, safety_settings=SAFETY_SETTINGS)
        response_text = getattr(result, 'text', None)
        if response_text is None:
            block_reason = "Unknown"
            try: # Safe access
                if hasattr(result, 'prompt_feedback') and result.prompt_feedback: block_reason = result.prompt_feedback.block_reason.name
                elif hasattr(result, 'candidates') and result.candidates and result.candidates[0].finish_reason.name != 'STOP': block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
            except Exception as feedback_err: logging.warning(f"Could not determine block reason: {feedback_err}")
            st.error(f"Theme Generation failed: API returned no text content. Block Reason: {block_reason}")
            logging.error(f"Theme Gen API returned empty/blocked response. Reason: {block_reason}")
            return None

        parsed_themes = clean_and_parse_json(response_text, expected_type=list) # Use Themer's parser

        if parsed_themes is None:
            st.error("Theme Generation failed: Could not parse a valid JSON list from the AI response.")
            logging.error("Theme Gen parsing failed or returned wrong type.")
            return None
        if not parsed_themes:
             st.warning("Theme Generation resulted in an empty list of themes.")
             logging.info("Theme Gen successful but resulted in an empty list.")
             return []

        validated_themes = []; seen_themes = set()
        for item in parsed_themes:
            if isinstance(item, dict) and 'theme' in item and 'sub_themes' in item and isinstance(item['sub_themes'], list):
                theme_label = str(item['theme']).strip()
                if not theme_label or theme_label in seen_themes: logging.warning(f"Skipping empty/duplicate theme label: '{theme_label}'"); continue
                seen_themes.add(theme_label)
                valid_sub_themes = []; seen_sub_themes = set()
                for sub in item['sub_themes']:
                    sub_label = str(sub).strip()
                    if sub_label and sub_label not in seen_sub_themes: valid_sub_themes.append(sub_label); seen_sub_themes.add(sub_label)
                item['theme'] = theme_label
                item['sub_themes'] = valid_sub_themes
                validated_themes.append(item)
            else: logging.warning(f"Skipping invalid theme structure item during validation: {item}")

        if not validated_themes:
             st.error("Theme Generation failed: No valid theme structures found after validation.")
             logging.error("Theme Gen post-validation resulted in zero valid themes.")
             return None

        logging.info(f"Successfully generated and validated {len(validated_themes)} themes.")
        return validated_themes
    except Exception as e:
        st.error(f"An unexpected error occurred during theme generation: {e}")
        logging.exception("Theme generation process failed unexpectedly.")
        return None


# --- Theme Description Generation Function ---
# (Keep the existing function from Themer's ai_core.py)
def generate_theme_descriptions_llm(theme_structure, survey_question, responses_sample, api_key, generation_config):
    # ... (Full function code as previously defined) ...
    logging.info("Attempting to generate theme descriptions...")
    if not theme_structure or not isinstance(theme_structure, list):
        logging.warning("Cannot generate descriptions for empty or invalid theme structure."); return None

    model = get_generative_model("gemini-1.5-flash-latest", api_key)
    if not model: return None

    theme_info_prompt = ""; original_theme_labels = set()
    for i, theme_data in enumerate(theme_structure):
        theme_label = theme_data.get('theme')
        if not theme_label or not isinstance(theme_label, str) or not theme_label.strip():
            logging.warning(f"Skipping desc gen for theme at index {i} due to invalid label."); continue
        original_theme_labels.add(theme_label)
        sub_themes = theme_data.get('sub_themes', [])
        sub_themes_str = [f'"{str(sub)}"' for sub in sub_themes if isinstance(sub, str) and sub.strip()]
        theme_info_prompt += f"{i+1}. Theme: \"{theme_label}\"\n   Sub-themes: [{', '.join(sub_themes_str) if sub_themes_str else 'N/A'}]\n"

    if not theme_info_prompt:
         logging.warning("No valid themes found in structure to generate descriptions for."); st.warning("No valid themes to generate descriptions for."); return None

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
[ {{"theme": "Ease of Use", "description": "Feedback on simplicity and interaction."}}, {{"theme": "Customer Support", "description": "Feedback on support team interactions."}} ]
---
Output ONLY the JSON list below:
"""
    try:
        logging.info(f"Sending Theme Description request for {len(original_theme_labels)} themes.")
        desc_config = GenerationConfig(temperature=0.4, top_k=generation_config.top_k, top_p=generation_config.top_p, max_output_tokens=generation_config.max_output_tokens)
        result = model.generate_content(desc_gen_prompt, generation_config=desc_config, safety_settings=SAFETY_SETTINGS)
        response_text = getattr(result, 'text', None)
        if response_text is None:
            block_reason = "Unknown"
            try: # Safe access
                if hasattr(result, 'prompt_feedback') and result.prompt_feedback: block_reason = result.prompt_feedback.block_reason.name
                elif hasattr(result, 'candidates') and result.candidates and result.candidates[0].finish_reason.name != 'STOP': block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
            except Exception: pass
            st.error(f"Theme Description generation failed: API returned no text. Block Reason: {block_reason}")
            logging.error(f"Theme Desc API returned empty/blocked response. Reason: {block_reason}")
            return None

        parsed_descriptions = clean_and_parse_json(response_text, expected_type=list) # Use Themer's parser

        if parsed_descriptions is None:
            st.error("Theme Description generation failed: Could not parse valid JSON list from response."); logging.error("Theme Desc parsing failed."); return None

        validated_descs_map = {}; themes_described = set(); processed_count = 0
        for item in parsed_descriptions:
            if isinstance(item, dict) and 'theme' in item and 'description' in item:
                theme_label = str(item['theme']).strip(); description = str(item['description']).strip()
                if theme_label in original_theme_labels and description:
                    if theme_label not in themes_described: validated_descs_map[theme_label] = description; processed_count += 1; themes_described.add(theme_label)
                    else: logging.warning(f"Duplicate desc received for theme: '{theme_label}'.")
                else: logging.warning(f"Skipping desc for invalid/unknown/empty theme or empty desc: {item}")
            else: logging.warning(f"Skipping invalid desc structure item: {item}")

        if not validated_descs_map: st.warning("No valid theme descriptions generated/matched."); logging.warning("Theme Desc post-processing yielded no valid descriptions."); return None

        missing_themes = original_theme_labels - themes_described
        if missing_themes: logging.warning(f"Could not generate descriptions for themes: {missing_themes}"); st.warning(f"Descriptions missing for: {', '.join(missing_themes)}")

        logging.info(f"Generated/validated descriptions for {processed_count}/{len(original_theme_labels)} themes.")
        return validated_descs_map
    except Exception as e:
        st.error(f"An error occurred during theme description generation: {e}")
        logging.exception("Theme description generation failed unexpectedly.")
        return None


# --- Theme Assignment (Batch) Function ---
# (Keep the existing function from Themer's ai_core.py)
def assign_themes_with_llm_batch(survey_question, responses_batch, theme_structure, api_key, generation_config):
    # ... (Full function code as previously defined) ...
    logging.info(f"Attempting to assign themes/confidence to batch of {len(responses_batch)} responses.")
    if not responses_batch: return []

    model = get_generative_model("gemini-1.5-flash-latest", api_key)
    if not model: return [{'assigned_theme': 'Error: Model Init Failed', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(responses_batch)

    theme_structure_string = ""; theme_map = {}; valid_theme_labels = []
    if not theme_structure or not isinstance(theme_structure, list):
        logging.error("Theme structure empty/invalid for assignment."); st.error("Cannot assign: Theme structure missing/invalid."); return [{'assigned_theme': 'Error: No Themes Provided', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(responses_batch)

    for i, theme_data in enumerate(theme_structure):
        theme_label = theme_data.get('theme')
        if not theme_label or not isinstance(theme_label, str) or not theme_label.strip(): logging.warning(f"Skipping theme index {i} in assignment prompt due to invalid label."); continue
        theme_label = theme_label.strip(); valid_theme_labels.append(theme_label)
        sub_themes_list = theme_data.get('sub_themes', [])
        valid_sub_themes = [str(sub).strip() for sub in sub_themes_list if isinstance(sub, str) and str(sub).strip()]
        theme_map[theme_label] = valid_sub_themes
        sub_themes_prompt_str = ", ".join([f'"{sub}"' for sub in valid_sub_themes]) if valid_sub_themes else "[No specific sub-themes defined]"
        theme_structure_string += f"{len(valid_theme_labels)}. Theme: \"{theme_label}\"\n   - Available Sub-themes: [{sub_themes_prompt_str}]\n" # Use running count for index

    if not theme_structure_string or not valid_theme_labels:
        logging.error("Theme structure contained no valid themes."); st.error("Cannot assign: No valid themes found."); return [{'assigned_theme': 'Error: No Valid Themes', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(responses_batch)

    uncategorized_theme_label = "Uncategorized"; uncategorized_index = len(valid_theme_labels) + 1
    theme_structure_string += f"{uncategorized_index}. Theme: \"{uncategorized_theme_label}\"\n   - Available Sub-themes: [\"N/A\"]\n"
    all_valid_theme_options = valid_theme_labels + [uncategorized_theme_label]

    assignment_prompt = f"""
You are an expert survey analyst performing thematic coding. Categorize EACH response in the provided batch based ONLY on the predefined theme structure below. Assign the single best main theme, the single best sub-theme, and a confidence score.
**Survey Question:** "{survey_question}"
**Predefined Theme Structure (Use ONLY these):**
{theme_structure_string}
**Instructions for EACH response in 'Input Responses Batch':**
1. Read response in context of question.
2. Select BEST fitting 'Theme' label (e.g., "{valid_theme_labels[0] if valid_theme_labels else 'Example'}" or "{uncategorized_theme_label}").
3. Select BEST fitting 'Sub-theme' from chosen Theme's list.
4. If Theme is "{uncategorized_theme_label}", Sub-theme MUST be "N/A".
5. If a numbered theme fits but no listed sub-theme matches well, use "General". If no sub-themes were defined, use "N/A".
6. Determine confidence: "High", "Medium", or "Low".
7. Adhere strictly to provided labels.
**Input Responses Batch (Process each):**
{json.dumps(responses_batch, indent=2)}
**Output Format:** ONLY a valid JSON list. EACH object must correspond EXACTLY to ONE input response (maintain order) with THREE keys: "assigned_theme" (string, must be one of {json.dumps(all_valid_theme_options)}), "assigned_sub_theme" (string), "assignment_confidence" (string: "High"/"Medium"/"Low").
**Example Output (batch of 2):**
[ {{"assigned_theme": "{valid_theme_labels[0] if valid_theme_labels else 'Example'}", "assigned_sub_theme": "Specific Sub A1", "assignment_confidence": "High"}}, {{"assigned_theme": "Theme B", "assigned_sub_theme": "General", "assignment_confidence": "Medium"}} ]
---
Output ONLY the JSON list below:
"""
    try:
        logging.info(f"Sending Theme Assignment request for batch of {len(responses_batch)}.")
        assign_config = GenerationConfig(temperature=0.15, top_k=1, top_p=generation_config.top_p, max_output_tokens=max(2048, generation_config.max_output_tokens))
        result = model.generate_content(assignment_prompt, generation_config=assign_config, safety_settings=SAFETY_SETTINGS)
        response_text = getattr(result, 'text', None)

        if response_text is None:
            block_reason = "Unknown"
            try: # Safe access
                 if hasattr(result, 'prompt_feedback') and result.prompt_feedback: block_reason = result.prompt_feedback.block_reason.name
                 elif hasattr(result, 'candidates') and result.candidates and result.candidates[0].finish_reason.name != 'STOP': block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
            except Exception: pass
            st.error(f"Assignment failed: API blocked/empty. Reason: {block_reason}"); logging.error(f"Theme Assign API blocked/empty. Reason: {block_reason}")
            return [{'assigned_theme': 'Error: API Blocked', 'assigned_sub_theme': block_reason, 'assignment_confidence': 'Low'}] * len(responses_batch)

        parsed_assignments = clean_and_parse_json(response_text, expected_type=list) # Use Themer's parser

        if parsed_assignments is None:
            st.error("Assignment failed: Could not parse valid JSON list."); logging.error("Theme Assign parsing failed.")
            return [{'assigned_theme': 'Error: Parsing Failed', 'assigned_sub_theme': 'Invalid Structure', 'assignment_confidence': 'Low'}] * len(responses_batch)

        validated_assignments = parsed_assignments # Start with parsed
        if len(parsed_assignments) != len(responses_batch):
            st.warning(f"Assignment length mismatch: Expected {len(responses_batch)}, got {len(parsed_assignments)}. Padding/truncating.")
            logging.warning(f"Theme Assign length mismatch: Expected {len(responses_batch)}, got {len(parsed_assignments)}")
            if len(parsed_assignments) < len(responses_batch): validated_assignments += [{'assigned_theme': 'Error: Missing Result', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * (len(responses_batch) - len(parsed_assignments))
            else: validated_assignments = parsed_assignments[:len(responses_batch)]

        final_results = []; valid_confidences = ["High", "Medium", "Low"]
        for idx, item in enumerate(validated_assignments):
            if isinstance(item, dict) and all(k in item for k in ['assigned_theme', 'assigned_sub_theme', 'assignment_confidence']):
                assigned_theme = str(item.get('assigned_theme', '')).strip(); assigned_sub_theme = str(item.get('assigned_sub_theme', '')).strip(); confidence = str(item.get('assignment_confidence', 'Low')).strip()
                if confidence not in valid_confidences: logging.warning(f"Invalid confidence '{confidence}' at idx {idx}, defaulting Low."); confidence = "Low"
                if not assigned_theme: assigned_theme = uncategorized_theme_label; assigned_sub_theme = "N/A"; confidence = "Low"; logging.warning(f"Empty theme at idx {idx}, defaulting Uncategorized.")
                elif not assigned_sub_theme: assigned_sub_theme = "N/A" if assigned_theme == uncategorized_theme_label else "General"; logging.warning(f"Empty sub-theme for '{assigned_theme}' at idx {idx}, defaulting '{assigned_sub_theme}'.")
                if assigned_theme not in all_valid_theme_options: logging.warning(f"Assigned theme '{assigned_theme}' not in structure at idx {idx}. Forcing Uncategorized."); assigned_theme = uncategorized_theme_label; assigned_sub_theme = "N/A"; confidence = "Low"
                if assigned_theme == uncategorized_theme_label and assigned_sub_theme != "N/A": logging.warning(f"Correcting sub-theme for Uncategorized at idx {idx} to N/A."); assigned_sub_theme = "N/A"
                final_results.append({'assigned_theme': assigned_theme, 'assigned_sub_theme': assigned_sub_theme, 'assignment_confidence': confidence})
            else: logging.warning(f"Invalid assignment structure at idx {idx}: {item}. Replacing."); final_results.append({'assigned_theme': 'Error: Invalid Format', 'assigned_sub_theme': str(item)[:50], 'assignment_confidence': 'Low'})
        return final_results
    except Exception as e:
        st.error(f"Unexpected error during theme assignment batch: {e}")
        logging.exception("Theme assignment batch failed unexpectedly.")
        return [{'assigned_theme': 'Error: Exception', 'assigned_sub_theme': str(e)[:50], 'assignment_confidence': 'Low'}] * len(responses_batch)


# --- AI Q&A Function (REPLACED with Analyzer Version) --- ## <<<< THIS IS THE UPDATED FUNCTION
def ask_ai_about_data(survey_question, responses_list, user_question, generation_config, api_key):
    """
    Ask the AI a question about the provided survey data (raw responses).
    Uses the logic from the AI Response Analyzer app.
    """
    logging.info(f"Themer Q&A: Processing request. Q: '{user_question[:50]}...'")
    if not responses_list:
        logging.warning("Themer Q&A: No responses provided.")
        return "Error: No responses provided to ask questions about."
    if not survey_question:
        logging.warning("Themer Q&A: Survey question context is missing.")
        # Proceed but the AI might lack context

    # Use Themer's model helper
    model = get_generative_model("gemini-1.5-flash-latest", api_key) # Use Flash for Q&A
    if not model:
        # Error already displayed by get_generative_model
        return "Error: Could not initialize AI model for Q&A."

    # Format context, limit length
    responses_context = "\n".join([f"- {r}" for r in responses_list if r and isinstance(r, str)])
    max_context_chars = 15000 # Set a reasonable limit
    if len(responses_context) > max_context_chars:
        responses_context = responses_context[:max_context_chars] + "\n... (responses truncated)"
        logging.warning("Themer Q&A: Context truncated.")
        st.caption(f"Note: Input responses context truncated to {max_context_chars} chars for AI.")

    if not responses_context.strip():
        logging.warning("Themer Q&A: Context empty after processing.")
        return "Error: No valid response content for Q&A context."

    # Use the prompt structure from the Analyzer code
    qa_prompt = f"""
Context:
You are analyzing feedback for the following survey question.

Survey Question:
"{survey_question}"

Provided Responses (potentially truncated):
{responses_context}

---
Task:
Based *only* on the provided Survey Question and Responses context above, answer the following user question as accurately and concisely as possible. Do not invent information not present in the responses. If the question asks for specific examples (like verbatims), provide them directly from the 'Provided Responses'. Use markdown formatting for lists or emphasis where appropriate.

User Question:
"{user_question}"

Answer:
"""

    try:
        logging.info(f"Themer Q&A: Sending query to Gemini: {user_question[:50]}...")
        # Define QA specific config, potentially based on passed config
        qa_config = GenerationConfig(
            temperature=0.5, # Allow some flexibility for summarization/explanation
            top_k=generation_config.top_k, # Use base config value from sidebar
            top_p=generation_config.top_p, # Use base config value from sidebar
            max_output_tokens=max(1024, generation_config.max_output_tokens) # Ensure enough length
        )
        result = model.generate_content(
            qa_prompt,
            generation_config=qa_config, # Use the specific QA config
            safety_settings=SAFETY_SETTINGS # Use Themer's safety settings from config
        )

        # Robust Response Handling (copied from Analyzer's implementation)
        block_reason = None
        response_text = None
        try: # Safe access to feedback/candidates
            if hasattr(result, 'prompt_feedback') and result.prompt_feedback and result.prompt_feedback.block_reason:
                 block_reason = result.prompt_feedback.block_reason.name
            elif result.candidates and result.candidates[0].finish_reason.name != 'STOP':
                  block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
            # Try getting text only if not clearly blocked
            if not block_reason or block_reason == 'Finish Reason: MAX_TOKENS':
                response_text = result.text
        except ValueError as e: # Often indicates blocked content
            logging.warning(f"ValueError accessing Q&A result.text: {e}")
            if not block_reason: block_reason = "CONTENT_BLOCKED"
        except Exception as e: # Catch other potential errors
            logging.error(f"Unexpected error accessing Q&A result parts: {e}")
            if not block_reason: block_reason = f"ACCESS_ERROR: {e}"

        if block_reason and block_reason != 'Finish Reason: MAX_TOKENS':
             st.error(f"AI Q&A response blocked. Reason: {block_reason}")
             logging.error(f"Themer Q&A blocked: {block_reason}")
             # Provide a more user-friendly error message
             return f"Error: The AI's response was blocked (Reason: {block_reason}). Please try rephrasing your question or check the content safety settings."

        if not response_text:
             st.error("AI Q&A returned no response text.")
             logging.error("Themer Q&A returned empty response text.")
             return "Error: AI returned an empty response."

        logging.info("Themer Q&A: Response received successfully.")
        return response_text.strip() # Return the AI's answer

    except Exception as e:
        st.error(f"An unexpected error occurred during AI Q&A: {e}")
        logging.exception("Themer: Exception occurred during AI Q&A")
        return f"Error: An exception occurred while generating the answer: {e}"
