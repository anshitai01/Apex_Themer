# main_app.py
"""
Main Streamlit application file for the PPL APEX AI Themer.
Orchestrates the UI, state management, and interactions between modules.
Run this file with `streamlit run main_app.py`.
ADAPTED FOR PHRONESIS APEX THEME.
"""

# --- Core Imports ---
import streamlit as st
import pandas as pd
import logging
import time
import random
from google.generativeai.types import GenerationConfig
import base64 # Needed for logo
from pathlib import Path # Needed for logo path

# --- Application Modules Imports ---
import config as cfg
from styling import apply_styling
import utils
import ai_core
import ui_components

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI Themer - Phronesis Apex", # Updated Title
    page_icon="🏷️" # Keep themer icon or use Phronesis icon "📊" ?
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("AI Themer Application Started (Apex Theme).")

# --- Apply Custom Styling ---
apply_styling() # Applies the Phronesis Apex theme CSS

# --- Session State Initialization (Uses new defaults from config if needed) ---
# ... (Keep the existing default_state_values loop, it will use new config) ...
default_state_values = {
    cfg.INITIALIZED_STATE_KEY: False, cfg.API_KEY_STATE_KEY: None,
    cfg.API_KEY_SOURCE_STATE_KEY: None, cfg.INPUT_METHOD_KEY: 'Paste Text',
    cfg.SURVEY_QUESTION_KEY: '', cfg.RESPONSES_RAW_KEY: [],
    cfg.UPLOADED_DF_KEY: None, cfg.SELECTED_COLUMN_KEY: None,
    cfg.CURRENT_FILE_NAME_KEY: None, cfg.RESPONSES_INPUT_AREA_VAL_KEY: '',
    cfg.SELECTED_COLUMN_IDX_KEY: 0, cfg.GENERATED_THEMES_KEY: None,
    cfg.EDITED_THEMES_KEY: [], cfg.ASSIGNMENT_DF_KEY: None,
    cfg.AI_QA_HISTORY_KEY: [], cfg.BATCH_SIZE_KEY: 15, cfg.SAMPLE_SIZE_KEY: 100,
    cfg.GEN_TEMP_KEY: 0.5, cfg.GEN_TOP_K_KEY: 40, cfg.GEN_TOP_P_KEY: 0.95,
    cfg.GEN_MAX_TOKENS_KEY: 4096,
}
for key, default_value in default_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
        logging.debug(f"Initialized session state key '{key}' with default.")


# --- Helper function to clear app-specific state (Unchanged) ---
def clear_app_state(clear_api_keys=False):
    # ... (Keep the existing clear_app_state function) ...
    keys_to_clear = [
        cfg.SURVEY_QUESTION_KEY, cfg.RESPONSES_RAW_KEY, cfg.UPLOADED_DF_KEY,
        cfg.SELECTED_COLUMN_KEY, cfg.INPUT_METHOD_KEY, cfg.CURRENT_FILE_NAME_KEY,
        cfg.GENERATED_THEMES_KEY, cfg.EDITED_THEMES_KEY, cfg.ASSIGNMENT_DF_KEY,
        cfg.AI_QA_HISTORY_KEY, cfg.RESPONSES_INPUT_AREA_VAL_KEY, cfg.SELECTED_COLUMN_IDX_KEY,
        cfg.UPLOADED_THEME_FILE_WIDGET_KEY, cfg.ASSIGNMENT_EDITOR_WIDGET_KEY,
    ]
    if clear_api_keys:
        keys_to_clear.extend([
            cfg.INITIALIZED_STATE_KEY, cfg.API_KEY_STATE_KEY, cfg.API_KEY_SOURCE_STATE_KEY
        ])
    cleared_count = 0
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]; cleared_count += 1
            logging.debug(f"Cleared session state key: {key}")
    if clear_api_keys:
        st.session_state[cfg.INITIALIZED_STATE_KEY] = False
        st.session_state[cfg.API_KEY_STATE_KEY] = None
        st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = None
        logging.info(f"Cleared {cleared_count} keys and reset API/init state.")
    else:
         st.session_state[cfg.RESPONSES_RAW_KEY] = []
         st.session_state[cfg.EDITED_THEMES_KEY] = []
         st.session_state[cfg.AI_QA_HISTORY_KEY] = []
         logging.info(f"Cleared {cleared_count} data/state keys.")


# --- Logo Loading Function (from landing page) ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(str(bin_file), 'rb') as f: # Ensure path is string
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.sidebar.warning(f"Logo file '{cfg.LOGO_FILENAME}' not found.") # Show warning in sidebar
        logging.warning(f"Logo file not found at {bin_file}")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading logo: {e}")
        logging.error(f"Error loading logo file {bin_file}: {e}")
        return None

# ==========================================================================
# --- Sidebar ---
# ==========================================================================
with st.sidebar:
    # --- Logo Handling (Adapted from landing page) ---
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    LOGO_PATH = current_dir / cfg.LOGO_FILENAME # Use filename from config
    logo_base64 = get_base64_of_bin_file(LOGO_PATH)

    # Define the logo HTML - Adjust class/style for sidebar if needed
    # The CSS in styling.py should ideally handle '.sidebar-logo' class
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Phronesis Apex Logo" class="sidebar-logo">' if logo_base64 else '<div class="sidebar-logo-placeholder">Logo</div>'

    # Add CSS specifically for sidebar logo if not covered globally
    st.markdown("""
    <style>
    .sidebar-logo {
        display: block; /* Center logo */
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 1.5rem; /* Space below logo */
        height: 80px; /* Adjust size for sidebar */
        width: auto;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
    }
    .sidebar-logo-placeholder { /* Style placeholder if logo fails */
        height: 80px; width: 80px; background-color: #333; border: 1px dashed #555;
        display: flex; align-items: center; justify-content: center; color: #888;
        font-size: 0.9em; text-align: center; border-radius: 5px; margin: 0 auto 1.5rem auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display the logo using markdown
    st.markdown(logo_html, unsafe_allow_html=True)

    #st.header("⚙️ Settings & Controls") # Header remains

    # --- API Key Handling Logic (Unchanged) ---
    # ... (Keep the existing API key handling logic) ...
    api_key_source = None
    if st.session_state.get(cfg.INITIALIZED_STATE_KEY, False) and st.session_state.get(cfg.API_KEY_STATE_KEY):
        st.success("✅ Initialized") # Simpler message
        current_source = st.session_state.get(cfg.API_KEY_SOURCE_STATE_KEY, "N/A")
        st.caption(f"Key Source: {current_source.capitalize()}")
    else:
        secrets_api_key = None; secret_key_name = "GEMINI_API_KEY"
        try:
            if hasattr(st, 'secrets') and secret_key_name in st.secrets:
                secrets_api_key = st.secrets[secret_key_name]
                logging.info(f"Found API key in Streamlit Secrets ('{secret_key_name}').")
                api_key_source = "secrets"
            else:
                logging.info(f"'{secret_key_name}' not found in st.secrets. Fallback to manual input.")
                api_key_source = "manual"
        except Exception as e:
             logging.warning(f"Could not access Streamlit secrets: {e}. Assuming manual input needed.")
             api_key_source = "manual"

        if api_key_source == "secrets" and secrets_api_key:
            # st.info("🔑 Attempting initialization via Secrets...")
            is_valid, message = utils.validate_api_key(secrets_api_key)
            if is_valid:
                st.session_state[cfg.API_KEY_STATE_KEY] = secrets_api_key
                st.session_state[cfg.INITIALIZED_STATE_KEY] = True
                st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = "secrets"
                st.success("✅ Initialized via Secrets!")
                logging.info("Themer initialized using key from Streamlit Secrets.")
                time.sleep(0.5); st.rerun()
            else:
                st.error(f"Secrets key validation failed: {message}")
                logging.error(f"API Key from secrets failed validation: {message}")
                api_key_source = "manual"
                st.session_state[cfg.API_KEY_STATE_KEY] = None
                st.session_state[cfg.INITIALIZED_STATE_KEY] = False
                st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = None

        if not st.session_state.get(cfg.INITIALIZED_STATE_KEY, False):
            if api_key_source == "manual":
                # st.info("🔑 Gemini API Key required.")
                api_key_input = st.text_input("Enter Gemini API Key:", type="password", help="Get your key from Google AI Studio.", key="api_input_sidebar_manual")
                if st.button("Initialize Themer", key=cfg.INIT_BUTTON_KEY, type="primary"):
                    if api_key_input:
                        is_valid, message = utils.validate_api_key(api_key_input)
                        if is_valid:
                            st.session_state[cfg.API_KEY_STATE_KEY] = api_key_input
                            st.session_state[cfg.INITIALIZED_STATE_KEY] = True
                            st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = "manual"
                            st.success("✅ Initialized (Manual Key)!")
                            logging.info("Themer initialized using manually entered key.")
                            time.sleep(0.5); st.rerun()
                        else:
                            st.error(f"Initialization failed: {message}")
                            st.session_state[cfg.API_KEY_STATE_KEY] = None
                            st.session_state[cfg.INITIALIZED_STATE_KEY] = False
                            st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = None
                    else:
                        st.warning("Please enter your Gemini API Key.")

    if not st.session_state.get(cfg.INITIALIZED_STATE_KEY, False):
        st.warning("Please provide a valid Gemini API Key and initialize.")
        st.info("Enter key above or configure in Streamlit Secrets as `GEMINI_API_KEY`.")
        st.stop()

    # --- Post-Initialization Settings (Unchanged logic, uses state keys) ---
    st.markdown("---")
    st.subheader("AI Parameters")
    st.session_state[cfg.BATCH_SIZE_KEY] = st.slider("Batch Size (Assignment)", 1, 50, value=st.session_state.get(cfg.BATCH_SIZE_KEY, 15), key="batch_size_slider", help="Responses per API call during assignment.")
    st.session_state[cfg.SAMPLE_SIZE_KEY] = st.slider("Sample Size (Theme Gen)", 50, 500, value=st.session_state.get(cfg.SAMPLE_SIZE_KEY, 100), step=10, key="sample_size_slider", help="Responses used for initial theme discovery.")
    with st.expander("🤖 Advanced Generation Config", expanded=False):
        st.session_state[cfg.GEN_TEMP_KEY] = st.slider("Temperature", 0.0, 1.0, value=st.session_state.get(cfg.GEN_TEMP_KEY, 0.5), step=0.05, key="temp_slider", help="Controls randomness (lower = more deterministic).")
        st.session_state[cfg.GEN_TOP_K_KEY] = st.number_input("Top K", 1, 100, value=st.session_state.get(cfg.GEN_TOP_K_KEY, 40), step=1, key="topk_input")
        st.session_state[cfg.GEN_TOP_P_KEY] = st.slider("Top P", 0.0, 1.0, value=st.session_state.get(cfg.GEN_TOP_P_KEY, 0.95), step=0.05, key="topp_slider")
        st.session_state[cfg.GEN_MAX_TOKENS_KEY] = st.number_input("Max Output Tokens", 256, 8192, value=st.session_state.get(cfg.GEN_MAX_TOKENS_KEY, 4096), step=64, key="maxtokens_input", help="Max length of AI response per call.")

    # --- Reset Button (Unchanged) ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset API Key & Clear All Data", key=cfg.RESET_BUTTON_KEY):
        logging.warning("Reset button clicked. Clearing API key and all app data.")
        clear_app_state(clear_api_keys=True)
        st.success("API Key and all session data cleared. Re-initializing...")
        time.sleep(1); st.rerun()

# ==========================================================================
# --- Main Application Area ---
# ==========================================================================

# Use a simpler title now, logo is in sidebar
st.markdown("<h1 style='margin-bottom: 1rem; font-size: 2.2rem;'>AI Themer</h1>", unsafe_allow_html=True)
# Optional: Add a subtitle
# st.markdown("<p style='text-align: center; margin-bottom: 2rem; color: #8b98b8;'>Generate, refine, and apply themes to open-ended responses.</p>", unsafe_allow_html=True)


# Define tab names
tab_list = [
    "📥 Input & Generate",
    "✏️ Review & Refine",
    "🏷️ Assign & Edit Results",
    "📊 Explore & Visualize",
    "❓ Ask AI"
]
input_tab, review_tab, assign_edit_tab, viz_tab, ai_qa_tab = st.tabs(tab_list)

# ======================= Tab Content (Logic remains the same) =============
# The content generation within each tab uses the functions from
# ui_components and ai_core, which now operate within the new theme
# provided by styling.py and use updated parameters from config.py.

# Create GenerationConfig object here once using current state
# (as it was before, but ensuring state keys from cfg are used)
theming_gen_config = GenerationConfig(
    temperature=st.session_state[cfg.GEN_TEMP_KEY],
    top_k=st.session_state[cfg.GEN_TOP_K_KEY],
    top_p=st.session_state[cfg.GEN_TOP_P_KEY],
    max_output_tokens=st.session_state[cfg.GEN_MAX_TOKENS_KEY]
)

with input_tab:
    # ... (Keep existing Input Tab code - PASTE HERE) ...
    # It will now render with the new theme applied via CSS
    st.header("Input Data and Generate Initial Themes")
    st.session_state[cfg.SURVEY_QUESTION_KEY] = st.text_area("❓ **Survey Question:**", value=st.session_state.get(cfg.SURVEY_QUESTION_KEY, ''), height=100, key="theming_q_input_main", placeholder="e.g., What aspects of the recent workshop did you find most valuable?")
    st.markdown("---")
    st.subheader("Provide Responses")
    st.radio("Choose input method:", ("Paste Text", "Upload File (.csv, .xlsx)"), key=cfg.INPUT_METHOD_KEY, horizontal=True, label_visibility="collapsed")
    input_method = st.session_state[cfg.INPUT_METHOD_KEY]
    responses_list_input = []
    if input_method == "Paste Text":
        if st.session_state.get(cfg.CURRENT_FILE_NAME_KEY):
            st.session_state[cfg.UPLOADED_DF_KEY] = None; st.session_state[cfg.SELECTED_COLUMN_KEY] = None
            st.session_state[cfg.CURRENT_FILE_NAME_KEY] = None; st.session_state[cfg.SELECTED_COLUMN_IDX_KEY] = 0
            st.session_state[cfg.RESPONSES_RAW_KEY] = []; logging.info("Switched to text input, cleared file state.")
        st.session_state[cfg.RESPONSES_INPUT_AREA_VAL_KEY] = st.text_area("📋 Paste Responses (One per line):", value=st.session_state.get(cfg.RESPONSES_INPUT_AREA_VAL_KEY, ''), height=200, key="theming_responses_paste_area_widget", placeholder="Response 1...\nResponse 2...\nResponse 3...")
        raw_lines = st.session_state[cfg.RESPONSES_INPUT_AREA_VAL_KEY].splitlines()
        responses_list_input = [r.strip() for r in raw_lines if r and r.strip()]
        st.caption(f"{len(responses_list_input)} response(s) entered.")
        st.session_state[cfg.RESPONSES_RAW_KEY] = responses_list_input
    elif input_method == "Upload File":
        if st.session_state.get(cfg.RESPONSES_INPUT_AREA_VAL_KEY):
            st.session_state[cfg.RESPONSES_INPUT_AREA_VAL_KEY] = ''; st.session_state[cfg.RESPONSES_RAW_KEY] = []
            logging.info("Switched to file upload, cleared text input state.")
        uploaded_file = st.file_uploader("📁 Upload Data File:", type=['csv', 'xlsx', 'xls'], key="theming_file_uploader_widget", accept_multiple_files=False)
        if uploaded_file is not None:
            current_df = st.session_state.get(cfg.UPLOADED_DF_KEY)
            reload_file = (st.session_state.get(cfg.CURRENT_FILE_NAME_KEY) != uploaded_file.name) or (current_df is None or not isinstance(current_df, pd.DataFrame))
            if reload_file:
                logging.info(f"New file ('{uploaded_file.name}') detected or DF missing. Loading.")
                with st.spinner(f"Reading and processing {uploaded_file.name}..."):
                    st.session_state[cfg.SELECTED_COLUMN_KEY] = None; st.session_state[cfg.RESPONSES_RAW_KEY] = []
                    st.session_state[cfg.SELECTED_COLUMN_IDX_KEY] = 0; st.session_state[cfg.UPLOADED_DF_KEY] = None
                    loaded_data = utils.load_data_from_file(uploaded_file)
                    if loaded_data is not None and isinstance(loaded_data, pd.DataFrame) and not loaded_data.empty:
                        st.session_state[cfg.UPLOADED_DF_KEY] = loaded_data
                        st.session_state[cfg.CURRENT_FILE_NAME_KEY] = uploaded_file.name
                        logging.info(f"Successfully loaded DF from {uploaded_file.name}. Shape: {loaded_data.shape}")
                        st.success(f"Loaded `{uploaded_file.name}` ({len(loaded_data)} rows). Select column below.")
                        st.rerun()
                    else:
                        st.session_state[cfg.UPLOADED_DF_KEY] = None; st.session_state[cfg.CURRENT_FILE_NAME_KEY] = None
            df_for_selection = st.session_state.get(cfg.UPLOADED_DF_KEY)
            if df_for_selection is not None and isinstance(df_for_selection, pd.DataFrame):
                st.markdown("---")
                st.write(f"**File:** `{st.session_state.get(cfg.CURRENT_FILE_NAME_KEY, 'N/A')}` ({len(df_for_selection)} rows)")
                with st.expander("Preview Data"):
                    try: st.dataframe(df_for_selection.head(), use_container_width=True, height=200)
                    except Exception as e_disp: st.warning(f"Could not display file preview: {e_disp}")
                available_columns = df_for_selection.columns.tolist()
                if available_columns:
                    current_index = st.session_state.get(cfg.SELECTED_COLUMN_IDX_KEY, 0)
                    if current_index >= len(available_columns): current_index = 0
                    if current_index == 0 and st.session_state.get(cfg.SELECTED_COLUMN_KEY) is None:
                         plausible_cols = [c for i, c in enumerate(available_columns) if any(kw in str(c).lower() for kw in ['response', 'feedback', 'text', 'comment', 'verbatim', 'open end', 'answer'])]
                         if plausible_cols:
                             try: current_index = available_columns.index(plausible_cols[0]); logging.info(f"Auto-selected plausible column: '{plausible_cols[0]}'")
                             except ValueError: pass
                    selected_col_name = st.selectbox("⬇️ **Select column with responses:**", options=available_columns, index=current_index, key="theming_column_selector_widget", help="Choose the column with the text answers.")
                    if selected_col_name != st.session_state.get(cfg.SELECTED_COLUMN_KEY):
                         st.session_state[cfg.SELECTED_COLUMN_KEY] = selected_col_name
                         try: st.session_state[cfg.SELECTED_COLUMN_IDX_KEY] = available_columns.index(selected_col_name)
                         except ValueError: st.session_state[cfg.SELECTED_COLUMN_IDX_KEY] = 0
                         st.session_state[cfg.RESPONSES_RAW_KEY] = []
                    if selected_col_name and not st.session_state.get(cfg.RESPONSES_RAW_KEY):
                        try:
                             if selected_col_name in df_for_selection.columns:
                                 responses_series = df_for_selection[selected_col_name].astype(str).fillna('')
                                 responses_list_input = [r.strip() for r in responses_series if r and r.strip()]
                                 st.caption(f"{len(responses_list_input)} valid response(s) extracted from '{selected_col_name}'.")
                                 st.session_state[cfg.RESPONSES_RAW_KEY] = responses_list_input
                             else: st.error(f"Selected column '{selected_col_name}' not found."); st.session_state[cfg.RESPONSES_RAW_KEY] = []
                        except Exception as e: st.error(f"Error extracting data from column '{selected_col_name}': {e}"); logging.exception(f"Error extracting column data"); st.session_state[cfg.RESPONSES_RAW_KEY] = []
                else: st.error("Uploaded file has no columns after cleaning."); st.session_state[cfg.UPLOADED_DF_KEY] = None
        else:
             if st.session_state.get(cfg.CURRENT_FILE_NAME_KEY):
                 st.session_state[cfg.UPLOADED_DF_KEY] = None; st.session_state[cfg.SELECTED_COLUMN_KEY] = None
                 st.session_state[cfg.CURRENT_FILE_NAME_KEY] = None; st.session_state[cfg.RESPONSES_RAW_KEY] = []
                 st.session_state[cfg.SELECTED_COLUMN_IDX_KEY] = 0; logging.info("File removed, clearing related state.")
    st.markdown("---")
    st.subheader("Generate Themes")
    question_ready = st.session_state.get(cfg.SURVEY_QUESTION_KEY, '').strip()
    responses_ready = isinstance(st.session_state.get(cfg.RESPONSES_RAW_KEY), list) and bool(st.session_state.get(cfg.RESPONSES_RAW_KEY))
    can_generate = question_ready and responses_ready
    if not can_generate:
        if not question_ready: st.warning("⚠️ Please provide the Survey Question.")
        if not responses_ready: st.warning("⚠️ Please provide Responses.")
    if st.button("🤖 Generate Themes & Descriptions", key="generate_themes_desc_main_btn", disabled=not can_generate, type="primary"):
        if can_generate:
            question = st.session_state[cfg.SURVEY_QUESTION_KEY]; responses_raw = st.session_state[cfg.RESPONSES_RAW_KEY]
            sample_size = st.session_state.get(cfg.SAMPLE_SIZE_KEY, 100); api_key = st.session_state.get(cfg.API_KEY_STATE_KEY)
            st.session_state[cfg.GENERATED_THEMES_KEY] = None; st.session_state[cfg.EDITED_THEMES_KEY] = []
            st.session_state[cfg.ASSIGNMENT_DF_KEY] = None; logging.info("Cleared previous results.")
            actual_sample_size = min(sample_size, len(responses_raw))
            if actual_sample_size == 0: st.error("Cannot generate themes: No responses."); st.stop()
            if actual_sample_size < len(responses_raw):
                st.info(f"Using sample of {actual_sample_size} (out of {len(responses_raw)}) responses.")
                try: responses_sample = random.sample(responses_raw, actual_sample_size)
                except ValueError as e: st.error(f"Sample error: {e}. Using all."); responses_sample = responses_raw
            else: st.info(f"Using all {len(responses_raw)} responses."); responses_sample = responses_raw
            theme_gen_success = False
            with st.spinner("AI discovering themes..."):
                generated_themes = ai_core.generate_themes_with_llm(question, responses_sample, api_key, theming_gen_config)
            if generated_themes is not None:
                st.session_state[cfg.GENERATED_THEMES_KEY] = generated_themes
                if not generated_themes: st.warning("AI found no distinct themes."); theme_gen_success = True; st.session_state[cfg.EDITED_THEMES_KEY] = []
                else: st.success(f"Generated {len(generated_themes)} initial themes."); theme_gen_success = True
                if generated_themes: # Only generate descriptions if themes were found
                    with st.spinner("AI writing descriptions..."):
                        theme_descriptions_map = ai_core.generate_theme_descriptions_llm(generated_themes, question, responses_sample, api_key, theming_gen_config)
                    merged_themes = []; themes_without_desc = []
                    if isinstance(generated_themes, list):
                        for theme_item in generated_themes:
                            if isinstance(theme_item, dict):
                                theme_label = theme_item.get('theme', ''); new_item = theme_item.copy(); description = ''
                                if theme_descriptions_map and isinstance(theme_descriptions_map, dict): description = theme_descriptions_map.get(theme_label, '')
                                new_item['description'] = description
                                if not description and theme_label: themes_without_desc.append(theme_label)
                                merged_themes.append(new_item)
                        if theme_descriptions_map:
                            st.success("Descriptions generated.")
                            if themes_without_desc: st.warning(f"No descriptions for: {', '.join(themes_without_desc)}")
                        else: st.warning("Could not generate descriptions.")
                        st.session_state[cfg.EDITED_THEMES_KEY] = merged_themes
                        logging.info(f"Initialized '{cfg.EDITED_THEMES_KEY}' with {len(merged_themes)} themes.")
            else: st.error("Failed to generate themes."); st.session_state[cfg.GENERATED_THEMES_KEY] = None; st.session_state[cfg.EDITED_THEMES_KEY] = []
            if theme_gen_success: st.info("✅ Proceed to '✏️ Review & Refine' tab."); time.sleep(1)

with review_tab:
    # ... (Keep existing Review Tab code - PASTE HERE) ...
    # It calls ui_components.display_theme_editor which is styled by CSS
    st.header("2. Review and Refine Themes")
    if not st.session_state.get(cfg.EDITED_THEMES_KEY) and not st.session_state.get(cfg.GENERATED_THEMES_KEY):
         st.info("Generate themes on the '📥 Input & Generate' tab or load a theme structure below.")
    st.markdown("""Review the AI-generated themes (or add/load your own). Edit labels, add/remove sub-themes, delete themes, or load/download the structure. **Click '💾 Save ALL Refined Themes'** to confirm changes.""")
    st.markdown("---")
    ui_components.display_theme_editor(themes_data_state_key=cfg.EDITED_THEMES_KEY)

with assign_edit_tab:
    # ... (Keep existing Assign Tab code - PASTE HERE) ...
    # Calls ui_components.display_assignment_results_editable, styled by CSS
    st.header("3. Assign Themes & Edit Results")
    themes_ready = isinstance(st.session_state.get(cfg.EDITED_THEMES_KEY), list) and bool(st.session_state.get(cfg.EDITED_THEMES_KEY))
    responses_ready = isinstance(st.session_state.get(cfg.RESPONSES_RAW_KEY), list) and bool(st.session_state.get(cfg.RESPONSES_RAW_KEY))
    st.subheader("Assign Themes to All Responses")
    if not themes_ready: st.warning("⚠️ Please define/save themes in '✏️ Review & Refine'.")
    elif not responses_ready: st.warning("⚠️ Please provide responses in '📥 Input & Generate'.")
    else:
        final_theme_structure_assign = st.session_state[cfg.EDITED_THEMES_KEY]
        all_responses_assign = st.session_state[cfg.RESPONSES_RAW_KEY]; num_responses_assign = len(all_responses_assign)
        num_themes_assign = len(final_theme_structure_assign); batch_size = st.session_state.get(cfg.BATCH_SIZE_KEY, 15)
        api_key = st.session_state.get(cfg.API_KEY_STATE_KEY); question = st.session_state.get(cfg.SURVEY_QUESTION_KEY, "(Survey question missing)")
        st.info(f"Ready to assign **{num_themes_assign}** themes to **{num_responses_assign}** responses (batch size: {batch_size}).")
        disable_assign = not final_theme_structure_assign
        if st.button(f"🏷️ Assign Themes to {num_responses_assign} Responses", key="assign_themes_main_button", type="primary", disabled=disable_assign):
            logging.info(f"Starting assignment for {num_responses_assign} responses, batch size {batch_size}.")
            st.session_state[cfg.ASSIGNMENT_DF_KEY] = None; all_assignments = []
            progress_bar = st.progress(0, text="Initializing assignment...")
            response_batches = utils.batch_responses(all_responses_assign, batch_size); total_batches = len(response_batches)
            start_time_assign = time.time(); assignment_errors = 0
            with st.spinner(f"AI assigning themes... ({total_batches} batches)"):
                for i, batch in enumerate(response_batches):
                    if not batch: continue
                    progress = min(1.0, (i + 1) / total_batches); elapsed_time = time.time() - start_time_assign
                    est_remaining = (elapsed_time / (i + 1)) * (total_batches - (i + 1)) if (i + 1) > 0 else 0
                    progress_text = f"Assigning: Batch {i+1}/{total_batches} ({i*batch_size+len(batch)}/{num_responses_assign}) | Est. time left: {est_remaining:.0f}s"
                    progress_bar.progress(progress, text=progress_text); logging.debug(f"Processing batch {i+1}/{total_batches}")
                    batch_results = ai_core.assign_themes_with_llm_batch(question, batch, final_theme_structure_assign, api_key, theming_gen_config)
                    if isinstance(batch_results, list) and len(batch_results) == len(batch):
                        all_assignments.extend(batch_results)
                        errors_in_batch = sum(1 for res in batch_results if isinstance(res, dict) and "Error" in res.get('assigned_theme', ''))
                        if errors_in_batch > 0: assignment_errors += errors_in_batch; logging.warning(f"Batch {i+1} had {errors_in_batch} errors.")
                    else:
                        st.error(f"Error processing batch {i+1}. Adding placeholders."); logging.error(f"Assignment failed batch {i+1}.")
                        error_placeholders = [{'assigned_theme': 'Error: Batch Failed', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(batch)
                        all_assignments.extend(error_placeholders); assignment_errors += len(batch)
            total_time_assign = time.time() - start_time_assign
            progress_bar.progress(1.0, text=f"Assignment complete! Time: {total_time_assign:.1f}s")
            time.sleep(1); progress_bar.empty()
            if len(all_assignments) == num_responses_assign:
                try:
                    first_valid_result = next((res for res in all_assignments if isinstance(res, dict) and "Error" not in res.get('assigned_theme','')), None)
                    df_cols = list(first_valid_result.keys()) if first_valid_result else ['assigned_theme', 'assigned_sub_theme', 'assignment_confidence']
                    results_df = pd.DataFrame(all_assignments, columns=df_cols)
                    results_df.insert(0, 'response', all_responses_assign)
                    st.session_state[cfg.ASSIGNMENT_DF_KEY] = results_df
                    logging.info(f"Assignment finished. Shape: {results_df.shape}. Errors: {assignment_errors}")
                    if assignment_errors > 0: st.warning(f"✅ Assignment complete, but {assignment_errors} errors occurred.")
                    else: st.success("✅ Theme assignment complete!")
                    st.rerun()
                except Exception as df_e: st.error(f"Failed to create results DF: {df_e}"); logging.exception("Error creating DF."); st.session_state[cfg.ASSIGNMENT_DF_KEY] = None
            else: st.error(f"❌ Assignment failed: Results count mismatch ({len(all_assignments)} vs {num_responses_assign})."); logging.error("Assignment length mismatch."); st.session_state[cfg.ASSIGNMENT_DF_KEY] = None
    st.markdown("---")
    st.subheader("View and Manually Edit Assigned Themes")
    ui_components.display_assignment_results_editable(df_state_key=cfg.ASSIGNMENT_DF_KEY, themes_state_key=cfg.EDITED_THEMES_KEY)


with viz_tab:
    # ... (Keep existing Viz Tab code - PASTE HERE) ...
    # Calls ui_components functions which were updated for dark theme
    st.header("📊 Explore & Visualize Themes")
    assignment_df_exists = cfg.ASSIGNMENT_DF_KEY in st.session_state and isinstance(st.session_state.get(cfg.ASSIGNMENT_DF_KEY), pd.DataFrame) and not st.session_state.get(cfg.ASSIGNMENT_DF_KEY, pd.DataFrame()).empty
    if not assignment_df_exists: st.info("Assign themes in '🏷️ Assign & Edit Results' to see visualizations.")
    else:
        st.subheader("Explore Themes and Examples")
        ui_components.display_theme_examples(df_state_key=cfg.ASSIGNMENT_DF_KEY, themes_state_key=cfg.EDITED_THEMES_KEY)
        st.markdown("---")
        st.subheader("Visualize Distributions")
        ui_components.display_theme_distribution(df_state_key=cfg.ASSIGNMENT_DF_KEY)
        st.markdown("---")
        st.subheader("☁️ Word Cloud per Theme")
        results_df_viz = st.session_state[cfg.ASSIGNMENT_DF_KEY]
        try:
            valid_themes_for_wc = sorted([str(th) for th in results_df_viz['assigned_theme'].unique() if pd.notna(th) and isinstance(th, str) and "Error" not in th and th != "Uncategorized"])
            if not valid_themes_for_wc: st.info("No valid themes found for word cloud.")
            else:
                 selected_theme_for_wc = st.selectbox("Select Theme for Word Cloud:", valid_themes_for_wc, key="wc_theme_select_viz_tab_widget")
                 if selected_theme_for_wc:
                     if 'assigned_theme' in results_df_viz.columns and 'response' in results_df_viz.columns:
                         theme_responses_wc = results_df_viz.loc[results_df_viz['assigned_theme'] == selected_theme_for_wc, 'response'].tolist()
                     else: st.error("Required columns missing for word cloud."); theme_responses_wc = []
                     if theme_responses_wc:
                         with st.spinner(f"Generating Word Cloud for '{selected_theme_for_wc}'..."):
                             fig_wc = ui_components.create_word_cloud(theme_responses_wc)
                             if fig_wc: st.pyplot(fig_wc, use_container_width=True)
                     else: st.info(f"No responses found for theme '{selected_theme_for_wc}'.")
        except Exception as wc_e: st.error(f"Error in word cloud section: {wc_e}"); logging.exception("Word cloud UI error.")


with ai_qa_tab:
    # ... (Keep existing AI QA Tab code - PASTE HERE) ...
    # Styling handled by CSS
    st.header("❓ Ask AI About the Raw Data")
    responses_available_qa = isinstance(st.session_state.get(cfg.RESPONSES_RAW_KEY), list) and bool(st.session_state.get(cfg.RESPONSES_RAW_KEY))
    question_available_qa = isinstance(st.session_state.get(cfg.SURVEY_QUESTION_KEY), str) and bool(st.session_state.get(cfg.SURVEY_QUESTION_KEY, '').strip())
    if not responses_available_qa or not question_available_qa: st.info("Provide survey question and responses in '📥 Input & Generate' first.")
    else:
        question_qa = st.session_state[cfg.SURVEY_QUESTION_KEY]; responses_list_qa = st.session_state[cfg.RESPONSES_RAW_KEY]
        api_key = st.session_state.get(cfg.API_KEY_STATE_KEY)
        st.markdown("Ask a question about the **original raw responses**.")
        st.info(f"Context: Question \"_{question_qa}_\" and {len(responses_list_qa)} responses.")
        user_ai_question = st.text_area("Your Question:", key="ai_qa_input_main_widget", height=100, placeholder="e.g., Summarize the feedback mentioning 'support'.")
        if st.button("Ask AI", key="ask_ai_button_main_widget", type="primary"):
            user_question_stripped = user_ai_question.strip()
            if not user_question_stripped: st.warning("Please enter a question.")
            else:
                with st.spinner("🧠 AI is thinking..."):
                    if not api_key: st.error("API Key not found."); ai_answer = "Error: API Key missing."
                    else: ai_answer = ai_core.ask_ai_about_data(question_qa, responses_list_qa, user_question_stripped, theming_gen_config, api_key)
                if ai_answer is not None:
                     st.session_state[cfg.AI_QA_HISTORY_KEY].insert(0, {"question": user_question_stripped, "answer": ai_answer})
                     logging.info(f"AI Q&A executed. Q: '{user_question_stripped[:50]}...'")
                     # Clear input box by resetting its state variable (widget key) after adding to history
                     st.session_state.ai_qa_input_main_widget = "" # Reset widget value
                     st.rerun()
        st.markdown("---"); st.subheader("Q&A History")
        if st.session_state[cfg.AI_QA_HISTORY_KEY]:
             history_limit = 5
             for i, qa_pair in enumerate(st.session_state[cfg.AI_QA_HISTORY_KEY]):
                 q_text = str(qa_pair.get('question', '(...)'))
                 a_text = str(qa_pair.get('answer', '...'))
                 with st.expander(f"Q: {q_text}", expanded=(i < history_limit)):
                     st.markdown(f"**AI Answer:**"); st.markdown(a_text, unsafe_allow_html=True)
        else: st.caption("No questions asked yet.")

# --- Footer (Add the custom footer from landing page) ---
st.markdown(
    f"""
    <div class="footer">
        <p>© {time.strftime('%Y')} Phronesis Partners. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- End of App ---
logging.info("Reached end of main_app.py execution (Apex Theme).")
