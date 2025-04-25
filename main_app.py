# main_app.py (for AI Themer App)
"""
Main Streamlit application file for the PPL APEX AI Themer.
Orchestrates the UI, state management, and interactions between modules.
Includes fixes for column detection, Ask AI input clearing, and sampling logic.
"""

# --- Core Imports ---
import streamlit as st
import pandas as pd
import logging
import time
import random
from pathlib import Path
import base64
from google.generativeai.types import GenerationConfig

# --- Application Modules Imports ---
import config as cfg
from styling import apply_styling
import utils
import ai_core
import ui_components

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI Themer - Apex",
    page_icon="🏷️"
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("AI Themer Application Started (Apex Theme).")

# --- Apply Custom Styling ---
apply_styling()

# --- Session State Initialization ---
default_state_values = {
    cfg.INITIALIZED_STATE_KEY: False, cfg.API_KEY_STATE_KEY: None,
    cfg.API_KEY_SOURCE_STATE_KEY: None, cfg.INPUT_METHOD_KEY: 'Paste Text',
    cfg.SURVEY_QUESTION_KEY: '', cfg.RESPONSES_RAW_KEY: [],
    cfg.UPLOADED_DF_KEY: None, cfg.SELECTED_COLUMN_KEY: None,
    cfg.CURRENT_FILE_NAME_KEY: None, cfg.RESPONSES_INPUT_AREA_VAL_KEY: '',
    cfg.SELECTED_COLUMN_IDX_KEY: 0, cfg.GENERATED_THEMES_KEY: None,
    cfg.EDITED_THEMES_KEY: [], cfg.ASSIGNMENT_DF_KEY: None,
    cfg.AI_QA_HISTORY_KEY: [],
    cfg.BATCH_SIZE_KEY: 15, cfg.SAMPLE_SIZE_KEY: 100, cfg.GEN_TEMP_KEY: 0.5,
    cfg.GEN_TOP_K_KEY: 40, cfg.GEN_TOP_P_KEY: 0.95, cfg.GEN_MAX_TOKENS_KEY: 4096,
    #cfg.UPLOADED_THEME_FILE_WIDGET_KEY: None, 
    #cfg.ASSIGNMENT_EDITOR_WIDGET_KEY: None,
}
for key, default_value in default_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
        logging.debug(f"Initialized Themer session state key '{key}' with default.")

# --- Helper function to clear app-specific state ---
def clear_themer_app_state(clear_api_keys=False):
    """Clears specific session state keys related to the themer app."""
    keys_to_clear = [
        cfg.SURVEY_QUESTION_KEY, cfg.RESPONSES_RAW_KEY, cfg.UPLOADED_DF_KEY,
        cfg.SELECTED_COLUMN_KEY, cfg.INPUT_METHOD_KEY, cfg.CURRENT_FILE_NAME_KEY,
        cfg.GENERATED_THEMES_KEY, cfg.EDITED_THEMES_KEY, cfg.ASSIGNMENT_DF_KEY,
        cfg.AI_QA_HISTORY_KEY, cfg.RESPONSES_INPUT_AREA_VAL_KEY, cfg.SELECTED_COLUMN_IDX_KEY,
        #cfg.UPLOADED_THEME_FILE_WIDGET_KEY, 
        cfg.ASSIGNMENT_EDITOR_WIDGET_KEY,
        "ai_qa_input_main_widget", # Clearing specific widget state key if it exists
    ]
    if clear_api_keys:
        keys_to_clear.extend([
            cfg.INITIALIZED_STATE_KEY, cfg.API_KEY_STATE_KEY, cfg.API_KEY_SOURCE_STATE_KEY,
            cfg.BATCH_SIZE_KEY, cfg.SAMPLE_SIZE_KEY, cfg.GEN_TEMP_KEY,
            cfg.GEN_TOP_K_KEY, cfg.GEN_TOP_P_KEY, cfg.GEN_MAX_TOKENS_KEY,
        ])
    cleared_count = 0
    for key in keys_to_clear:
        if key in st.session_state:
            try: del st.session_state[key]
            except KeyError: pass
            cleared_count += 1
            logging.debug(f"Cleared Themer session state key: {key}")
    if clear_api_keys:
        st.session_state[cfg.INITIALIZED_STATE_KEY] = False; st.session_state[cfg.API_KEY_STATE_KEY] = None; st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = None
        st.session_state[cfg.BATCH_SIZE_KEY] = 15; st.session_state[cfg.SAMPLE_SIZE_KEY] = 100 # Reset defaults
        logging.info(f"Cleared {cleared_count} Themer keys and reset API/init/settings state.")
    else:
         st.session_state[cfg.RESPONSES_RAW_KEY] = []; st.session_state[cfg.EDITED_THEMES_KEY] = []; st.session_state[cfg.AI_QA_HISTORY_KEY] = []; st.session_state[cfg.ASSIGNMENT_DF_KEY] = None
         logging.info(f"Cleared {cleared_count} Themer data/state keys.")

# --- Logo Loading Function ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(str(bin_file), 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: st.sidebar.warning(f"Logo file '{cfg.LOGO_FILENAME}' not found."); logging.warning(f"Logo not found: {bin_file}"); return None
    except Exception as e: st.sidebar.error(f"Error loading logo: {e}"); logging.error(f"Logo error {bin_file}: {e}"); return None

# ==========================================================================
# --- Sidebar ---
# ==========================================================================
with st.sidebar:
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd(); LOGO_PATH = current_dir / cfg.LOGO_FILENAME
    logo_base64 = get_base64_of_bin_file(LOGO_PATH)
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Logo" class="sidebar-logo">' if logo_base64 else '<div class="sidebar-logo-placeholder">Logo</div>'
    st.markdown(logo_html, unsafe_allow_html=True) # Logo CSS is in styling.py
    #st.header("⚙️ Settings & Controls")

    # --- API Key Handling ---
    api_key_source = None
    if st.session_state.get(cfg.INITIALIZED_STATE_KEY, False) and st.session_state.get(cfg.API_KEY_STATE_KEY):
        st.success("✅ Initialized"); api_key_source = st.session_state.get(cfg.API_KEY_SOURCE_STATE_KEY, "manual"); st.caption(f"Key Source: {api_key_source.capitalize()}")
    else:
        secrets_api_key = None; secret_key_name = "GEMINI_API_KEY"
        try:
            if hasattr(st, 'secrets') and secret_key_name in st.secrets: secrets_api_key = st.secrets[secret_key_name]; api_key_source = "secrets"; logging.info(f"Themer: Found API key in Secrets ('{secret_key_name}').")
            else: api_key_source = "manual"
        except Exception as e: logging.warning(f"Themer: Could not access Secrets: {e}"); api_key_source = "manual"
        if api_key_source == "secrets" and secrets_api_key:
            is_valid, message = utils.validate_api_key(secrets_api_key)
            if is_valid:
                st.session_state[cfg.API_KEY_STATE_KEY] = secrets_api_key; st.session_state[cfg.INITIALIZED_STATE_KEY] = True; st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = "secrets"
                st.success("✅ Initialized via Secrets!"); logging.info("Themer: Initialized via Secrets."); time.sleep(0.5); st.rerun()
            else: st.error(f"Secrets key invalid: {message}"); logging.error(f"Themer: Secrets key fail: {message}"); api_key_source = "manual"; st.session_state.pop(cfg.API_KEY_STATE_KEY, None); st.session_state.pop(cfg.INITIALIZED_STATE_KEY, None); st.session_state.pop(cfg.API_KEY_SOURCE_STATE_KEY, None)
        if not st.session_state.get(cfg.INITIALIZED_STATE_KEY, False):
            if api_key_source == "manual":
                api_key_input = st.text_input("Enter Gemini API Key:", type="password", help="Get key from Google AI Studio.", key="themer_api_input_sidebar")
                if st.button("Initialize Themer", key=cfg.INIT_BUTTON_KEY, type="primary"):
                    if api_key_input:
                        is_valid, message = utils.validate_api_key(api_key_input)
                        if is_valid: st.session_state[cfg.API_KEY_STATE_KEY] = api_key_input; st.session_state[cfg.INITIALIZED_STATE_KEY] = True; st.session_state[cfg.API_KEY_SOURCE_STATE_KEY] = "manual"; st.success("✅ Initialized (Manual Key)!"); logging.info("Themer: Initialized via manual key."); time.sleep(0.5); st.rerun()
                        else: st.error(f"Init failed: {message}"); st.session_state.pop(cfg.API_KEY_STATE_KEY, None); st.session_state.pop(cfg.INITIALIZED_STATE_KEY, None); st.session_state.pop(cfg.API_KEY_SOURCE_STATE_KEY, None)
                    else: st.warning("Please enter API Key.")

    if not st.session_state.get(cfg.INITIALIZED_STATE_KEY, False): st.warning("Please provide a valid Gemini API Key and initialize."); st.info("Enter key above or configure in Streamlit Secrets."); st.stop()

    # --- Post-Initialization Settings ---
    st.markdown("---"); st.subheader("AI Parameters")
    st.session_state[cfg.BATCH_SIZE_KEY] = st.slider("Batch Size (Assignment)", 1, 50, value=st.session_state.get(cfg.BATCH_SIZE_KEY, 15), key="themer_batch_size_slider")
    st.session_state[cfg.SAMPLE_SIZE_KEY] = st.slider("Sample Size (Theme Gen)", 50, 500, value=st.session_state.get(cfg.SAMPLE_SIZE_KEY, 100), step=10, key="themer_sample_size_slider")
    with st.expander("🤖 Advanced Generation Config", expanded=False):
        st.session_state[cfg.GEN_TEMP_KEY] = st.slider("Temperature", 0.0, 1.0, value=st.session_state.get(cfg.GEN_TEMP_KEY, 0.5), step=0.05, key="themer_temp_slider")
        st.session_state[cfg.GEN_TOP_K_KEY] = st.number_input("Top K", 1, 100, value=st.session_state.get(cfg.GEN_TOP_K_KEY, 40), step=1, key="themer_topk_input")
        st.session_state[cfg.GEN_TOP_P_KEY] = st.slider("Top P", 0.0, 1.0, value=st.session_state.get(cfg.GEN_TOP_P_KEY, 0.95), step=0.05, key="themer_topp_slider")
        st.session_state[cfg.GEN_MAX_TOKENS_KEY] = st.number_input("Max Output Tokens", 256, 8192, value=st.session_state.get(cfg.GEN_MAX_TOKENS_KEY, 4096), step=128, key="themer_maxtokens_input")

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset API Key & Clear All Data", key=cfg.RESET_BUTTON_KEY): logging.warning("Themer: Reset button clicked."); clear_themer_app_state(clear_api_keys=True); st.success("API Key and session data cleared. Re-initializing..."); time.sleep(1); st.rerun()

# ==========================================================================
# --- Main Application Area ---
# ==========================================================================
st.markdown("<h1 style='margin-bottom: 1rem; font-size: 2.2rem;'>AI Themer-Beta</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem; color: #a9b4d2;'>Generate, refine, and apply themes to open-ended responses.</p>", unsafe_allow_html=True)

tab_list = [ "📥 Input & Generate", "✏️ Review & Refine", "🏷️ Assign & Edit Results", "📊 Explore & Visualize", "❓ Ask AI" ]
input_tab, review_tab, assign_edit_tab, viz_tab, ai_qa_tab = st.tabs(tab_list)

theming_gen_config = GenerationConfig(temperature=st.session_state[cfg.GEN_TEMP_KEY], top_k=st.session_state[cfg.GEN_TOP_K_KEY], top_p=st.session_state[cfg.GEN_TOP_P_KEY], max_output_tokens=st.session_state[cfg.GEN_MAX_TOKENS_KEY])

# ======================= Tab 1: Input & Generate ==========================
with input_tab:
    st.header("Input Data and Generate Initial Themes")
    st.session_state[cfg.SURVEY_QUESTION_KEY] = st.text_area("❓ **Survey Question:**", value=st.session_state.get(cfg.SURVEY_QUESTION_KEY, ''), height=100, key="theming_q_input_main", placeholder="e.g., What aspects of the event did you find most valuable?")
    st.markdown("---"); st.subheader("Provide Responses")
    st.radio("Choose input method:", ("Paste Text", "Upload File (.csv, .xlsx)"), key=cfg.INPUT_METHOD_KEY, horizontal=True, label_visibility="collapsed")
    input_method = st.session_state[cfg.INPUT_METHOD_KEY]; responses_list_input = []
    if input_method == "Paste Text":
        if st.session_state.get(cfg.CURRENT_FILE_NAME_KEY): keys_to_clear = [cfg.UPLOADED_DF_KEY, cfg.SELECTED_COLUMN_KEY, cfg.CURRENT_FILE_NAME_KEY, cfg.RESPONSES_RAW_KEY, cfg.SELECTED_COLUMN_IDX_KEY]; [st.session_state.pop(k, None) for k in keys_to_clear]; logging.info("Themer: Cleared file state.")
        st.session_state[cfg.RESPONSES_INPUT_AREA_VAL_KEY] = st.text_area("📋 Paste Responses (One per line):", value=st.session_state.get(cfg.RESPONSES_INPUT_AREA_VAL_KEY, ''), height=200, key="theming_responses_paste_area_widget", placeholder="Response 1...\nResponse 2...\nResponse 3...")
        raw_lines = st.session_state[cfg.RESPONSES_INPUT_AREA_VAL_KEY].splitlines(); responses_list_input = [r.strip() for r in raw_lines if r and r.strip()]
        st.caption(f"{len(responses_list_input)} response(s) entered."); st.session_state[cfg.RESPONSES_RAW_KEY] = responses_list_input
    elif input_method == "Upload File":
        if st.session_state.get(cfg.RESPONSES_INPUT_AREA_VAL_KEY): st.session_state.pop(cfg.RESPONSES_INPUT_AREA_VAL_KEY, None); st.session_state.pop(cfg.RESPONSES_RAW_KEY, None); logging.info("Themer: Cleared text input state.")
        uploaded_file = st.file_uploader("📁 Upload Data File:", type=['csv', 'xlsx', 'xls'], key="theming_file_uploader_widget", accept_multiple_files=False)
        if uploaded_file is not None:
            current_df = st.session_state.get(cfg.UPLOADED_DF_KEY); reload_file = (st.session_state.get(cfg.CURRENT_FILE_NAME_KEY) != uploaded_file.name) or (current_df is None or not isinstance(current_df, pd.DataFrame))
            if reload_file:
                logging.info(f"Themer: New file ('{uploaded_file.name}') loading."); st.session_state[cfg.RESPONSES_RAW_KEY] = []
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    keys_to_clear = [cfg.SELECTED_COLUMN_KEY, cfg.SELECTED_COLUMN_IDX_KEY, cfg.UPLOADED_DF_KEY]; [st.session_state.pop(k, None) for k in keys_to_clear]
                    loaded_data = utils.load_data_from_file(uploaded_file)
                    if loaded_data is not None and not loaded_data.empty: st.session_state[cfg.UPLOADED_DF_KEY] = loaded_data; st.session_state[cfg.CURRENT_FILE_NAME_KEY] = uploaded_file.name; logging.info(f"Themer: Loaded DF {loaded_data.shape}."); st.success(f"Loaded `{uploaded_file.name}` ({len(loaded_data)} rows)."); st.rerun()
                    else: st.session_state.pop(cfg.UPLOADED_DF_KEY, None); st.session_state.pop(cfg.CURRENT_FILE_NAME_KEY, None)
            df_for_selection = st.session_state.get(cfg.UPLOADED_DF_KEY)
            if df_for_selection is not None and isinstance(df_for_selection, pd.DataFrame):
                st.markdown("---"); st.write(f"**File:** `{st.session_state.get(cfg.CURRENT_FILE_NAME_KEY, 'N/A')}` ({len(df_for_selection)} rows)")
                with st.expander("Preview Data"):
                    try: st.dataframe(df_for_selection.head(), use_container_width=True, height=200)
                    except Exception as e: st.warning(f"Preview error: {e}")
                available_columns = df_for_selection.columns.tolist()
                if available_columns:
                    current_index = st.session_state.get(cfg.SELECTED_COLUMN_IDX_KEY, 0); current_index = 0 if current_index >= len(available_columns) else current_index
                    if st.session_state.get(cfg.SELECTED_COLUMN_KEY) is None:
                        kw = ['response', 'feedback', 'text', 'comment', 'verbatim', 'open end', 'answer']; pc = [c for c in available_columns if any(k in str(c).lower() for k in kw)]
                        # --- CORRECTED Column Detection BLOCK ---
                        if pc:
                            try: current_index = available_columns.index(pc[0]); logging.info(f"Themer: Auto-selected '{pc[0]}' @ index {current_index}")
                            except ValueError: pass
                        # --- END CORRECTED BLOCK ---
                    selected_col_name = st.selectbox("⬇️ **Select column with responses:**", options=available_columns, index=current_index, key="theming_column_selector_widget")
                    if selected_col_name != st.session_state.get(cfg.SELECTED_COLUMN_KEY): st.session_state[cfg.SELECTED_COLUMN_KEY] = selected_col_name; st.session_state[cfg.RESPONSES_RAW_KEY] = []
                    try: st.session_state[cfg.SELECTED_COLUMN_IDX_KEY] = available_columns.index(selected_col_name)
                    except ValueError: st.session_state[cfg.SELECTED_COLUMN_IDX_KEY] = 0
                    if selected_col_name and not st.session_state.get(cfg.RESPONSES_RAW_KEY):
                        try:
                            if selected_col_name in df_for_selection.columns: series = df_for_selection[selected_col_name].astype(str).fillna(''); responses_list_input = [r.strip() for r in series if r and r.strip()]; st.caption(f"{len(responses_list_input)} valid response(s) extracted."); st.session_state[cfg.RESPONSES_RAW_KEY] = responses_list_input
                            else: st.error(f"Column '{selected_col_name}' missing."); st.session_state[cfg.RESPONSES_RAW_KEY] = []
                        except Exception as e: st.error(f"Extraction error: {e}"); logging.exception("Themer: Column extraction error."); st.session_state[cfg.RESPONSES_RAW_KEY] = []
                else: st.error("Uploaded file has no columns."); st.session_state.pop(cfg.UPLOADED_DF_KEY, None)
        else:
            if st.session_state.get(cfg.CURRENT_FILE_NAME_KEY): keys_to_clear = [cfg.UPLOADED_DF_KEY, cfg.SELECTED_COLUMN_KEY, cfg.CURRENT_FILE_NAME_KEY, cfg.RESPONSES_RAW_KEY, cfg.SELECTED_COLUMN_IDX_KEY]; [st.session_state.pop(k, None) for k in keys_to_clear]; logging.info("Themer: File removed.")

    # --- Generate Themes Button ---
    st.markdown("---"); st.subheader("Generate Themes")
    question_ready = st.session_state.get(cfg.SURVEY_QUESTION_KEY, '').strip(); responses_ready = isinstance(st.session_state.get(cfg.RESPONSES_RAW_KEY), list) and bool(st.session_state.get(cfg.RESPONSES_RAW_KEY)); can_generate = question_ready and responses_ready
    if not can_generate:
        if not question_ready: st.warning("⚠️ Enter Survey Question.")
        if not responses_ready: st.warning("⚠️ Provide Responses.")
    if st.button("🤖 Generate Themes & Descriptions", key="generate_themes_desc_main_btn", disabled=not can_generate, type="primary"):
        if can_generate:
            question = st.session_state[cfg.SURVEY_QUESTION_KEY]; responses_raw = st.session_state[cfg.RESPONSES_RAW_KEY]; sample_size = st.session_state.get(cfg.SAMPLE_SIZE_KEY, 100); api_key = st.session_state.get(cfg.API_KEY_STATE_KEY)
            keys_to_clear = [cfg.GENERATED_THEMES_KEY, cfg.EDITED_THEMES_KEY, cfg.ASSIGNMENT_DF_KEY]; [st.session_state.pop(k, None) for k in keys_to_clear]; logging.info("Themer: Cleared previous results.")
            actual_sample_size = min(sample_size, len(responses_raw)); responses_sample = []
            if actual_sample_size == 0: st.error("No responses available."); st.stop()

            # --- CORRECTED SAMPLING LOGIC ---
            if actual_sample_size < len(responses_raw) and actual_sample_size > 0:
                st.info(f"Using a random sample of {actual_sample_size} (out of {len(responses_raw)}) responses for theme generation.")
                try:
                    k_sample = min(actual_sample_size, len(responses_raw)) # Ensure k <= population
                    responses_sample = random.sample(responses_raw, k_sample)
                except ValueError as sample_e:
                    st.error(f"Error creating sample: {sample_e}. Using all responses instead.")
                    logging.warning(f"Random sampling failed ({sample_e}), using all {len(responses_raw)} responses.")
                    responses_sample = responses_raw # Fallback to using all
            else: # Using all responses
                st.info(f"Using all {len(responses_raw)} responses for theme generation.")
                responses_sample = responses_raw
            # --- END CORRECTED SAMPLING LOGIC ---

            theme_gen_success = False
            with st.spinner("AI discovering themes..."): generated_themes = ai_core.generate_themes_with_llm(question, responses_sample, api_key, theming_gen_config)
            if generated_themes is not None:
                st.session_state[cfg.GENERATED_THEMES_KEY] = generated_themes; st.session_state[cfg.EDITED_THEMES_KEY] = [] # Reset edited themes
                if not generated_themes: st.warning("AI found no themes."); theme_gen_success = True
                else: st.success(f"Generated {len(generated_themes)} initial themes."); theme_gen_success = True
                if generated_themes:
                    with st.spinner("AI writing descriptions..."): desc_map = ai_core.generate_theme_descriptions_llm(generated_themes, question, responses_sample, api_key, theming_gen_config)
                    merged_themes = []; themes_no_desc = []
                    if isinstance(generated_themes, list):
                        for item in generated_themes:
                            if isinstance(item, dict): label = item.get('theme',''); new = item.copy(); d = desc_map.get(label, '') if desc_map and isinstance(desc_map, dict) else ''; new['description'] = d; merged_themes.append(new);
                            if not d and label: themes_no_desc.append(label)
                        if desc_map: st.success("Descriptions generated.");
                        if themes_no_desc: st.warning(f"No descriptions for: {', '.join(themes_no_desc)}")
                        st.session_state[cfg.EDITED_THEMES_KEY] = merged_themes; logging.info(f"Themer: Initialized edited themes: {len(merged_themes)}")
            else: st.error("Theme generation failed."); st.session_state[cfg.GENERATED_THEMES_KEY] = None; st.session_state[cfg.EDITED_THEMES_KEY] = []
            if theme_gen_success: st.info("✅ Proceed to '✏️ Review & Refine'."); time.sleep(1)


# ======================= Tab 2: Review & Refine ===========================
with review_tab:
    st.header("2. Review and Refine Themes")
    if not st.session_state.get(cfg.EDITED_THEMES_KEY) and not st.session_state.get(cfg.GENERATED_THEMES_KEY): st.info("Generate themes on '📥 Input & Generate' tab or load structure below.")
    st.markdown("Review AI themes (or add/load). Edit labels, manage sub-themes, delete themes, load/download. **Click '💾 Save ALL Refined Themes'** to confirm.")
    st.markdown("---")
    ui_components.display_theme_editor(themes_data_state_key=cfg.EDITED_THEMES_KEY)

# =================== Tab 3: Assign & Edit Results =========================
with assign_edit_tab:
    st.header("3. Assign Themes & Edit Results")
    themes_ready = isinstance(st.session_state.get(cfg.EDITED_THEMES_KEY), list) and bool(st.session_state.get(cfg.EDITED_THEMES_KEY)); responses_ready = isinstance(st.session_state.get(cfg.RESPONSES_RAW_KEY), list) and bool(st.session_state.get(cfg.RESPONSES_RAW_KEY))
    st.subheader("Assign Themes to All Responses")
    if not themes_ready: st.warning("⚠️ Define/save themes in '✏️ Review & Refine'.")
    elif not responses_ready: st.warning("⚠️ Provide responses in '📥 Input & Generate'.")
    else:
        final_themes = st.session_state[cfg.EDITED_THEMES_KEY]; all_responses = st.session_state[cfg.RESPONSES_RAW_KEY]; num_resp = len(all_responses); num_themes = len(final_themes); batch_size = st.session_state.get(cfg.BATCH_SIZE_KEY, 15); api_key = st.session_state.get(cfg.API_KEY_STATE_KEY); question = st.session_state.get(cfg.SURVEY_QUESTION_KEY, "(?)")
        st.info(f"Ready to assign **{num_themes}** themes to **{num_resp}** responses (batch: {batch_size}).")
        disable_assign = not final_themes
        if st.button(f"🏷️ Assign Themes to {num_resp} Responses", key="themer_assign_button", type="primary", disabled=disable_assign):
            logging.info(f"Themer: Starting assignment ({num_resp} responses, batch {batch_size})."); st.session_state[cfg.ASSIGNMENT_DF_KEY] = None; all_assignments = []
            progress_bar = st.progress(0, text="Initializing..."); batches = utils.batch_responses(all_responses, batch_size); total_batches = len(batches); start_time = time.time(); errors = 0
            with st.spinner(f"AI assigning themes... ({total_batches} batches)"):
                for i, batch in enumerate(batches):
                    if not batch: continue
                    progress = min(1.0, (i + 1) / total_batches); elapsed = time.time() - start_time; est_rem = (elapsed / (i + 1)) * (total_batches - (i + 1)) if (i + 1) > 0 else 0
                    prog_text = f"Assigning: Batch {i+1}/{total_batches} ({i*batch_size+len(batch)}/{num_resp}) | Est: {est_rem:.0f}s"; progress_bar.progress(progress, text=prog_text); logging.debug(f"Themer: Assign batch {i+1}")
                    batch_res = ai_core.assign_themes_with_llm_batch(question, batch, final_themes, api_key, theming_gen_config)
                    if isinstance(batch_res, list) and len(batch_res) == len(batch):
                        all_assignments.extend(batch_res); err_in_batch = sum(1 for r in batch_res if isinstance(r, dict) and "Error" in r.get('assigned_theme', ''));
                        if err_in_batch > 0: errors += err_in_batch; logging.warning(f"Themer: Batch {i+1} had {err_in_batch} errors.")
                    else: st.error(f"Error processing batch {i+1}. Add placeholders."); logging.error(f"Themer: Assign failed batch {i+1}."); placeholders = [{'assigned_theme': 'Error: Batch Failed', 'assigned_sub_theme': 'N/A', 'assignment_confidence': 'Low'}] * len(batch); all_assignments.extend(placeholders); errors += len(batch)
            total_time = time.time() - start_time; progress_bar.progress(1.0, text=f"Assignment complete! Time: {total_time:.1f}s"); time.sleep(1); progress_bar.empty()
            if len(all_assignments) == num_resp:
                try:
                    first = next((r for r in all_assignments if isinstance(r, dict) and "Error" not in r.get('assigned_theme','')), None); cols = list(first.keys()) if first else ['assigned_theme', 'assigned_sub_theme', 'assignment_confidence']
                    res_df = pd.DataFrame(all_assignments, columns=cols); res_df.insert(0, 'response', all_responses)
                    st.session_state[cfg.ASSIGNMENT_DF_KEY] = res_df; logging.info(f"Themer: Assign done. Shape: {res_df.shape}. Errors: {errors}")
                    if errors > 0: st.warning(f"✅ Assignment done, but {errors} errors occurred.")
                    else: st.success("✅ Theme assignment complete!")
                    st.rerun()
                except Exception as e: st.error(f"Failed DF creation: {e}"); logging.exception("Themer: Assign DF error."); st.session_state.pop(cfg.ASSIGNMENT_DF_KEY, None)
            else: st.error(f"❌ Assign fail: Count mismatch ({len(all_assignments)} vs {num_resp})."); logging.error("Themer: Assign length mismatch."); st.session_state.pop(cfg.ASSIGNMENT_DF_KEY, None)
    st.markdown("---"); st.subheader("View and Manually Edit Assigned Themes")
    ui_components.display_assignment_results_editable(df_state_key=cfg.ASSIGNMENT_DF_KEY, themes_state_key=cfg.EDITED_THEMES_KEY)

# ===================== Tab 4: Explore & Visualize =========================
with viz_tab:
    st.header("📊 Explore & Visualize Themes")
    assign_df = st.session_state.get(cfg.ASSIGNMENT_DF_KEY); themes_struct = st.session_state.get(cfg.EDITED_THEMES_KEY)
    assign_df_exists = isinstance(assign_df, pd.DataFrame) and not assign_df.empty; themes_struct_exists = isinstance(themes_struct, list) and bool(themes_struct)
    if not assign_df_exists: st.info("Assign themes in '🏷️ Assign & Edit Results' tab first.")
    else:
        st.subheader("Explore Themes and Examples")
        if themes_struct_exists: ui_components.display_theme_examples(df_state_key=cfg.ASSIGNMENT_DF_KEY, themes_state_key=cfg.EDITED_THEMES_KEY)
        else: st.warning("Saved theme structure not found. Cannot display descriptions."); ui_components.display_theme_examples(df_state_key=cfg.ASSIGNMENT_DF_KEY, themes_state_key=cfg.EDITED_THEMES_KEY) # Still call it
        st.markdown("---"); st.subheader("Visualize Distributions")
        ui_components.display_theme_distribution(df_state_key=cfg.ASSIGNMENT_DF_KEY)
        st.markdown("---"); st.subheader("☁️ Word Cloud per Theme")
        try:
            valid_wc_themes = sorted([str(th) for th in assign_df['assigned_theme'].unique() if pd.notna(th) and isinstance(th, str) and "Error" not in th and th != "Uncategorized"])
            if not valid_wc_themes: st.info("No valid themes for word cloud.")
            else:
                 sel_wc_theme = st.selectbox("Select Theme for Word Cloud:", valid_wc_themes, key="themer_wc_theme_select")
                 if sel_wc_theme:
                     if 'assigned_theme' in assign_df.columns and 'response' in assign_df.columns: wc_responses = assign_df.loc[assign_df['assigned_theme'] == sel_wc_theme, 'response'].tolist()
                     else: st.error("Required columns missing."); wc_responses = []
                     if wc_responses:
                         with st.spinner(f"Generating Word Cloud for '{sel_wc_theme}'..."):
                             fig = ui_components.create_word_cloud(wc_responses)
                             if fig: st.pyplot(fig, use_container_width=True)
                     else: st.info(f"No responses found for theme '{sel_wc_theme}'.")
        except Exception as e: st.error(f"Word cloud error: {e}"); logging.exception("Themer: Word cloud UI error.")

# ========================== Tab 5: Ask AI =================================
with ai_qa_tab:
    st.header("❓ Ask AI About the Raw Data")

    responses_avail_qa = isinstance(st.session_state.get(cfg.RESPONSES_RAW_KEY), list) and bool(st.session_state.get(cfg.RESPONSES_RAW_KEY))
    question_avail_qa = isinstance(st.session_state.get(cfg.SURVEY_QUESTION_KEY), str) and bool(st.session_state.get(cfg.SURVEY_QUESTION_KEY, '').strip())

    if not responses_avail_qa or not question_avail_qa:
        st.info("Provide survey question and responses in the '📥 Input & Generate' tab first to enable Q&A.")
    else:
        question_qa = st.session_state[cfg.SURVEY_QUESTION_KEY]
        responses_list_qa = st.session_state[cfg.RESPONSES_RAW_KEY]
        api_key = st.session_state.get(cfg.API_KEY_STATE_KEY)

        st.markdown("Ask a question about the **original raw responses**. The AI will use these responses and the survey question you provided as context to answer.")
        st.info(f"Context: Survey Question \"_{question_qa}_\" and {len(responses_list_qa)} raw responses.")

        # Q&A Input Area
        user_ai_question = st.text_area(
            "Your Question:", key="ai_qa_input_main_widget", height=100, # Unique key
            placeholder="e.g., What are the main recurring ideas in the feedback?\nList responses mentioning 'price' or 'cost'.\nSummarize the positive comments.\nWhich comments seem confused?"
        )

        # Ask Button
        if st.button("Ask AI", key="ask_ai_button_main_widget", type="primary"): # Unique key
            user_question_stripped = user_ai_question # Read from widget state implicitly
            if not user_question_stripped.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("🧠 AI is thinking..."):
                    if not api_key:
                         st.error("API Key not found. Please initialize in the sidebar.")
                         ai_answer = "Error: API Key missing."
                    else:
                         ai_answer = ai_core.ask_ai_about_data( # Call CORRECTED Q&A function
                             survey_question=question_qa,
                             responses_list=responses_list_qa,
                             user_question=user_question_stripped.strip(), # Pass stripped question
                             generation_config=theming_gen_config, # Use themer's base config
                             api_key=api_key
                         )

                # Add to history
                if ai_answer is not None:
                     if not isinstance(st.session_state.get(cfg.AI_QA_HISTORY_KEY), list): st.session_state[cfg.AI_QA_HISTORY_KEY] = []
                     st.session_state[cfg.AI_QA_HISTORY_KEY].insert(0, {"question": user_question_stripped.strip(), "answer": ai_answer})
                     logging.info(f"Themer Q&A: Q='{user_question_stripped[:50]}...', A_len={len(str(ai_answer))}")
                     # Line to clear input removed
                     st.rerun()
                else: logging.error("Themer Q&A: ask_ai_about_data returned None.")


        # Display Q&A History
        st.markdown("---"); st.subheader("Q&A History")
        qa_history_list = st.session_state.get(cfg.AI_QA_HISTORY_KEY, [])
        if qa_history_list:
             history_limit = 5
             for i, qa_pair in enumerate(qa_history_list):
                 q_text = str(qa_pair.get('question', '(...)')); a_text = str(qa_pair.get('answer', '...'))
                 with st.expander(f"Q: {q_text}", expanded=(i < history_limit)):
                     st.markdown("**AI Answer:**"); st.markdown(a_text, unsafe_allow_html=True)
        else: st.caption("No questions asked yet in this session.")


# --- Footer ---
st.markdown(f"""<div class="footer"><p>© {time.strftime('%Y')} Phronesis Partners. All rights reserved.</p></div>""", unsafe_allow_html=True)

# --- End of App ---
logging.info("Reached end of themer main_app.py execution.")
