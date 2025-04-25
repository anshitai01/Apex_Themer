# ui_components.py (for AI Themer App)
"""
Functions for creating and displaying UI components in the Streamlit app.
Includes visualizations (word cloud, charts), editors (themes, assignments),
and data exploration elements. Includes TEMPORARY DEBUGGING for data editor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import logging
import time
import json
import random
import io # Needed for df.info() buffer in debug

# Import specific config keys and constants from Themer's config
from config import (
    EDITED_THEMES_KEY, ASSIGNMENT_DF_KEY,
    UPLOADED_THEME_FILE_WIDGET_KEY, ASSIGNMENT_EDITOR_WIDGET_KEY,
    # Assuming Phronesis theme colors are now in Themer's config.py:
    PRIMARY_ACCENT_COLOR, CHART_SUCCESS_COLOR, CHART_WARNING_COLOR, CHART_ERROR_COLOR,
    MAIN_BACKGROUND_COLOR, BODY_TEXT_COLOR, CARD_BACKGROUND_COLOR, INPUT_BORDER_COLOR,
    MAIN_TITLE_COLOR, SUBTITLE_COLOR
)


# --- Visualization Functions ---

# Word Cloud (Adapted for Dark Theme)
def create_word_cloud(responses):
    """
    Generates a matplotlib figure containing a word cloud from text responses.
    Adapted for dark theme (transparent background).
    """
    if not responses:
        logging.info("Word cloud generation skipped: No responses provided.")
        return None
    try:
        text_list = [str(r).strip() for r in responses if r and isinstance(r, (str, int, float)) and str(r).strip()]
        if not text_list:
            logging.info("Word cloud generation skipped: No valid text content after cleaning.")
            return None
        text = ' '.join(text_list)
        if not text.strip():
            logging.info("Word cloud generation skipped: Joined text is empty.")
            return None

        wordcloud = WordCloud(
            width=800, height=350,
            background_color=None, # Use None for transparent background
            mode="RGBA",           # Ensure RGBA mode for transparency
            colormap='plasma',     # Choose a colormap suitable for dark backgrounds
            max_words=100,
            random_state=42
            ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0) # Make figure background transparent
        ax.patch.set_alpha(0.0)  # Make axes background transparent
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        logging.info("Word cloud figure generated successfully (dark theme).")
        return fig
    except ValueError as ve:
         if "empty vocabulary" in str(ve).lower():
             logging.warning(f"Word cloud generation failed: Empty vocabulary. {ve}")
             st.caption("Could not generate word cloud: No significant words found.")
         else:
             st.error(f"Error generating word cloud (ValueError): {ve}")
             logging.error(f"Word cloud ValueError: {ve}")
         return None
    except ImportError as ie:
        st.error(f"Error generating word cloud: Missing library ({ie}).")
        logging.error(f"Word cloud ImportError: {ie}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred generating the word cloud: {e}")
        logging.exception("Word cloud generation failed.")
        return None


# Theme Distribution Charts (Adapted for Dark Theme)
def display_theme_distribution(df_state_key):
    """
    Shows bar charts for theme, sub-theme, and confidence distribution.
    Adapted for dark theme.
    """
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame):
        st.info("Assign themes first for charts.")
        return
    df = st.session_state[df_state_key]
    if df.empty or 'assigned_theme' not in df.columns:
        st.info("No assignment data for charts.")
        return
    try:
        df['assigned_theme'] = df['assigned_theme'].astype(str)
        df['assigned_sub_theme'] = df['assigned_sub_theme'].astype(str)
        if 'assignment_confidence' in df.columns:
            df['assignment_confidence'] = df['assignment_confidence'].astype(str)
        else:
            logging.warning("Confidence column missing.")
    except Exception as e:
        st.error(f"Chart data prep error: {e}")
        logging.error(f"Chart column conversion error: {e}")
        return
    df_valid = df[~df['assigned_theme'].str.contains("Error", case=False, na=False)].copy()
    if df_valid.empty:
        st.info("No valid assignments for charts.")
        return

    try:
        col1, col2 = st.columns(2)
        plotly_template = "plotly_dark"

        # --- Main themes ---
        with col1:
            st.subheader("Main Theme Distribution")
            theme_counts = df_valid['assigned_theme'].value_counts().reset_index()
            theme_counts.columns = ['Main Theme', 'Count']
            if not theme_counts.empty:
                fig_main = px.bar(theme_counts, x='Main Theme', y='Count', title="Distribution of Main Themes", text_auto=True, color_discrete_sequence=px.colors.qualitative.Pastel, template=plotly_template)
                fig_main.update_layout(xaxis_title=None, yaxis_title="Responses", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_main, use_container_width=True)
            else:
                st.info("No main themes assigned.")

        # --- Confidence Distribution ---
        with col2:
            st.subheader("Confidence Distribution")
            if 'assignment_confidence' in df_valid.columns:
                 conf_counts = df_valid['assignment_confidence'].value_counts().reset_index()
                 conf_counts.columns = ['Confidence', 'Count']
                 conf_order = ["High", "Medium", "Low"]
                 color_map = {'High': CHART_SUCCESS_COLOR, 'Medium': CHART_WARNING_COLOR, 'Low': CHART_ERROR_COLOR}
                 try:
                     conf_counts['Confidence'] = pd.Categorical(conf_counts['Confidence'], categories=conf_order, ordered=True)
                     conf_counts = conf_counts.sort_values('Confidence')
                 except Exception as e:
                     st.warning(f"Could not sort confidence: {e}")
                     logging.warning(f"Confidence sort error: {e}")
                 if not conf_counts.empty:
                      fig_conf = px.bar(conf_counts, x='Confidence', y='Count', title="Distribution of Assignment Confidence", text_auto=True, color='Confidence', color_discrete_map=color_map, template=plotly_template)
                      fig_conf.update_layout(xaxis_title="Level", yaxis_title="Responses", showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                      st.plotly_chart(fig_conf, use_container_width=True)
                 else:
                      st.info("No confidence data.")
            else:
                 st.info("Confidence column not found.")

        # --- Sub-themes (overall) ---
        st.subheader("Overall Sub-theme Distribution")
        excluded_subs = ['N/A', 'Error', 'General', 'Other', '', 'nan', 'None']
        sub_counts = df_valid[~df_valid['assigned_sub_theme'].astype(str).isin(excluded_subs) & (df_valid['assigned_theme'] != "Uncategorized")]['assigned_sub_theme'].value_counts().reset_index()
        sub_counts.columns = ['Sub-theme', 'Count']
        if not sub_counts.empty:
            max_subs = 25
            if len(sub_counts) > max_subs: # Corrected multi-line if
                sub_counts = sub_counts.head(max_subs)
                st.caption(f"Showing top {max_subs} specific sub-themes.")
            fig_sub = px.bar(sub_counts, x='Sub-theme', y='Count', title="Distribution of Top Specific Sub-themes", color_discrete_sequence=px.colors.qualitative.Set2, template=plotly_template)
            fig_sub.update_layout(xaxis_title=None, yaxis_title="Responses", xaxis_tickangle=-45, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
            st.info("No specific sub-themes found for distribution.")
    except Exception as e:
        st.error(f"Chart error: {e}")
        logging.exception("Failed generating distribution plots.")


# --- Editor and Data Display Functions ---

# Theme Editor
def display_theme_editor(themes_data_state_key=EDITED_THEMES_KEY):
    """
    Renders an editable UI for themes/sub-themes with Add/Delete/Save functionality.
    Modifies the theme structure list directly in Streamlit's session state.
    Includes loading/saving theme structure from/to JSON.
    """
    # ... (Keep the full function code from previous versions - omitted for brevity) ...
    if themes_data_state_key not in st.session_state or not isinstance(st.session_state.get(themes_data_state_key), list): st.session_state[themes_data_state_key] = []; logging.info(f"Init state '{themes_data_state_key}' for editor.")
    themes_data = st.session_state[themes_data_state_key]
    if not themes_data: st.info("No themes generated/loaded. Add below or generate on first tab.")
    def _del_theme_cb(idx): current = st.session_state.get(themes_data_state_key, []); del current[idx]; st.session_state[themes_data_state_key] = current; st.toast(f"Theme {idx+1} removed."); st.rerun()
    def _del_sub_cb(t_idx, s_idx): current = st.session_state.get(themes_data_state_key, []); item = current[t_idx]; subs = item.get('sub_themes', []); del subs[s_idx]; item['sub_themes'] = subs; st.session_state[themes_data_state_key] = current; st.toast(f"Sub-theme removed from Theme {t_idx+1}."); st.rerun()
    def _add_sub_cb(t_idx): current = st.session_state.get(themes_data_state_key, []); item = current[t_idx]; item.setdefault('sub_themes', []).append(""); st.session_state[themes_data_state_key] = current; st.toast(f"New sub-theme added to Theme {t_idx+1}."); st.rerun()
    def _add_theme_cb(): current = st.session_state.get(themes_data_state_key, []); name = f'New Theme {len(current)+1}'; current.append({'theme': name, 'sub_themes': ['New Sub-theme 1'], 'description': ''}); st.session_state[themes_data_state_key] = current; st.toast(f"'{name}' added."); st.rerun()
    num_themes_render = len(st.session_state.get(themes_data_state_key, []))
    for i in range(num_themes_render):
        current_list = st.session_state.get(themes_data_state_key, [])
        if i >= len(current_list): logging.warning(f"Theme idx {i} out of bounds."); break
        item = current_list[i];
        if not isinstance(item, dict): logging.warning(f"Item {i} not dict."); continue
        t_key = f"theme_edit_{themes_data_state_key}_{i}"; s_base = f"sub_edit_{themes_data_state_key}_{i}"; del_t = f"del_theme_{themes_data_state_key}_{i}"; add_s = f"add_sub_{themes_data_state_key}_{i}"
        with st.container():
            st.markdown("---", unsafe_allow_html=True); cols_h = st.columns([0.8, 0.2])
            orig_label = item.get('theme', f'Theme {i+1}'); orig_subs = item.get('sub_themes', []); orig_desc = item.get('description', '')
            with cols_h[0]: st.text_input(f"**Theme {i+1}**", value=orig_label, key=t_key); st.caption(f"Desc: {orig_desc or '(N/A)'}") # Use key for state management
            with cols_h[1]: st.markdown("<div style='margin-top:28px;'></div>",True); st.button("🗑️", key=del_t, help=f"Del Theme {i+1}", on_click=_del_theme_cb, args=(i,))
            st.write("**Sub-themes:**"); sub_cols = st.columns(4)
            if not isinstance(orig_subs, list): orig_subs = []
            num_subs = len(orig_subs)
            for j in range(num_subs):
                 sub_val = orig_subs[j]; s_key = f"{s_base}_{j}"; del_s = f"del_sub_{themes_data_state_key}_{i}_{j}"; col_idx = j % len(sub_cols)
                 with sub_cols[col_idx]: st.text_input(f"Sub {j+1}", value=sub_val, key=s_key, label_visibility="collapsed"); st.button("✖", key=del_s, help=f"Del sub {j+1}", on_click=_del_sub_cb, args=(i,j))
            st.button("➕ Add Sub-theme", key=add_s, on_click=_add_sub_cb, args=(i,))
    st.markdown("---"); st.button("➕ Add New Theme", key=f"add_theme_btn_{themes_data_state_key}", on_click=_add_theme_cb)
    st.markdown("---")
    if st.button("💾 Save ALL Refined Themes", key=f"save_themes_btn_{themes_data_state_key}", type="primary"):
        updated_list = []; num_themes_state = len(st.session_state.get(themes_data_state_key, [])); all_ok = True
        for i in range(num_themes_state): # Iterate based on state length *before* potential deletes
            t_key = f"theme_edit_{themes_data_state_key}_{i}"; s_base = f"sub_edit_{themes_data_state_key}_{i}"
            if t_key in st.session_state: # Check if widget exists
                theme_label = st.session_state[t_key].strip() or f"Theme {i+1}" # Read value using key
                orig_desc = ""; current_list_save = st.session_state.get(themes_data_state_key, [])
                if i < len(current_list_save) and isinstance(current_list_save[i], dict): orig_desc = current_list_save[i].get('description', '')
                saved_subs = []; num_subs_exp = 0
                if i < len(current_list_save) and isinstance(current_list_save[i], dict): sub_list_state = current_list_save[i].get('sub_themes', []); num_subs_exp = len(sub_list_state) if isinstance(sub_list_state, list) else 0
                for j in range(num_subs_exp):
                     s_key = f"{s_base}_{j}"
                     if s_key in st.session_state: sub_label = st.session_state[s_key].strip(); saved_subs.append(sub_label) if sub_label else None # Read value using key
                     else: logging.warning(f"Sub key {s_key} miss during save."); all_ok = False
                updated_list.append({'theme': theme_label, 'sub_themes': [s for s in saved_subs if s], 'description': orig_desc}) # Ensure only non-empty saved
            else: logging.warning(f"Theme key {t_key} miss during save."); all_ok = False
        st.session_state[themes_data_state_key] = updated_list # Update state with collected values
        if all_ok: st.success("Themes saved."); st.toast("Themes updated!", icon="✏️")
        else: st.warning("Themes saved, but review structure below (some fields might be lost)."); st.toast("Themes updated (check review).", icon="⚠️")
        st.rerun()
    st.markdown("---"); st.subheader("Current Saved Structure"); current_saved = st.session_state.get(themes_data_state_key, [])
    if current_saved:
         try: json_str = json.dumps(current_saved, indent=2); st.download_button(label="💾 Download Structure (JSON)", data=json_str, file_name=f"themer_structure_{time.strftime('%Y%m%d')}.json", mime="application/json", key=f"dl_themes_{themes_data_state_key}")
         except Exception as e: st.error(f"Download error: {e}")
         for idx, item in enumerate(current_saved):
             label = item.get('theme', f'Theme {idx+1}'); desc = item.get('description', 'N/A'); subs = item.get('sub_themes', [])
             with st.expander(f"Theme {idx+1}: {label}", expanded=False): st.caption(f"Desc: {desc}"); st.write("**Sub-themes:**"); st.code(json.dumps(subs, indent=2), language='json') if subs else st.caption("(None)")
    else: st.info("No themes saved.")
    st.markdown("---"); st.subheader("Load Structure")
    up_file = st.file_uploader("⬆️ Upload Structure (JSON)", type=['json'], key=UPLOADED_THEME_FILE_WIDGET_KEY) # Use config key
    if up_file:
        try:
            loaded = json.load(up_file); valid = isinstance(loaded, list); processed = []
            if valid:
                for idx, item in enumerate(loaded):
                    if not (isinstance(item, dict) and 'theme' in item and 'sub_themes' in item): st.error(f"Invalid item {idx}: Missing keys."); valid = False; break
                    if not isinstance(item.get('theme'), str) or not isinstance(item.get('sub_themes'), list): st.error(f"Invalid item {idx}: Wrong types."); valid = False; break
                    proc = {}; proc['theme'] = str(item.get('theme', '')).strip(); proc['description'] = str(item.get('description', '')).strip(); proc['sub_themes'] = [str(s).strip() for s in item.get('sub_themes', []) if str(s).strip()]
                    if not proc['theme']: st.error(f"Invalid item {idx}: Empty theme."); valid = False; break
                    processed.append(proc)
            if valid and processed: st.session_state[themes_data_state_key] = processed; st.success(f"Loaded {len(processed)} themes."); st.toast("Themes loaded!", icon="⬆️"); st.rerun()
            elif not valid: logging.warning(f"Invalid theme structure uploaded: {up_file.name}")
            else: st.warning("Uploaded empty list."); st.session_state[themes_data_state_key] = []; st.rerun()
        except json.JSONDecodeError: st.error("Failed JSON parse."); logging.error(f"JSONDecodeError loading theme: {up_file.name}")
        except Exception as e: st.error(f"Load error: {e}"); logging.exception(f"Error loading theme structure: {up_file.name}")


# --- Assignment Results Editor (Includes DEBUGGING) ---
def display_assignment_results_editable(df_state_key=ASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    """
    Displays the theme assignment results in an editable st.data_editor.
    Includes robust checks and TEMPORARY DEBUGGING output.
    """
    # --- Get Data and Validate ---
    if df_state_key not in st.session_state:
        st.info("No assignment results available. Run assignment first.")
        return

    results_df_from_state = st.session_state.get(df_state_key)

    if not isinstance(results_df_from_state, pd.DataFrame):
        logging.error(f"State key {df_state_key} is not DataFrame (Type: {type(results_df_from_state)}). Clearing state.")
        st.warning("Cleared invalid assignment data. Please run assignment again.")
        st.session_state.pop(df_state_key, None)
        st.rerun()
        return
    if results_df_from_state.empty:
         st.info("Assignment results are empty. No data to edit.")
         return

    # If we passed checks, proceed using a copy for safety before editing/cleaning
    results_df = results_df_from_state.copy()

    st.info("Manually review/edit assignments below. Click 'Save Manual Changes' to update.")

    # --- DataFrame Preparation (Clean the COPY before config/editor) ---
    required_cols = ["response", "assigned_theme", "assigned_sub_theme", "assignment_confidence"]
    if not all(col in results_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        st.error(f"Assignment DF missing required columns: {', '.join(missing_cols)}. Cannot display editor.")
        logging.error(f"Themer Assignment DF missing cols: {missing_cols}")
        return

    try:
        # Clean the DataFrame copy
        results_df['assigned_theme'] = results_df['assigned_theme'].fillna("Uncategorized").astype(str)
        results_df['assigned_sub_theme'] = results_df['assigned_sub_theme'].fillna("N/A").astype(str)
        results_df['assignment_confidence'] = results_df['assignment_confidence'].fillna("Low").astype(str)
        results_df['response'] = results_df['response'].fillna("").astype(str)
        results_df.loc[results_df['assigned_theme'] == "Uncategorized", 'assigned_sub_theme'] = "N/A"
    except Exception as e:
        st.error(f"Failed preparing data for editor: {e}")
        logging.error(f"Themer: Error cleaning state DF for data editor: {e}")
        return

    # --- Prepare Theme Options (Based on cleaned DataFrame and structure) ---
    theme_structure = st.session_state.get(themes_state_key, [])
    defined_theme_labels = set()
    if isinstance(theme_structure, list) and theme_structure:
        defined_theme_labels = set(str(t.get('theme', '')).strip() for t in theme_structure if t.get('theme') and str(t.get('theme','')).strip())
    else: st.warning("Theme structure missing/invalid. Theme dropdown options may be limited.")
    try: themes_in_data = set(results_df['assigned_theme'].unique()) # Use cleaned data
    except Exception as e: st.error(f"Error getting themes from DF: {e}"); themes_in_data = set()
    all_theme_options_set = defined_theme_labels | themes_in_data | {"Uncategorized"}
    theme_options = sorted([opt for opt in all_theme_options_set if pd.notna(opt) and str(opt).strip()])
    if not theme_options: theme_options = ["Uncategorized"]

    # Ensure themes in the dataframe are actually in the options list
    results_df['assigned_theme'] = results_df['assigned_theme'].apply(lambda x: x if x in theme_options else "Uncategorized")
    # Re-apply Uncategorized rule after potential reset above
    results_df.loc[results_df['assigned_theme'] == "Uncategorized", 'assigned_sub_theme'] = "N/A"


    # --- Define Column Configurations ---
    column_config = {
        "response": st.column_config.TextColumn("Response", width="large", disabled=True),
        "assigned_theme": st.column_config.SelectboxColumn("Assigned Theme", width="medium", options=theme_options, required=True),
        "assigned_sub_theme": st.column_config.TextColumn("Assigned Sub-theme", width="medium", required=True),
        "assignment_confidence": st.column_config.SelectboxColumn("Confidence", width="small", options=["High", "Medium", "Low"], required=True),
    }

    # --- >>> ADD TEMP DEBUGGING <<< ---
    with st.expander("DEBUG INFO (Data Editor Input)", expanded=False):
        st.write("**DataFrame Info passed to editor:**")
        try:
            # Create a string buffer to capture df.info() output
            buffer = io.StringIO()
            results_df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s) # Display df info including types and non-null counts
            st.write("**First 2 rows passed to editor:**")
            st.dataframe(results_df.head(2)) # Show first few rows of the cleaned df
        except Exception as e:
            st.error(f"Error displaying DataFrame info: {e}")
        st.write("**Column Config:**")
        # Convert config values to strings for JSON serialization if they aren't primitives
        serializable_config = {}
        for k, v in column_config.items():
            try: # Attempt to represent the config object, may not be perfect
                serializable_config[k] = str(v)
            except:
                 serializable_config[k] = f"Cannot serialize config object for {k}"
        st.json(serializable_config, expanded=False) # Display the config dict
        st.write(f"**Editor Key:** `{ASSIGNMENT_EDITOR_WIDGET_KEY}`")
    # --- >>> END TEMP DEBUGGING <<< ---


    # --- Display Data Editor ---
    st.markdown("#### Edit Assignments")
    edited_df = None # Initialize edited_df to None
    try:
        edited_df = st.data_editor(
            results_df, # Pass the cleaned DataFrame
            key=ASSIGNMENT_EDITOR_WIDGET_KEY,
            use_container_width=True, column_config=column_config,
            column_order=required_cols, num_rows="fixed", hide_index=True,
            disabled=["response"]
        )
    except Exception as editor_err:
         # Catch errors during the editor call itself
         st.error(f"Data editor failed to render: {editor_err}. See logs for details. Try refreshing.")
         logging.exception("Themer: st.data_editor failed to render.")
         # Optionally display the debug info again here if it failed during render
         return # Stop function execution if editor fails to render

    # --- Save Changes Button ---
    # Check if edited_df was successfully created by the editor
    if edited_df is not None and isinstance(edited_df, pd.DataFrame):
        if st.button("💾 Save Manual Assignment Changes", key=f"save_manual_assignments_btn_{df_state_key}"):

            # --- Validation Logic ---
            valid = True; validation_warnings = []
            for index, row in edited_df.iterrows():
                theme = row['assigned_theme']; sub_theme = row['assigned_sub_theme']
                if not theme or not sub_theme or not row['assignment_confidence']: valid=False; validation_warnings.append(f"Row {index+1}: Required fields empty.")
                if theme == "Uncategorized" and sub_theme != "N/A": validation_warnings.append(f"Row {index+1}: Uncategorized needs 'N/A' sub."); valid = False
                # Add other validation if needed

            if valid:
                try:
                    # Ensure index consistency before saving back to state
                    edited_df_copy = edited_df.copy() # Work with a copy for saving
                    original_index = st.session_state[df_state_key].index
                    if len(original_index) == len(edited_df_copy):
                        edited_df_copy.index = original_index
                        st.session_state[df_state_key] = edited_df_copy # Save the copy
                        st.success("Manual assignment changes saved!"); st.toast("Assignments updated!", icon="💾")
                        logging.info(f"Themer: Manual assignments saved '{df_state_key}'.")
                        time.sleep(0.5); st.rerun()
                    else:
                         st.error("Error saving: Row count mismatch after editing. Cannot save.")
                         logging.error(f"Themer: Row count mismatch during save. Original: {len(original_index)}, Edited: {len(edited_df_copy)}")

                except Exception as save_err:
                    st.error(f"Error saving changes: {save_err}")
                    logging.error(f"Themer: Error saving edited DF: {save_err}")
            else:
                st.error("Validation Failed! Correct issues below before saving:");
                for i, w in enumerate(validation_warnings): st.warning(w) if i<5 else st.warning(f"...{len(validation_warnings)-5} more issues."); break
                logging.warning(f"Themer: Manual assignment save validation failed ({len(validation_warnings)} issues).")
    elif edited_df is None:
         # This case might be hit if the editor failed to render initially
         st.warning("Data editor did not render correctly. Cannot save changes.")


# Theme Examples Display
def display_theme_examples(df_state_key=ASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    # ... (Existing code - omitted for brevity) ...
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame): st.info("Assign themes first."); return
    results_df = st.session_state[df_state_key]
    if results_df.empty or 'assigned_theme' not in results_df.columns: st.info("No assignment data."); return
    theme_structure = st.session_state.get(themes_state_key, []); theme_map = {}
    if isinstance(theme_structure, list) and theme_structure: theme_map = {str(t.get('theme','')): t for t in theme_structure if t.get('theme')}
    else: st.caption("Theme structure missing.")
    try: valid_themes_in_data = sorted([str(th) for th in results_df['assigned_theme'].unique() if pd.notna(th) and isinstance(th, str) and "Error" not in th])
    except Exception as e: st.error(f"Error processing themes: {e}"); logging.error(f"Error getting unique themes: {e}"); valid_themes_in_data = []
    if not valid_themes_in_data: st.warning("No valid themes in results."); return
    selected_theme = st.selectbox("Select Theme to Explore:", valid_themes_in_data, key=f"explore_theme_select_{df_state_key}")
    if selected_theme:
        st.markdown(f"#### Exploring Theme: **{selected_theme}**"); theme_details = theme_map.get(selected_theme)
        if theme_details:
            st.markdown("**Description:**"); description = theme_details.get('description', '').strip(); st.markdown(f"> {description}" if description else "_(N/A)_")
            st.markdown("**Defined Sub-themes:**"); sub_themes = theme_details.get('sub_themes', [])
            if sub_themes and isinstance(sub_themes, list): subs_str = [str(s) for s in sub_themes]; st.code(f"{', '.join(subs_str)}" if subs_str else "(None)", language=None)
            else: st.caption("(None)")
        elif selected_theme != "Uncategorized": st.caption(f"Details not found for '{selected_theme}'.")
        st.markdown("**Example Responses (Sample):**")
        try:
            if 'assigned_theme' in results_df.columns: theme_responses_df = results_df[results_df['assigned_theme'] == selected_theme].copy()
            else: st.error("'assigned_theme' column missing."); return
            if not theme_responses_df.empty:
                 num_examples = min(5, len(theme_responses_df)); examples = theme_responses_df.sample(n=num_examples, random_state=42)
                 for index, row in examples.iterrows():
                     st.markdown("---"); response = str(row.get('response', 'N/A')); sub = str(row.get('assigned_sub_theme', 'N/A')); conf = str(row.get('assignment_confidence', 'N/A'))
                     st.markdown(f"> {response}"); st.caption(f"Sub-theme: `{sub}` | Confidence: `{conf}`")
            else: st.info(f"No responses found for '{selected_theme}'.")
        except KeyError as ke: st.error(f"Missing column: {ke}"); logging.error(f"KeyError examples: {ke}")
        except Exception as e: st.error(f"Error getting examples: {e}"); logging.exception(f"Error displaying examples for '{selected_theme}'.")
