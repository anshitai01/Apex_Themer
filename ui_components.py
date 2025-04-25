# ui_components.py (for AI Themer App)
"""
Functions for creating and displaying UI components in the Streamlit app.
Includes visualizations (word cloud, charts), editors (themes, assignments),
and data exploration elements.
Includes fix for data editor state handling.
"""

import streamlit as st
import pandas as pd
import numpy as np # Import numpy
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import logging
import time
import json
import random

# Import specific config keys and constants from Themer's config
from config import (
    EDITED_THEMES_KEY, ASSIGNMENT_DF_KEY,
    UPLOADED_THEME_FILE_WIDGET_KEY, ASSIGNMENT_EDITOR_WIDGET_KEY,
    # Assuming Phronesis theme colors are now in Themer's config.py:
    PRIMARY_ACCENT_COLOR, CHART_SUCCESS_COLOR, CHART_WARNING_COLOR, CHART_ERROR_COLOR,
    MAIN_BACKGROUND_COLOR, BODY_TEXT_COLOR, CARD_BACKGROUND_COLOR, INPUT_BORDER_COLOR,
    MAIN_TITLE_COLOR, SUBTITLE_COLOR # Add others needed by charts/components
)
# Import COLOR_SCHEME if still used elsewhere, otherwise remove if fully replaced
# from config import COLOR_SCHEME


# --- Visualization Functions ---
# (Keep create_word_cloud and display_theme_distribution functions as they were)
# Word Cloud (Adapted for Dark Theme)
def create_word_cloud(responses):
    # ... (existing code) ...
    if not responses: logging.info("Word cloud skipped: No responses."); return None
    try:
        text_list = [str(r).strip() for r in responses if r and isinstance(r, (str, int, float)) and str(r).strip()]
        if not text_list: logging.info("Word cloud skipped: No valid text."); return None
        text = ' '.join(text_list)
        if not text.strip(): logging.info("Word cloud skipped: Empty text."); return None
        wordcloud = WordCloud(width=800, height=350, background_color=None, mode="RGBA", colormap='plasma', max_words=100, random_state=42).generate(text)
        fig, ax = plt.subplots(figsize=(10, 4)); fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off'); logging.info("Word cloud generated (dark theme).")
        return fig
    except ValueError as ve:
         if "empty vocabulary" in str(ve).lower(): logging.warning(f"Word cloud failed: Empty vocabulary. {ve}"); st.caption("Word cloud: No significant words found.")
         else: st.error(f"Word cloud ValueError: {ve}"); logging.error(f"Word cloud ValueError: {ve}")
         return None
    except ImportError as ie: st.error(f"Word cloud ImportError: {ie}"); logging.error(f"Word cloud ImportError: {ie}"); return None
    except Exception as e: st.error(f"Word cloud unexpected error: {e}"); logging.exception("Word cloud generation failed."); return None

# Theme Distribution Charts (Adapted for Dark Theme)
def display_theme_distribution(df_state_key):
    # ... (existing code) ...
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame): st.info("Assign themes first for charts."); return
    df = st.session_state[df_state_key]
    if df.empty or 'assigned_theme' not in df.columns: st.info("No assignment data for charts."); return
    try:
        df['assigned_theme'] = df['assigned_theme'].astype(str); df['assigned_sub_theme'] = df['assigned_sub_theme'].astype(str)
        if 'assignment_confidence' in df.columns: df['assignment_confidence'] = df['assignment_confidence'].astype(str)
        else: logging.warning("Confidence column missing.")
    except Exception as e: st.error(f"Chart data prep error: {e}"); logging.error(f"Chart column conversion error: {e}"); return
    df_valid = df[~df['assigned_theme'].str.contains("Error", case=False, na=False)].copy()
    if df_valid.empty: st.info("No valid assignments for charts."); return
    try:
        col1, col2 = st.columns(2); plotly_template = "plotly_dark"
        with col1: # Main themes
            st.subheader("Main Theme Distribution"); theme_counts = df_valid['assigned_theme'].value_counts().reset_index(); theme_counts.columns = ['Main Theme', 'Count']
            if not theme_counts.empty: fig_main = px.bar(theme_counts, x='Main Theme', y='Count', title="Distribution of Main Themes", text_auto=True, color_discrete_sequence=px.colors.qualitative.Pastel, template=plotly_template); fig_main.update_layout(xaxis_title=None, yaxis_title="Responses", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig_main, use_container_width=True)
            else: st.info("No main themes assigned.")
        with col2: # Confidence
            st.subheader("Confidence Distribution")
            if 'assignment_confidence' in df_valid.columns:
                 conf_counts = df_valid['assignment_confidence'].value_counts().reset_index(); conf_counts.columns = ['Confidence', 'Count']; conf_order = ["High", "Medium", "Low"]; color_map = {'High': CHART_SUCCESS_COLOR, 'Medium': CHART_WARNING_COLOR, 'Low': CHART_ERROR_COLOR}
                 try: conf_counts['Confidence'] = pd.Categorical(conf_counts['Confidence'], categories=conf_order, ordered=True); conf_counts = conf_counts.sort_values('Confidence')
                 except Exception as e: st.warning(f"Could not sort confidence: {e}"); logging.warning(f"Confidence sort error: {e}")
                 if not conf_counts.empty: fig_conf = px.bar(conf_counts, x='Confidence', y='Count', title="Distribution of Assignment Confidence", text_auto=True, color='Confidence', color_discrete_map=color_map, template=plotly_template); fig_conf.update_layout(xaxis_title="Level", yaxis_title="Responses", showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig_conf, use_container_width=True)
                 else: st.info("No confidence data.")
            else: st.info("Confidence column not found.")
        st.subheader("Overall Sub-theme Distribution") # Sub-themes
        excluded_subs = ['N/A', 'Error', 'General', 'Other', '', 'nan', 'None']; sub_counts = df_valid[~df_valid['assigned_sub_theme'].astype(str).isin(excluded_subs) & (df_valid['assigned_theme'] != "Uncategorized")]['assigned_sub_theme'].value_counts().reset_index(); sub_counts.columns = ['Sub-theme', 'Count']
        if not sub_counts.empty:
            max_subs = 25; if len(sub_counts) > max_subs: sub_counts = sub_counts.head(max_subs); st.caption(f"Showing top {max_subs} sub-themes.")
            fig_sub = px.bar(sub_counts, x='Sub-theme', y='Count', title="Distribution of Top Specific Sub-themes", color_discrete_sequence=px.colors.qualitative.Set2, template=plotly_template); fig_sub.update_layout(xaxis_title=None, yaxis_title="Responses", xaxis_tickangle=-45, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig_sub, use_container_width=True)
        else: st.info("No specific sub-themes found for distribution.")
    except Exception as e: st.error(f"Chart error: {e}"); logging.exception("Failed generating distribution plots.")


# --- Editor and Data Display Functions ---

# Theme Editor (No changes needed for this fix)
def display_theme_editor(themes_data_state_key=EDITED_THEMES_KEY):
    # ... (Keep the full function code from previous versions) ...
    if themes_data_state_key not in st.session_state or not isinstance(st.session_state.get(themes_data_state_key), list): st.session_state[themes_data_state_key] = []; logging.info(f"Init state '{themes_data_state_key}' for editor.")
    themes_data = st.session_state[themes_data_state_key]
    if not themes_data: st.info("No themes generated/loaded. Add below or generate on first tab.")
    def _del_theme_cb(idx): current = st.session_state.get(themes_data_state_key, []); del current[idx]; st.session_state[themes_data_state_key] = current; st.toast(f"Theme {idx+1} removed.")
    def _del_sub_cb(t_idx, s_idx): current = st.session_state.get(themes_data_state_key, []); item = current[t_idx]; subs = item.get('sub_themes', []); del subs[s_idx]; item['sub_themes'] = subs; st.session_state[themes_data_state_key] = current; st.toast(f"Sub-theme removed from Theme {t_idx+1}.")
    def _add_sub_cb(t_idx): current = st.session_state.get(themes_data_state_key, []); item = current[t_idx]; item.setdefault('sub_themes', []).append(""); st.session_state[themes_data_state_key] = current; st.toast(f"New sub-theme added to Theme {t_idx+1}.")
    def _add_theme_cb(): current = st.session_state.get(themes_data_state_key, []); name = f'New Theme {len(current)+1}'; current.append({'theme': name, 'sub_themes': ['New Sub-theme 1'], 'description': ''}); st.session_state[themes_data_state_key] = current; st.toast(f"'{name}' added.")
    num_themes_render = len(st.session_state.get(themes_data_state_key, [])); widget_vals = {}
    for i in range(num_themes_render):
        current_list = st.session_state.get(themes_data_state_key, [])
        if i >= len(current_list): logging.warning(f"Theme idx {i} out of bounds."); break
        item = current_list[i];
        if not isinstance(item, dict): logging.warning(f"Item {i} not dict."); continue
        t_key = f"theme_edit_{themes_data_state_key}_{i}"; s_base = f"sub_edit_{themes_data_state_key}_{i}"; del_t = f"del_theme_{themes_data_state_key}_{i}"; add_s = f"add_sub_{themes_data_state_key}_{i}"
        with st.container():
            st.markdown("---", unsafe_allow_html=True); cols_h = st.columns([0.8, 0.2])
            orig_label = item.get('theme', f'Theme {i+1}'); orig_subs = item.get('sub_themes', []); orig_desc = item.get('description', '')
            with cols_h[0]: widget_vals[t_key] = st.text_input(f"**Theme {i+1}**", value=orig_label, key=t_key); st.caption(f"Desc: {orig_desc or '(N/A)'}")
            with cols_h[1]: st.markdown("<div style='margin-top:28px;'></div>",True); st.button("🗑️", key=del_t, help=f"Del Theme {i+1}", on_click=_del_theme_cb, args=(i,))
            st.write("**Sub-themes:**"); sub_cols = st.columns(4)
            if not isinstance(orig_subs, list): orig_subs = []
            num_subs = len(orig_subs); widget_vals[s_base] = {}
            for j in range(num_subs):
                 sub_val = orig_subs[j]; s_key = f"{s_base}_{j}"; del_s = f"del_sub_{themes_data_state_key}_{i}_{j}"; col_idx = j % len(sub_cols)
                 with sub_cols[col_idx]: widget_vals[s_base][s_key] = st.text_input(f"Sub {j+1}", value=sub_val, key=s_key, label_visibility="collapsed"); st.button("✖", key=del_s, help=f"Del sub {j+1}", on_click=_del_sub_cb, args=(i,j))
            st.button("➕ Add Sub-theme", key=add_s, on_click=_add_sub_cb, args=(i,))
    st.markdown("---"); st.button("➕ Add New Theme", key=f"add_theme_btn_{themes_data_state_key}", on_click=_add_theme_cb)
    st.markdown("---")
    if st.button("💾 Save ALL Refined Themes", key=f"save_themes_btn_{themes_data_state_key}", type="primary"):
        updated_list = []; num_themes_save = len(st.session_state.get(themes_data_state_key, [])); all_ok = True
        for i in range(num_themes_save):
            t_key = f"theme_edit_{themes_data_state_key}_{i}"; s_base = f"sub_edit_{themes_data_state_key}_{i}"
            if t_key in st.session_state:
                theme_label = st.session_state[t_key].strip() or f"Theme {i+1}"
                orig_desc = ""; current_list_save = st.session_state.get(themes_data_state_key, [])
                if i < len(current_list_save) and isinstance(current_list_save[i], dict): orig_desc = current_list_save[i].get('description', '')
                saved_subs = []; num_subs_exp = 0
                if i < len(current_list_save) and isinstance(current_list_save[i], dict): sub_list_state = current_list_save[i].get('sub_themes', []); num_subs_exp = len(sub_list_state) if isinstance(sub_list_state, list) else 0
                for j in range(num_subs_exp):
                     s_key = f"{s_base}_{j}"
                     if s_key in st.session_state: sub_label = st.session_state[s_key].strip(); saved_subs.append(sub_label) if sub_label else None
                     else: logging.warning(f"Sub key {s_key} miss."); all_ok = False
                updated_list.append({'theme': theme_label, 'sub_themes': saved_subs, 'description': orig_desc})
            else: logging.warning(f"Theme key {t_key} miss."); all_ok = False
        st.session_state[themes_data_state_key] = updated_list
        if all_ok: st.success("Themes saved."); st.toast("Themes updated!", icon="✏️")
        else: st.warning("Themes saved, review below."); st.toast("Themes updated (check review).", icon="⚠️")
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
    up_file = st.file_uploader("⬆️ Upload Structure (JSON)", type=['json'], key=UPLOADED_THEME_FILE_WIDGET_KEY)
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


# Assignment Results Editor (with NaN/None handling fix + Safe editor output check)
def display_assignment_results_editable(df_state_key=ASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    """
    Displays the theme assignment results in an editable st.data_editor.
    Includes robust handling for NaN/None before displaying and safe handling of editor output.
    """
    # --- Initial Check ---
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame):
        st.info("No assignment results to display/edit. Run assignment first.")
        return

    # --- Get Data and Ensure it's DataFrame ---
    results_df = st.session_state[df_state_key]
    if not isinstance(results_df, pd.DataFrame): # Extra check
        st.error("Invalid assignment data type found in session state. Please rerun assignment.")
        st.session_state.pop(df_state_key, None) # Clear invalid state
        return
    if results_df.empty:
        st.info("Assignment results are empty.")
        # Optional: Provide a way to clear this empty state if needed
        # if st.button("Clear Empty Results"): st.session_state.pop(df_state_key, None); st.rerun()
        return

    st.info("Manually review/edit assignments below. Click 'Save Manual Changes' to update.")

    # --- Prepare Theme Options ---
    theme_structure = st.session_state.get(themes_state_key, [])
    valid_theme_labels = []
    if isinstance(theme_structure, list) and theme_structure:
        valid_theme_labels = sorted([str(t.get('theme', '')).strip() for t in theme_structure if t.get('theme') and str(t.get('theme','')).strip()])
    else: st.warning("Theme structure missing/invalid. Theme dropdown limited.")
    all_theme_options_set = set(valid_theme_labels) | {"Uncategorized"} | set(results_df['assigned_theme'].astype(str).unique())
    theme_options = sorted([opt for opt in all_theme_options_set if pd.notna(opt) and str(opt).strip() and "Error" not in opt])
    if not theme_options: theme_options = ["Uncategorized"]

    # --- DataFrame Preparation for Editor ---
    required_cols = ["response", "assigned_theme", "assigned_sub_theme", "assignment_confidence"]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        st.error(f"Assignment DF missing required columns: {', '.join(missing_cols)}. Cannot display editor.")
        logging.error(f"Themer Assignment DF missing cols: {missing_cols}")
        return

    try:
        df_display = results_df.copy()
        # FIX: Explicitly fill NaN/None with defaults & ensure string types
        df_display['assigned_theme'] = df_display['assigned_theme'].fillna("Uncategorized").astype(str)
        df_display['assigned_sub_theme'] = df_display['assigned_sub_theme'].fillna("N/A").astype(str)
        df_display['assignment_confidence'] = df_display['assignment_confidence'].fillna("Low").astype(str)
        df_display['response'] = df_display['response'].fillna("").astype(str)
        # Enforce 'Uncategorized' rule before display
        df_display.loc[df_display['assigned_theme'] == "Uncategorized", 'assigned_sub_theme'] = "N/A"
    except Exception as e:
        st.error(f"Failed preparing data for editor: {e}")
        logging.error(f"Themer: Error setting column types/defaults for data editor: {e}")
        return

    # --- Define Column Configurations ---
    column_config = {
        "response": st.column_config.TextColumn("Response", width="large", disabled=True),
        "assigned_theme": st.column_config.SelectboxColumn("Assigned Theme", width="medium", options=theme_options, required=True),
        "assigned_sub_theme": st.column_config.TextColumn("Assigned Sub-theme", width="medium", required=True),
        "assignment_confidence": st.column_config.SelectboxColumn("Confidence", width="small", options=["High", "Medium", "Low"], required=True),
    }

    # --- Display Data Editor ---
    st.markdown("#### Edit Assignments")
    edited_df = st.data_editor(
        df_display,
        key=ASSIGNMENT_EDITOR_WIDGET_KEY,
        use_container_width=True, column_config=column_config,
        column_order=required_cols, num_rows="fixed", hide_index=True,
        disabled=["response"]
    )

    # --- Save Changes Button ---
    if st.button("💾 Save Manual Assignment Changes", key=f"save_manual_assignments_btn_{df_state_key}"):
        # --- SAFEGUARD: Check if editor returned a valid DataFrame ---
        if not isinstance(edited_df, pd.DataFrame):
            st.error("Error saving changes: Editor did not return valid data. Please refresh and try again.")
            logging.error("Themer: st.data_editor did not return a DataFrame object.")
            return # Stop processing if edited_df is not what we expect

        # --- Validation Logic ---
        valid = True; validation_warnings = []
        theme_subtheme_map = {str(t.get('theme','')): [str(s) for s in t.get('sub_themes',[])] for t in theme_structure if t.get('theme')}
        for index, row in edited_df.iterrows():
            theme = row['assigned_theme']; sub_theme = row['assigned_sub_theme']
            # Ensure required fields are not empty after editing
            if not theme: validation_warnings.append(f"Row {index+1}: Assigned Theme cannot be empty."); valid=False
            if not sub_theme: validation_warnings.append(f"Row {index+1}: Assigned Sub-theme cannot be empty."); valid=False
            if not row['assignment_confidence']: validation_warnings.append(f"Row {index+1}: Confidence cannot be empty."); valid=False
            # Check specific rules
            if theme == "Uncategorized" and sub_theme != "N/A": validation_warnings.append(f"Row {index+1}: Uncategorized theme needs 'N/A' sub-theme."); valid = False
            if theme != "Uncategorized" and sub_theme == "N/A": pass # Allow N/A for categorized themes if user chose it
            elif theme != "Uncategorized" and sub_theme == "General": pass # Allow General
            # Add any other critical validation here

        if valid:
            # Save the edited DataFrame back to session state
            # Ensure index consistency if needed - data_editor usually maintains it for 'fixed' rows
            try:
                # Make sure index aligns if it got reset by editor (unlikely for fixed rows but safe)
                edited_df.index = st.session_state[df_state_key].index
                st.session_state[df_state_key] = edited_df.copy() # Save a copy
                st.success("Manual assignment changes saved successfully!")
                st.toast("Assignments updated!", icon="💾")
                logging.info(f"Themer: Manual assignments saved to state '{df_state_key}'.")
                time.sleep(0.5); st.rerun()
            except Exception as save_err:
                 st.error(f"Error saving changes: {save_err}")
                 logging.error(f"Themer: Error saving edited DataFrame: {save_err}")

        else:
            st.error("Validation Failed! Correct issues below before saving:")
            for i, w in enumerate(validation_warnings):
                 if i < 5: st.warning(w)
                 else: st.warning(f"...and {len(validation_warnings) - 5} more issues."); break
            logging.warning(f"Themer: Manual assignment save failed validation ({len(validation_warnings)} issues).")


# Theme Examples Display
def display_theme_examples(df_state_key=ASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    # ... (Keep the full function code from previous versions) ...
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
