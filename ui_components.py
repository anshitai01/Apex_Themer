# ui_components.py
"""
Functions for creating and displaying UI components in the Streamlit app.
Includes visualizations (word cloud, charts), editors (themes, assignments),
and data exploration elements.
ADAPTED FOR PHRONESIS APEX THEME.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import logging
import time
import json
import random

# Import necessary config keys and NEW color variables
from config import (
    EDITED_THEMES_KEY, ASSIGNMENT_DF_KEY, UPLOADED_THEME_FILE_WIDGET_KEY,
    ASSIGNMENT_EDITOR_WIDGET_KEY,
    # New theme colors needed for charts etc.
    MAIN_BACKGROUND_COLOR, BODY_TEXT_COLOR, PRIMARY_ACCENT_COLOR,
    CHART_SUCCESS_COLOR, CHART_WARNING_COLOR, CHART_ERROR_COLOR
)

# --- Visualization Functions ---

def create_word_cloud(responses):
    """
    Generates a matplotlib figure containing a word cloud from text responses.
    ADAPTED FOR DARK THEME.

    Args:
        responses (list): A list of strings.

    Returns:
        matplotlib.figure.Figure or None: The generated figure, or None if generation fails.
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

        # Generate word cloud object - TRANSPARENT BACKGROUND, new colormap
        wordcloud = WordCloud(
            width=800, height=350,
            background_color=None, # Use None for transparent background
            mode="RGBA",           # Ensure RGBA mode for transparency
            colormap='plasma',     # Choose a colormap suitable for dark backgrounds
            max_words=100,
            random_state=42
            ).generate(text)

        # Create Matplotlib figure - Ensure figure/axis background is transparent
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(MAIN_BACKGROUND_COLOR) # Match app background
        fig.patch.set_alpha(0.0) # Make figure background transparent if possible
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


def display_theme_distribution(df_state_key):
    """
    Shows bar charts for theme, sub-theme, and confidence distribution.
    ADAPTED FOR DARK THEME.

    Args:
        df_state_key (str): The session state key holding the assignment DataFrame.
    """
    # ... (initial data loading and validation code remains the same) ...
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame):
        st.info("Assign themes first to view distribution charts.")
        return
    df = st.session_state[df_state_key]
    if df.empty or 'assigned_theme' not in df.columns:
        st.info("No assignment data available for visualization.")
        return
    try:
        df['assigned_theme'] = df['assigned_theme'].astype(str)
        df['assigned_sub_theme'] = df['assigned_sub_theme'].astype(str)
        if 'assignment_confidence' in df.columns:
            df['assignment_confidence'] = df['assignment_confidence'].astype(str)
        else:
            logging.warning("Confidence column missing.")
    except Exception as e:
        st.error(f"Could not prepare assignment columns for charting: {e}")
        logging.error(f"Failed converting assignment columns to string type: {e}")
        return
    df_valid = df[~df['assigned_theme'].str.contains("Error", case=False, na=False)].copy()
    if df_valid.empty:
        st.info("No valid (non-error) theme assignments found for visualization.")
        return

    # --- Plotting Section with Dark Theme ---
    try:
        col1, col2 = st.columns(2)
        plotly_template = "plotly_dark" # Use Plotly's built-in dark theme

        # --- Main themes ---
        with col1:
            st.subheader("Main Theme Distribution")
            theme_counts = df_valid['assigned_theme'].value_counts().reset_index()
            theme_counts.columns = ['Main Theme', 'Count']
            if not theme_counts.empty:
                fig_main = px.bar(theme_counts, x='Main Theme', y='Count',
                                  title="Distribution of Main Themes", text_auto=True,
                                  color_discrete_sequence=px.colors.qualitative.Pastel, # Keep pastel or choose another sequence
                                  template=plotly_template) # Apply dark theme
                fig_main.update_layout(xaxis_title=None, yaxis_title="Number of Responses", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_main, use_container_width=True)
            else: st.info("No main themes assigned.")

        # --- Confidence Distribution ---
        with col2:
            st.subheader("Confidence Distribution")
            if 'assignment_confidence' in df_valid.columns:
                 confidence_counts = df_valid['assignment_confidence'].value_counts().reset_index()
                 confidence_counts.columns = ['Confidence', 'Count']
                 confidence_order = ["High", "Medium", "Low"]
                 # Use new colors from config.py
                 color_map = {'High': CHART_SUCCESS_COLOR, 'Medium': CHART_WARNING_COLOR, 'Low': CHART_ERROR_COLOR}
                 try:
                     confidence_counts['Confidence'] = pd.Categorical(confidence_counts['Confidence'], categories=confidence_order, ordered=True)
                     confidence_counts = confidence_counts.sort_values('Confidence')
                 except Exception as cat_e:
                     st.warning(f"Could not sort confidence levels: {cat_e}")
                     logging.warning(f"Categorical conversion/sorting failed for confidence: {cat_e}")

                 if not confidence_counts.empty:
                      fig_conf = px.bar(confidence_counts, x='Confidence', y='Count',
                                       title="Distribution of Assignment Confidence", text_auto=True,
                                       color='Confidence', color_discrete_map=color_map,
                                       template=plotly_template) # Apply dark theme
                      fig_conf.update_layout(xaxis_title="Confidence Level", yaxis_title="Number of Responses", showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                      st.plotly_chart(fig_conf, use_container_width=True)
                 else: st.info("No confidence data available.")
            else:
                 st.info("Confidence data column not found.")

        # --- Sub-themes (overall) ---
        st.subheader("Overall Sub-theme Distribution")
        excluded_sub_themes = ['N/A', 'Error', 'General', 'Other', '', 'nan', 'None']
        sub_theme_counts = df_valid[
            ~df_valid['assigned_sub_theme'].astype(str).isin(excluded_sub_themes) &
            (df_valid['assigned_theme'] != "Uncategorized")
            ]['assigned_sub_theme'].value_counts().reset_index()
        sub_theme_counts.columns = ['Sub-theme', 'Count']

        if not sub_theme_counts.empty:
            max_subthemes_to_show = 25
            if len(sub_theme_counts) > max_subthemes_to_show:
                sub_theme_counts = sub_theme_counts.head(max_subthemes_to_show)
                st.caption(f"Showing top {max_subthemes_to_show} specific sub-themes.")

            fig_sub = px.bar(sub_theme_counts, x='Sub-theme', y='Count',
                             title="Distribution of Top Specific Sub-themes",
                             color_discrete_sequence=px.colors.qualitative.Set2, # Choose a sequence
                             template=plotly_template) # Apply dark theme
            fig_sub.update_layout(xaxis_title=None, yaxis_title="Number of Responses", xaxis_tickangle=-45, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
            st.info("No specific sub-themes assigned or available for distribution.")

    except Exception as e:
        st.error(f"Error creating distribution plots: {e}")
        logging.exception("Failed to generate theme distribution plots.")


# --- Editor and Data Display Functions ---
# display_theme_editor(...) - No direct theme changes needed here, styling handled by CSS
# display_assignment_results_editable(...) - No direct theme changes needed here, styling handled by CSS
# display_theme_examples(...) - No direct theme changes needed here, styling handled by CSS
# Keep the existing functions for these as they primarily deal with data and logic,
# relying on the CSS in styling.py for their appearance.

# Need to copy the existing functions over from the previous step's ui_components.py
# as they were not included in the user prompt for this step.

# --- display_theme_editor ---
def display_theme_editor(themes_data_state_key=EDITED_THEMES_KEY):
    # ... (Keep the full function code from the previous step) ...
    # It relies on CSS for styling, no internal changes needed for the theme.
    # Ensure the functions like _delete_theme_callback etc. are defined inside or passed appropriately
    # --- [PASTE THE FULL display_theme_editor FUNCTION HERE] ---
    # Ensure the state key exists and is a list
    if themes_data_state_key not in st.session_state or not isinstance(st.session_state.get(themes_data_state_key), list):
        st.session_state[themes_data_state_key] = []
        logging.info(f"Initialized/reset session state key '{themes_data_state_key}' to empty list for editor.")

    themes_data = st.session_state[themes_data_state_key]

    if not themes_data:
        st.info("No themes generated or loaded yet. Add a new theme structure below or generate themes on the first tab.")

    # --- Button Callbacks (Modify session state directly) ---
    def _delete_theme_callback(index_to_delete):
        current_themes = st.session_state.get(themes_data_state_key, [])
        if 0 <= index_to_delete < len(current_themes):
            del current_themes[index_to_delete]
            st.session_state[themes_data_state_key] = current_themes # Update state
            st.toast(f"Theme {index_to_delete+1} removed.")

    def _delete_sub_theme_callback(theme_idx, sub_theme_idx_to_delete):
        current_themes = st.session_state.get(themes_data_state_key, [])
        if 0 <= theme_idx < len(current_themes):
             theme_item = current_themes[theme_idx]
             sub_themes = theme_item.get('sub_themes', [])
             if isinstance(sub_themes, list) and 0 <= sub_theme_idx_to_delete < len(sub_themes):
                del sub_themes[sub_theme_idx_to_delete]
                theme_item['sub_themes'] = sub_themes
                st.session_state[themes_data_state_key] = current_themes
                st.toast(f"Sub-theme removed from Theme {theme_idx+1}.")
             else:
                 logging.warning(f"Sub-theme index {sub_theme_idx_to_delete} invalid for theme {theme_idx} during delete.")
                 st.warning("Could not delete sub-theme (index invalid).")

    def _add_sub_theme_callback(theme_idx):
         current_themes = st.session_state.get(themes_data_state_key, [])
         if 0 <= theme_idx < len(current_themes):
              theme_item = current_themes[theme_idx]
              if 'sub_themes' not in theme_item or not isinstance(theme_item.get('sub_themes'), list):
                  theme_item['sub_themes'] = []
              theme_item['sub_themes'].append("")
              st.session_state[themes_data_state_key] = current_themes
              st.toast(f"New sub-theme input added to Theme {theme_idx+1}.")
         else:
             logging.warning(f"Theme index {theme_idx} invalid during add sub-theme.")
             st.warning("Could not add sub-theme (theme index invalid).")

    def _add_new_theme_entry_callback():
        current_themes = st.session_state.get(themes_data_state_key, [])
        new_theme_name = f'New Theme {len(current_themes) + 1}'
        current_themes.append({
            'theme': new_theme_name,
            'sub_themes': ['New Sub-theme 1'],
            'description': ''
        })
        st.session_state[themes_data_state_key] = current_themes
        st.toast(f"'{new_theme_name}' added. Scroll down to edit.")

    # --- Render Themes Iteratively ---
    num_themes_to_render = len(st.session_state.get(themes_data_state_key, []))
    widget_values = {}

    for i in range(num_themes_to_render):
        current_themes_list = st.session_state.get(themes_data_state_key, [])
        if i >= len(current_themes_list):
            logging.warning(f"Theme index {i} became out of bounds during render loop. Breaking.")
            break
        item = current_themes_list[i]
        if not isinstance(item, dict):
             logging.warning(f"Item at index {i} in theme list is not a dict: {item}. Skipping render.")
             continue

        theme_key = f"theme_edit_{themes_data_state_key}_{i}"
        sub_theme_base_key = f"sub_theme_edit_{themes_data_state_key}_{i}"
        delete_theme_key = f"del_theme_{themes_data_state_key}_{i}"
        add_sub_key = f"add_sub_{themes_data_state_key}_{i}"

        with st.container():
            st.markdown(f"---", unsafe_allow_html=True)
            cols_header = st.columns([0.8, 0.2])

            original_theme_label = item.get('theme', f'Unnamed Theme {i+1}')
            original_sub_themes = item.get('sub_themes', [])
            original_description = item.get('description', '')

            with cols_header[0]:
                widget_values[theme_key] = st.text_input(
                    f"**Theme {i+1}**", value=original_theme_label, key=theme_key
                )
                if original_description:
                     st.caption(f"Description: {original_description}")
                else:
                     st.caption("Description: (Not available or generated)")

            with cols_header[1]:
                 st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                 st.button("🗑️", key=delete_theme_key, help=f"Delete Theme {i+1}",
                           on_click=_delete_theme_callback, args=(i,))

            st.write("**Sub-themes:**") # Made bold
            sub_cols = st.columns(4)
            if not isinstance(original_sub_themes, list):
                logging.warning(f"Sub-themes for theme {i} is not a list. Skipping sub-theme rendering.")
                original_sub_themes = []

            num_sub_themes_rendered = len(original_sub_themes)
            widget_values[sub_theme_base_key] = {}

            for j in range(num_sub_themes_rendered):
                 sub_theme_value = original_sub_themes[j]
                 sub_key = f"{sub_theme_base_key}_{j}"
                 delete_sub_key = f"del_sub_{themes_data_state_key}_{i}_{j}"
                 col_index = j % len(sub_cols)
                 with sub_cols[col_index]:
                     widget_values[sub_theme_base_key][sub_key] = st.text_input(
                         f"Sub {j+1}", value=sub_theme_value, key=sub_key, label_visibility="collapsed"
                         )
                     st.button("✖", key=delete_sub_key, help=f"Delete sub-theme {j+1}",
                                on_click=_delete_sub_theme_callback, args=(i, j))

            st.button("➕ Add Sub-theme", key=add_sub_key, on_click=_add_sub_theme_callback, args=(i,))

    st.markdown("---")
    st.button("➕ Add New Theme Structure", key=f"add_new_theme_main_btn_{themes_data_state_key}",
              on_click=_add_new_theme_entry_callback)

    # --- Save ALL Changes Button ---
    st.markdown("---")
    if st.button("💾 Save ALL Refined Themes", key=f"save_themes_btn_main_editor_{themes_data_state_key}", type="primary"):
        updated_theme_list = []
        num_themes_in_state_at_save_start = len(st.session_state.get(themes_data_state_key, []))
        all_keys_processed = True

        for i in range(num_themes_in_state_at_save_start):
            theme_key = f"theme_edit_{themes_data_state_key}_{i}"
            sub_theme_base_key = f"sub_theme_edit_{themes_data_state_key}_{i}"

            if theme_key in st.session_state:
                theme_label = st.session_state[theme_key].strip() or f"Unnamed Theme {i+1}"
                original_description = ""
                current_themes_list_at_save = st.session_state.get(themes_data_state_key, [])
                if i < len(current_themes_list_at_save) and isinstance(current_themes_list_at_save[i], dict):
                    original_description = current_themes_list_at_save[i].get('description', '')

                saved_sub_themes = []
                num_sub_themes_expected = 0
                if i < len(current_themes_list_at_save) and isinstance(current_themes_list_at_save[i], dict):
                    sub_themes_list_in_state = current_themes_list_at_save[i].get('sub_themes', [])
                    if isinstance(sub_themes_list_in_state, list):
                         num_sub_themes_expected = len(sub_themes_list_in_state)

                for j in range(num_sub_themes_expected):
                     sub_key = f"{sub_theme_base_key}_{j}"
                     if sub_key in st.session_state:
                         sub_theme_label = st.session_state[sub_key].strip()
                         if sub_theme_label:
                             saved_sub_themes.append(sub_theme_label)
                     else:
                          logging.warning(f"Sub-theme widget key {sub_key} not found during save.")
                          all_keys_processed = False

                updated_theme_list.append({
                    'theme': theme_label,
                    'sub_themes': saved_sub_themes,
                    'description': original_description
                })
            else:
                 logging.warning(f"Theme widget key {theme_key} not found during save processing for index {i}. Theme skipped.")
                 all_keys_processed = False

        st.session_state[themes_data_state_key] = updated_theme_list
        if all_keys_processed:
             st.success("Refined themes saved successfully!")
             st.toast("Themes updated!", icon="✏️")
        else:
             st.warning("Themes saved, but some input fields may not have been fully captured. Please review.")
        st.rerun()

    # --- Display Current Saved Structure & Save/Load Functionality ---
    st.markdown("---")
    st.subheader("Current Saved Theme Structure")
    current_saved_themes = st.session_state.get(themes_data_state_key, [])
    if current_saved_themes:
         try:
             json_string = json.dumps(current_saved_themes, indent=2)
             st.download_button(
                 label="💾 Download Theme Structure (JSON)", data=json_string,
                 file_name=f"theme_structure_{time.strftime('%Y%m%d_%H%M')}.json",
                 mime="application/json", key=f"download_themes_json_btn_{themes_data_state_key}"
             )
         except Exception as e:
             st.error(f"Error preparing theme structure for download: {e}")
             logging.error(f"Error creating JSON for theme download: {e}")

         for idx, theme_item in enumerate(current_saved_themes):
             theme_label = theme_item.get('theme', f'Unnamed Theme {idx+1}')
             description = theme_item.get('description', 'N/A')
             sub_themes_list = theme_item.get('sub_themes', [])
             with st.expander(f"Theme {idx+1}: {theme_label}", expanded=False):
                 st.caption(f"Description: {description}")
                 st.write("**Sub-themes:**")
                 if sub_themes_list:
                     # Use st.code for better display in dark mode
                     st.code(json.dumps(sub_themes_list, indent=2), language='json')
                 else:
                     st.caption("(None)")
    else:
         st.info("No theme structure currently saved.")
         st.caption("Use the editor above to add themes or load a structure below.")

    # --- Load Theme Structure ---
    st.markdown("---")
    st.subheader("Load Theme Structure")
    uploaded_theme_file = st.file_uploader(
        "⬆️ Upload Theme Structure (JSON)", type=['json'], key=UPLOADED_THEME_FILE_WIDGET_KEY
    )
    if uploaded_theme_file is not None:
        try:
            loaded_themes = json.load(uploaded_theme_file)
            is_valid_structure = isinstance(loaded_themes, list)
            processed_loaded_themes = []
            if is_valid_structure:
                for item_idx, item in enumerate(loaded_themes):
                    if not (isinstance(item, dict) and 'theme' in item and 'sub_themes' in item):
                         st.error(f"Invalid item structure at index {item_idx}: Missing 'theme' or 'sub_themes'. Item: `{str(item)[:100]}`")
                         is_valid_structure = False; break
                    if not isinstance(item.get('theme'), str) or not isinstance(item.get('sub_themes'), list):
                         st.error(f"Invalid item type at index {item_idx}: 'theme' must be string, 'sub_themes' must be list.")
                         is_valid_structure = False; break

                    processed_item = {}
                    processed_item['theme'] = str(item.get('theme', '')).strip()
                    processed_item['description'] = str(item.get('description', '')).strip()
                    processed_item['sub_themes'] = [str(s).strip() for s in item.get('sub_themes', []) if str(s).strip()]

                    if not processed_item['theme']:
                        st.error(f"Invalid item at index {item_idx}: 'theme' label cannot be empty.")
                        is_valid_structure = False; break
                    processed_loaded_themes.append(processed_item)

            if is_valid_structure and processed_loaded_themes:
                st.session_state[themes_data_state_key] = processed_loaded_themes
                st.success(f"Successfully loaded and validated {len(processed_loaded_themes)} themes from '{uploaded_theme_file.name}'.")
                st.toast("Themes loaded!", icon="⬆️")
                st.rerun()
            elif not is_valid_structure:
                 logging.warning(f"Invalid theme structure in uploaded file: {uploaded_theme_file.name}")
            else:
                 st.warning("Uploaded theme file contains an empty list.")
                 st.session_state[themes_data_state_key] = []
                 st.rerun()

        except json.JSONDecodeError:
            st.error("Failed to parse file as JSON. Ensure it's a valid JSON file.")
            logging.error(f"JSONDecodeError loading theme file: {uploaded_theme_file.name}")
        except Exception as e:
            st.error(f"An unexpected error occurred loading theme structure: {e}")
            logging.exception(f"Error loading theme structure from file: {uploaded_theme_file.name}")


# --- display_assignment_results_editable ---
def display_assignment_results_editable(df_state_key=ASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    # ... (Keep the full function code from the previous step) ...
    # It relies on CSS for styling, no internal changes needed for the theme.
    # --- [PASTE THE FULL display_assignment_results_editable FUNCTION HERE] ---
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame):
        st.info("No theme assignment results to display or edit. Please run assignment first.")
        return

    results_df = st.session_state[df_state_key]
    if results_df.empty:
         st.info("Assignment results are empty.")
         return

    st.info("Manually review and edit the assigned themes/sub-themes/confidence below. Click 'Save Manual Changes' to update.")

    theme_structure = st.session_state.get(themes_state_key, [])
    if not theme_structure or not isinstance(theme_structure, list):
        st.warning("Theme structure not found or invalid. Theme dropdown will be limited.")
        valid_theme_labels = []
    else:
        valid_theme_labels = sorted([str(t.get('theme', '')) for t in theme_structure if t.get('theme') and str(t.get('theme','')).strip()])

    theme_options = sorted(list(set(valid_theme_labels + ["Uncategorized"] + results_df['assigned_theme'].unique().tolist())))
    theme_options = [opt for opt in theme_options if pd.notna(opt) and str(opt).strip()]

    required_cols = ["response", "assigned_theme", "assigned_sub_theme", "assignment_confidence"]
    missing_cols = [col for col in required_cols if col not in results_df.columns]

    if missing_cols:
        st.error(f"Assignment DataFrame is missing required columns: {', '.join(missing_cols)}. Cannot display editor.")
        logging.error(f"Assignment DF missing columns: {missing_cols}")
        return

    try:
        df_display = results_df.copy()
        df_display['assigned_theme'] = df_display['assigned_theme'].astype(str)
        df_display['assigned_sub_theme'] = df_display['assigned_sub_theme'].astype(str)
        df_display['assignment_confidence'] = df_display['assignment_confidence'].astype(str)
        df_display['response'] = df_display['response'].astype(str)
    except Exception as e:
        st.error(f"Failed to set column types for editor: {e}")
        logging.error(f"Error setting column types for data editor: {e}")
        return

    column_config = {
        "response": st.column_config.TextColumn("Response", width="large", disabled=True, help="Original survey response (read-only)."),
        "assigned_theme": st.column_config.SelectboxColumn("Assigned Theme", width="medium", options=theme_options, required=True, help="Select the best fitting main theme."),
        "assigned_sub_theme": st.column_config.TextColumn("Assigned Sub-theme", width="medium", required=True, help="Manually type the best fitting sub-theme (e.g., a specific one, 'General', 'N/A')."),
        "assignment_confidence": st.column_config.SelectboxColumn("Confidence", width="small", options=["High", "Medium", "Low"], required=True, help="Manually set the confidence level of this assignment.")
    }

    st.markdown("#### Edit Assignments")
    edited_df = st.data_editor(
        df_display,
        key=ASSIGNMENT_EDITOR_WIDGET_KEY,
        use_container_width=True,
        column_config=column_config,
        column_order=required_cols,
        # num_rows="dynamic", # Or fixed?
        num_rows="fixed", # Keep rows fixed seems safer
        hide_index=True,
        disabled=["response"]
    )

    if st.button("💾 Save Manual Assignment Changes", key=f"save_manual_assignments_btn_{df_state_key}"):
        is_valid = True
        validation_warnings = []
        theme_subtheme_map = {str(t.get('theme','')): [str(s) for s in t.get('sub_themes',[])] for t in theme_structure if t.get('theme')}

        for index, row in edited_df.iterrows():
            theme = row['assigned_theme']
            sub_theme = row['assigned_sub_theme']
            if theme == "Uncategorized" and sub_theme != "N/A":
                 validation_warnings.append(f"Row {index+1}: Theme 'Uncategorized' requires Sub-theme 'N/A' (found '{sub_theme}').")
                 is_valid = False
            if theme != "Uncategorized" and not sub_theme:
                 validation_warnings.append(f"Row {index+1}: Sub-theme cannot be empty for theme '{theme}'. Use 'General' or 'N/A'.")
                 is_valid = False

        if is_valid:
            st.session_state[df_state_key] = edited_df.copy()
            st.success("Manual assignment changes saved successfully!")
            st.toast("Assignments updated!", icon="💾")
            logging.info(f"Manual assignment changes saved to state key '{df_state_key}'.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Validation Failed! Please correct the issues listed below before saving:")
            for i, warning in enumerate(validation_warnings):
                 if i < 5: st.warning(warning)
                 else: st.warning(f"...and {len(validation_warnings) - 5} more issues."); break
            logging.warning(f"Manual assignment save validation failed with {len(validation_warnings)} issues.")


# --- display_theme_examples ---
def display_theme_examples(df_state_key=ASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    # ... (Keep the full function code from the previous step) ...
    # It relies on CSS for styling, no internal changes needed for the theme.
    # --- [PASTE THE FULL display_theme_examples FUNCTION HERE] ---
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame):
        st.info("Assign themes first to explore examples.")
        return
    results_df = st.session_state[df_state_key]
    if results_df.empty or 'assigned_theme' not in results_df.columns:
        st.info("No assignment data available to explore.")
        return

    theme_structure = st.session_state.get(themes_state_key, [])
    theme_map = {}
    if isinstance(theme_structure, list) and theme_structure:
        theme_map = {str(t.get('theme','')): t for t in theme_structure if t.get('theme')}
    else:
        st.caption("Theme structure (for descriptions/sub-themes) not found or empty.")

    try:
        valid_themes_in_data = sorted([
            str(th) for th in results_df['assigned_theme'].unique()
            if pd.notna(th) and isinstance(th, str) and "Error" not in th
            ])
    except Exception as e:
        st.error(f"Error processing assigned themes for selection: {e}")
        logging.error(f"Error getting unique themes from assignment DF: {e}")
        valid_themes_in_data = []

    if not valid_themes_in_data:
        st.warning("No valid themes found in assignment results to explore.")
        return

    selected_theme = st.selectbox(
        "Select Theme to Explore:", valid_themes_in_data,
        key=f"explore_theme_select_widget_{df_state_key}"
        )

    if selected_theme:
        st.markdown(f"#### Exploring Theme: **{selected_theme}**")
        theme_details = theme_map.get(selected_theme)

        if theme_details:
            st.markdown("**Description:**")
            description = theme_details.get('description', '').strip()
            if description: st.markdown(f"> {description}")
            else: st.caption("(No description available in theme structure)")

            st.markdown("**Defined Sub-themes in Structure:**")
            sub_themes = theme_details.get('sub_themes', [])
            if sub_themes and isinstance(sub_themes, list):
                sub_themes_str = [str(s) for s in sub_themes]
                if sub_themes_str: st.code(f"{', '.join(sub_themes_str)}", language=None) # Use code for lists
                else: st.caption("(None defined in structure)")
            else: st.caption("(None defined in structure)")
        elif selected_theme != "Uncategorized":
             st.caption(f"Details for theme '{selected_theme}' not found in the saved theme structure.")

        st.markdown("**Example Responses (Random Sample):**")
        try:
            if 'assigned_theme' in results_df.columns:
                 theme_responses_df = results_df[results_df['assigned_theme'] == selected_theme].copy()
            else:
                 st.error("Critical error: 'assigned_theme' column not found.")
                 return

            if not theme_responses_df.empty:
                 num_available = len(theme_responses_df)
                 num_examples_to_show = min(5, num_available)
                 examples = theme_responses_df.sample(n=num_examples_to_show, random_state=42)

                 for index, row in examples.iterrows():
                     st.markdown("---")
                     response_text = str(row.get('response', 'N/A'))
                     sub_theme_text = str(row.get('assigned_sub_theme', 'N/A'))
                     confidence_text = str(row.get('assignment_confidence', 'N/A'))
                     st.markdown(f"> {response_text}") # Blockquote for response
                     st.caption(f"Assigned Sub-theme: `{sub_theme_text}` | Confidence: `{confidence_text}`")
            else:
                 st.info(f"No responses found assigned to the theme '{selected_theme}'.")
        except KeyError as ke:
             st.error(f"Error retrieving examples: Missing expected column - {ke}")
             logging.error(f"KeyError accessing columns for theme examples: {ke}")
        except Exception as e:
             st.error(f"An unexpected error occurred retrieving examples for theme '{selected_theme}': {e}")
             logging.exception(f"Error sampling/displaying theme examples for '{selected_theme}'.")

# Make sure all required functions are defined before being called.
# Paste the full functions for display_theme_editor, display_assignment_results_editable,
# and display_theme_examples here if they weren't fully included above.