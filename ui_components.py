# ui_components.py (for AI Themer App)
"""
Functions for creating and displaying UI components in the Streamlit app.
Includes visualizations (word cloud, charts), editors (themes, assignments),
and data exploration elements. FINAL VERSION with fixes for data editor loop.
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
    EDITED_THEMES_KEY, ASSs `ui_components.py`** file.

This version includes:
*   The Phronesis Apex theme adaptations (dark theme charts, etc.).
*   The corrected initial validation logic in `display_assignment_results_editable` to prevent the error loop.
*   The removal of any previous debugging print statements.

```python
# ui_components.py (for AI Themer App)
"""
Functions for creating and displaying UI components in the Streamlit app.
Includes visualizations (word cloud, charts), editors (themes, assignments),
and data exploration elements. Final version with fixes for data editor state loop.
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
import io # Needed for df.info() buffer if debugging is re-added

# Import specific config keys and constants from Themer's config
# Ensure these variables are correctly defined in your config.py
from config import (
    EDITED_THEMES_KEY, ASSIGNMENT_DF_KEY,
    UPLOADED_THEME_FILE_WIDGET_KEY, ASSIGNMENT_EDITOR_WIDGET_KEY,
    PRIMARY_ACCENT_COLOR, CHART_SUCCESS_COLOR, CHART_WARNING_COLOR, CHART_ERROR_COLOR,
    MAIN_BACKGROUND_COLOR, BODY_TEXT_COLOR, CARD_BACKGROUND_COLOR, INPUT_BORDER_COLOR,
    MAIN_TITLE_COLOR, SUBTITLE_COLOR
)


# --- Visualization Functions ---

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
            width=800, heightIGNMENT_DF_KEY,
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
        st.error=350,
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
    except ImportError(f"Error generating word cloud: Missing library ({ie}).")
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
            df['assignment_confidence'] as ie:
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
    Adapted for dark theme.
    """
    if df_state_key not in st.session_state or not isinstance(st = df['assignment_confidence'].astype(str)
        else:
            logging.warning("Confidence column missing.")
    except Exception as e:
        st.error(f"Chart data prep error: {e}")
        logging.error(f"Chart column conversion error: {e}")
        return
    df_valid = df[~df['assigned_theme'].str.contains("Error", case=False, na=False)].copy().session_state.get(df_state_key), pd.DataFrame):
        st.info("Assign themes first for charts.")
        return
    df = st.session_state[df_state_key]
    if
    if df_valid.empty:
        st.info("No valid assignments for charts.")
        return

    try:
        col1, col2 = st.columns(2)
        plotly_template = "plotly_dark"

        # --- Main themes ---
        with col1:
            st.subheader("Main Theme Distribution df.empty or 'assigned_theme' not in df.columns:
        st.info("No assignment data for")
            theme_counts = df_valid['assigned_theme'].value_counts().reset_index()
            theme_counts.columns = ['Main Theme', 'Count']
            if not theme_counts.empty: charts.")
        return
    try:
        # Ensure columns are string type for accurate plotting/filtering
        df_
                fig_main = px.bar(theme_counts, x='Main Theme', y='Count', title="Distribution ofplot = df.copy() # Work on a copy
        df_plot['assigned_theme'] = df_ Main Themes", text_auto=True, color_discrete_sequence=px.colors.qualitative.Pastelplot['assigned_theme'].astype(str)
        df_plot['assigned_sub_theme'] = df, template=plotly_template)
                fig_main.update_layout(xaxis_title=None, y_plot['assigned_sub_theme'].astype(str)
        if 'assignment_confidence' in df_plot.columns:
            df_plot['assignment_confidence'] = df_plot['assignment_confidence'].astypeaxis_title="Responses", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart((str)
        else:
            logging.warning("Confidence column missing for distribution plot.")
    except Exception as e:fig_main, use_container_width=True)
            else:
                st.info("No main
        st.error(f"Chart data prep error: {e}")
        logging.error(f" themes assigned.")

        # --- Confidence Distribution ---
        with col2:
            st.subheader("Confidence DistributionChart column conversion error: {e}")
        return

    df_valid = df_plot[~df_plot['assigned")
            if 'assignment_confidence' in df_valid.columns:
                 conf_counts = df_valid['_theme'].str.contains("Error", case=False, na=False)].copy()
    if df_assignment_confidence'].value_counts().reset_index()
                 conf_counts.columns = ['Confidence', 'Count']
                 conf_order = ["High", "Medium", "Low"]
                 color_map = {'valid.empty:
        st.info("No valid assignments for charts.")
        return

    try:
        colHigh': CHART_SUCCESS_COLOR, 'Medium': CHART_WARNING_COLOR, 'Low': CHART_ERROR_1, col2 = st.columns(2)
        plotly_template = "plotly_dark"

        # --- Main themes ---
        with col1:
            st.subheader("Main Theme Distribution")
            themeCOLOR}
                 try:
                     conf_counts['Confidence'] = pd.Categorical(conf_counts['Confidence'], categories=conf_order, ordered=True)
                     conf_counts = conf_counts.sort__counts = df_valid['assigned_theme'].value_counts().reset_index()
            theme_counts.columns =values('Confidence')
                 except Exception as e:
                     st.warning(f"Could not sort confidence: {e}")
                     logging.warning(f"Confidence sort error: {e}")
                 if not conf_ ['Main Theme', 'Count']
            if not theme_counts.empty:
                fig_main = px.bar(theme_counts, x='Main Theme', y='Count', title="Distribution of Main Themes", textcounts.empty:
                      fig_conf = px.bar(conf_counts, x='Confidence', y='Count', title="Distribution of Assignment Confidence", text_auto=True, color='Confidence', color_discrete_map=color__auto=True, color_discrete_sequence=px.colors.qualitative.Pastel, template=plotly_template)
                fig_main.update_layout(xaxis_title=None, yaxis_title="map, template=plotly_template)
                      fig_conf.update_layout(xaxis_title="Level",Responses", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_main, yaxis_title="Responses", showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                       use_container_width=True)
            else:
                st.info("No main themes assigned.")

st.plotly_chart(fig_conf, use_container_width=True)
                 else:
                              # --- Confidence Distribution ---
        with col2:
            st.subheader("Confidence Distribution")
            ifst.info("No confidence data.")
            else:
                 st.info("Confidence column not found.")

 'assignment_confidence' in df_valid.columns:
                 conf_counts = df_valid['assignment_confidence'].        # --- Sub-themes (overall) ---
        st.subheader("Overall Sub-theme Distribution")
        value_counts().reset_index()
                 conf_counts.columns = ['Confidence', 'Count']
                 conf_order = ["High", "Medium", "Low"]
                 color_map = {'High': CHART_excluded_subs = ['N/A', 'Error', 'General', 'Other', '', 'nan', 'None']
        SUCCESS_COLOR, 'Medium': CHART_WARNING_COLOR, 'Low': CHART_ERROR_COLOR}
                 sub_counts = df_valid[~df_valid['assigned_sub_theme'].astype(str).isintry:
                     conf_counts['Confidence'] = pd.Categorical(conf_counts['Confidence'], categories=conf_order, ordered=True)
                     conf_counts = conf_counts.sort_values('Confidence')(excluded_subs) & (df_valid['assigned_theme'] != "Uncategorized")]['assigned_sub_theme'].value_counts().reset_index()
        sub_counts.columns = ['Sub-theme
                 except Exception as e:
                     st.warning(f"Could not sort confidence: {e}")
', 'Count']
        if not sub_counts.empty:
            max_subs = 25
            if len(sub_counts) > max_subs: # Corrected multi-line if
                sub_                     logging.warning(f"Confidence sort error: {e}")
                 if not conf_counts.empty:
                      fig_conf = px.bar(conf_counts, x='Confidence', y='Count', title="Distribution ofcounts = sub_counts.head(max_subs)
                st.caption(f"Showing top {max Assignment Confidence", text_auto=True, color='Confidence', color_discrete_map=color_map, template_subs} specific sub-themes.")
            fig_sub = px.bar(sub_counts, x='Sub-theme', y='Count', title="Distribution of Top Specific Sub-themes", color_discrete_sequence=px.colors=plotly_template)
                      fig_conf.update_layout(xaxis_title="Level", yaxis_title="Responses", showlegend=False, height=400, paper_bgcolor='rgba(0,0.qualitative.Set2, template=plotly_template)
            fig_sub.update_layout(xaxis_title=None, yaxis_title="Responses", xaxis_tickangle=-45, paper_bgcolor='rgba,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                      st.plotly_chart(fig_conf, use_container_width=True)
                 else:
                      st.info(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
("No confidence data.")
            else:
                 st.info("Confidence column not found.")

        # --- Sub-themes (overall) ---
        st.subheader("Overall Sub-theme Distribution")
        excluded_subs            st.info("No specific sub-themes found for distribution.")
    except Exception as e:
        st.error(f"Chart error: {e}")
        logging.exception("Failed generating distribution plots.")


# --- Editor and = ['N/A', 'Error', 'General', 'Other', '', 'nan', 'None']
        sub_counts = df_valid[~df_valid['assigned_sub_theme'].isin(excluded_subs Data Display Functions ---

# Theme Editor
def display_theme_editor(themes_data_state_key=) & (df_valid['assigned_theme'] != "Uncategorized")]['assigned_sub_theme'].EDITED_THEMES_KEY):
    """
    Renders an editable UI for themes/sub-themesvalue_counts().reset_index()
        sub_counts.columns = ['Sub-theme', 'Count'] with Add/Delete/Save functionality.
    Modifies the theme structure list directly in Streamlit's session state.
    
        if not sub_counts.empty:
            max_subs = 25
            if len(sub_counts) > max_subs: # Corrected multi-line if
                sub_counts = sub_counts.Includes loading/saving theme structure from/to JSON.
    """
    if themes_data_state_key not in st.session_state or not isinstance(st.session_state.get(themes_data_statehead(max_subs)
                st.caption(f"Showing top {max_subs} specific sub-themes.")
            fig_sub = px.bar(sub_counts, x='Sub-theme', y='_key), list):
        st.session_state[themes_data_state_key] = []
        logging.info(f"Init state '{themes_data_state_key}' for editor.")
    themesCount', title="Distribution of Top Specific Sub-themes", color_discrete_sequence=px.colors.qualitative.Set2, template=plotly_template)
            fig_sub.update_layout(xaxis_title=_data = st.session_state[themes_data_state_key]
    if not themes_data:
        st.info("No themes generated/loaded. Add below or generate on first tab.")

    # --- ButtonNone, yaxis_title="Responses", xaxis_tickangle=-45, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly Callbacks ---
    def _del_theme_cb(idx): current = st.session_state.get(themes_data_state_key, []); del current[idx]; st.session_state[themes_data_state__chart(fig_sub, use_container_width=True)
        else:
            st.infokey] = current; st.toast(f"Theme {idx+1} removed."); st.rerun()
("No specific sub-themes found for distribution.")
    except Exception as e:
        st.error(f"Chart    def _del_sub_cb(t_idx, s_idx): current = st.session_state error: {e}")
        logging.exception("Failed generating distribution plots.")


# --- Editor and Data Display Functions ---

#.get(themes_data_state_key, []); item = current[t_idx]; subs = item. Theme Editor
def display_theme_editor(themes_data_state_key=EDITED_THEMES_KEY):
get('sub_themes', []); del subs[s_idx]; item['sub_themes'] = subs; st    """
    Renders an editable UI for themes/sub-themes with Add/Delete/Save functionality.
    Modifies.session_state[themes_data_state_key] = current; st.toast(f"Sub-theme removed from Theme {t_idx+1}."); st.rerun()
    def _add_sub the theme structure list directly in Streamlit's session state.
    Includes loading/saving theme structure from/to JSON.
    """
    if themes_data_state_key not in st.session_state or not isinstance(st_cb(t_idx): current = st.session_state.get(themes_data_state_key, []); item = current[t_idx]; item.setdefault('sub_themes', []).append(""); st.session_.session_state.get(themes_data_state_key), list):
        st.session_statestate[themes_data_state_key] = current; st.toast(f"New sub-theme added[themes_data_state_key] = []
        logging.info(f"Init state '{themes_data to Theme {t_idx+1}."); st.rerun()
    def _add_theme_cb_state_key}' for editor.")
    themes_data = st.session_state[themes_data_state_key]
    if not themes_data:
        st.info("No themes generated/loaded.(): current = st.session_state.get(themes_data_state_key, []); name = f'New Add below or generate on first tab.")

    # --- Button Callbacks ---
    def _del_theme_ Theme {len(current)+1}'; current.append({'theme': name, 'sub_themes': ['New Sub-cb(idx): current = st.session_state.get(themes_data_state_key, []); del current[idx]; st.session_state[themes_data_state_key] = current; st.toasttheme 1'], 'description': ''}); st.session_state[themes_data_state_key] =(f"Theme {idx+1} removed."); st.rerun()
    def _del_sub_cb(t_idx, s_idx): current = st.session_state.get(themes_data_state current; st.toast(f"'{name}' added."); st.rerun()

    # --- Render Themes ---
    num_themes_render = len(st.session_state.get(themes_data_state_key, []))
    for i in range(num_themes_render):
        current_list = st.session_state.get(themes_data_state_key, [])
        if i >= len(current_list): logging_key, []); item = current[t_idx]; subs = item.get('sub_themes', []); del subs[s_idx]; item['sub_themes'] = subs; st.session_state[themes_data_state_key] = current; st.toast(f"Sub-theme removed from Theme {t_idx.warning(f"Theme idx {i} out of bounds."); break
        item = current_list[i];
        if not isinstance(item, dict): logging.warning(f"Item {i} not dict."); continue
        t_key = f"theme_edit_{themes_data_state_key}_{i}"; s+1}."); st.rerun()
    def _add_sub_cb(t_idx): current = st.session_state.get(themes_data_state_key, []); item = current[t__base = f"sub_edit_{themes_data_state_key}_{i}"; del_t = f"del_theme_{themes_data_state_key}_{i}"; add_s = f"add_subidx]; item.setdefault('sub_themes', []).append(""); st.session_state[themes_data_state_key] = current; st.toast(f"New sub-theme added to Theme {t_idx+1}."); st.rerun()
    def _add_theme_cb(): current = st.session_state._{themes_data_state_key}_{i}"
        with st.container():
            st.markdown("---", unsafe_allow_html=True); cols_h = st.columns([0.8, 0.2get(themes_data_state_key, []); name = f'New Theme {len(current)+1}'; current.append({'theme': name, 'sub_themes': ['New Sub-theme 1'], 'description': ''});])
            orig_label = item.get('theme', f'Theme {i+1}'); orig_subs = item.get('sub_themes', []); orig_desc = item.get('description', '')
            with cols_h[0 st.session_state[themes_data_state_key] = current; st.toast(f"'{name]: st.text_input(f"**Theme {i+1}**", value=orig_label,}' added."); st.rerun()

    # --- Render Themes ---
    num_themes_render = len( key=t_key); st.caption(f"Desc: {orig_desc or '(N/A)'st.session_state.get(themes_data_state_key, []))
    for i in range}") # Use key for state management
            with cols_h[1]: st.markdown("<div style='margin(num_themes_render):
        current_list = st.session_state.get(themes_data-top:28px;'></div>",True); st.button("🗑️", key=del_t,_state_key, [])
        if i >= len(current_list): logging.warning(f"Theme idx {i} out of bounds."); break
        item = current_list[i];
        if not isinstance(item help=f"Del Theme {i+1}", on_click=_del_theme_cb, args=(i,))
            st.write("**Sub-themes:**"); sub_cols = st.columns(4)
            , dict): logging.warning(f"Item {i} not dict."); continue
        t_key = fif not isinstance(orig_subs, list): orig_subs = []
            num_subs = len(orig_subs)
            for j in range(num_subs):
                 sub_val = orig_subs["theme_edit_{themes_data_state_key}_{i}"; s_base = f"sub_editj]; s_key = f"{s_base}_{j}"; del_s = f"del_sub_{themes_data_{themes_data_state_key}_{i}"; del_t = f"del_theme_{themes_data_state_key}_{i}"; add_s = f"add_sub_{themes_data_state_key_state_key}_{i}_{j}"; col_idx = j % len(sub_cols)
                 with sub_}_{i}"
        with st.container():
            st.markdown("---", unsafe_allow_html=True); cols_h = st.columns([0.8, 0.2])
            orig_label = itemcols[col_idx]: st.text_input(f"Sub {j+1}", value=sub_val, key=s_key, label_visibility="collapsed"); st.button("✖", key=del_s, help=f".get('theme', f'Theme {i+1}'); orig_subs = item.get('sub_themes', []);Del sub {j+1}", on_click=_del_sub_cb, args=(i,j))
            st.button("➕ Add Sub-theme", key=add_s, on_click=_add_sub orig_desc = item.get('description', '')
            with cols_h[0]: st.text_input(f"**Theme {i+1}**", value=orig_label, key=t_key);_cb, args=(i,))
    st.markdown("---"); st.button("➕ Add New Theme", key=f"add_theme_btn_{themes_data_state_key}", on_click=_add_theme_ st.caption(f"Desc: {orig_desc or '(N/A)'}") # Use key for state management
            with cols_h[1]: st.markdown("<div style='margin-top:28px;'></div>cb)
    st.markdown("---")

    # --- Save Button Logic ---
    if st.button("💾 Save ALL Refined Themes", key=f"save_themes_btn_{themes_data_state_",True); st.button("🗑️", key=del_t, help=f"Del Theme {i+1}", on_click=_del_theme_cb, args=(i,))
            st.write("**Subkey}", type="primary"):
        updated_list = []; num_themes_state = len(st.session_state.get(themes_data_state_key, [])); all_ok = True
        for i-themes:**"); sub_cols = st.columns(4)
            if not isinstance(orig_subs, in range(num_themes_state): # Iterate based on state length *before* potential deletes
            t_key = list): orig_subs = []
            num_subs = len(orig_subs)
            for j in range(num_subs):
                 sub_val = orig_subs[j]; s_key = f"{ f"theme_edit_{themes_data_state_key}_{i}"; s_base = f"sub_edit_{themess_base}_{j}"; del_s = f"del_sub_{themes_data_state_key}_{i}_{j}"; col_idx = j % len(sub_cols)
                 with sub_cols[col_data_state_key}_{i}"
            if t_key in st.session_state: # Check if widget exists
                theme_label = st.session_state[t_key].strip() or f"Theme {i_idx]: st.text_input(f"Sub {j+1}", value=sub_val, key+1}" # Read value using key
                orig_desc = ""; current_list_save = st.session_state=s_key, label_visibility="collapsed"); st.button("✖", key=del_s, help=f".get(themes_data_state_key, [])
                if i < len(current_list_save) and isinstance(current_list_save[i], dict): orig_desc = current_list_save[Del sub {j+1}", on_click=_del_sub_cb, args=(i,j))
            st.button("➕ Add Sub-theme", key=add_s, on_click=_add_subi].get('description', '')
                saved_subs = []; num_subs_exp = 0
                if i < len(current_list_save) and isinstance(current_list_save[i], dict):_cb, args=(i,))
    st.markdown("---"); st.button("➕ Add New Theme", key=f"add_theme_btn_{themes_data_state_key}", on_click=_add_theme_ sub_list_state = current_list_save[i].get('sub_themes', []); num_subs_exp = len(sub_list_state) if isinstance(sub_list_state, list) else cb)
    st.markdown("---")

    # --- Save Button Logic ---
    if st.button("💾 Save ALL Refined Themes", key=f"save_themes_btn_{themes_data_state_0
                for j in range(num_subs_exp):
                     s_key = f"{s_base}_{j}"
                     if s_key in st.session_state: sub_label = st.sessionkey}", type="primary"):
        updated_list = []; num_themes_state = len(st.session_state.get(themes_data_state_key, [])); all_ok = True
        for i in_state[s_key].strip(); saved_subs.append(sub_label) if sub_label else range(num_themes_state): # Iterate based on state length *before* potential deletes
            t_key = f None # Read value using key
                     else: logging.warning(f"Sub key {s_key} miss during save"theme_edit_{themes_data_state_key}_{i}"; s_base = f"sub_edit."); all_ok = False
                updated_list.append({'theme': theme_label, 'sub_themes_{themes_data_state_key}_{i}"
            if t_key in st.session_state:': [s for s in saved_subs if s], 'description': orig_desc}) # Ensure only non- # Check if widget exists
                theme_label = st.session_state[t_key].strip() orempty saved
            else: logging.warning(f"Theme key {t_key} miss during save."); all_ok = f"Theme {i+1}" # Read value using key
                orig_desc = ""; current_list_ False
        st.session_state[themes_data_state_key] = updated_list # Update state with collectedsave = st.session_state.get(themes_data_state_key, [])
                if i < len(current values
        if all_ok: st.success("Themes saved."); st.toast("Themes updated!", icon="_list_save) and isinstance(current_list_save[i], dict): orig_desc = current_✏️")
        else: st.warning("Themes saved, but review structure below (some fields might be lost).");list_save[i].get('description', '')
                saved_subs = []; num_subs_exp = st.toast("Themes updated (check review).", icon="⚠️")
        st.rerun()

    # --- Display 0
                if i < len(current_list_save) and isinstance(current_list_save[/Load/Save ---
    st.markdown("---"); st.subheader("Current Saved Structure"); current_saved = sti], dict): sub_list_state = current_list_save[i].get('sub_themes',.session_state.get(themes_data_state_key, [])
    if current_saved:
 []); num_subs_exp = len(sub_list_state) if isinstance(sub_list_state,         try: json_str = json.dumps(current_saved, indent=2); st.download_button list) else 0
                for j in range(num_subs_exp):
                     s_key =(label="💾 Download Structure (JSON)", data=json_str, file_name=f"themer_structure_{ f"{s_base}_{j}"
                     if s_key in st.session_state: sub_label = st.session_state[s_key].strip(); saved_subs.append(sub_label) iftime.strftime('%Y%m%d')}.json", mime="application/json", key=f"dl_themes_{themes_data_state_key}")
         except Exception as e: st.error(f" sub_label else None # Read value using key
                     else: logging.warning(f"Sub key {s_key}Download error: {e}")
         for idx, item in enumerate(current_saved):
             label = item miss during save."); all_ok = False
                updated_list.append({'theme': theme_label, 'sub_themes': [s for s in saved_subs if s], 'description': orig_desc}) # Ensure only non-.get('theme', f'Theme {idx+1}'); desc = item.get('description', 'N/A'); subs = item.get('sub_themes', [])
             with st.expander(f"Themeempty saved
            else: logging.warning(f"Theme key {t_key} miss during save."); all_ok = {idx+1}: {label}", expanded=False): st.caption(f"Desc: {desc}"); st.write("** False
        st.session_state[themes_data_state_key] = updated_list # Update state with collected values
        if all_ok: st.success("Themes saved."); st.toast("Themes updated!", icon="Sub-themes:**"); st.code(json.dumps(subs, indent=2), language='json') if✏️")
        else: st.warning("Themes saved, but review structure below (some fields might be lost). subs else st.caption("(None)")
    else: st.info("No themes saved.")
    st.markdown("---"); st.subheader("Load Structure")
    up_file = st.file_uploader("⬆️ Upload Structure ("); st.toast("Themes updated (check review).", icon="⚠️")
        st.rerun()

    # --- Display/Load/Save ---
    st.markdown("---"); st.subheader("Current Saved Structure"); current_saved =JSON)", type=['json'], key=UPLOADED_THEME_FILE_WIDGET_KEY) # Use config key
    if up_file:
        try:
            loaded = json.load(up_file st.session_state.get(themes_data_state_key, [])
    if current_saved:); valid = isinstance(loaded, list); processed = []
            if valid:
                for idx, item in enumerate
         try: json_str = json.dumps(current_saved, indent=2); st.download_button((loaded):
                    if not (isinstance(item, dict) and 'theme' in item and 'sub_themes' in item): st.error(f"Invalid item {idx}: Missing keys."); valid = False; break
label="💾 Download Structure (JSON)", data=json_str, file_name=f"themer_structure                    if not isinstance(item.get('theme'), str) or not isinstance(item.get('sub_themes'), list):_{time.strftime('%Y%m%d')}.json", mime="application/json", key=f"dl_themes_{themes_data_state_key}")
         except Exception as e: st.error(f"Download error: {e}")
         for idx, item in enumerate(current_saved):
             label = st.error(f"Invalid item {idx}: Wrong types."); valid = False; break
                    proc = {}; proc['theme'] = str(item.get('theme', '')).strip(); proc['description'] = str( item.get('theme', f'Theme {idx+1}'); desc = item.get('description', 'N/A'); subs = item.get('sub_themes', [])
             with st.expander(f"item.get('description', '')).strip(); proc['sub_themes'] = [str(s).strip() for s in item.get('sub_themes', []) if str(s).strip()]
                    if not procTheme {idx+1}: {label}", expanded=False): st.caption(f"Desc: {desc}"); st.write("**Sub-themes:**"); st.code(json.dumps(subs, indent=2), language='json')['theme']: st.error(f"Invalid item {idx}: Empty theme."); valid = False; break
                    processed.append(proc)
            if valid and processed: st.session_state[themes_data_state_key] = processed; st.success(f"Loaded {len(processed)} themes."); st.toast("Themes loaded!", icon if subs else st.caption("(None)")
    else: st.info("No themes saved.")
    st.markdown("---"); st.subheader("Load Structure")
    up_file = st.file_uploader("⬆️ Upload Structure (="⬆️"); st.rerun()
            elif not valid: logging.warning(f"Invalid theme structure uploaded: {JSON)", type=['json'], key=UPLOADED_THEME_FILE_WIDGET_KEY) # Use config key
    if up_file:
        try:
            loaded = json.load(up_fileup_file.name}")
            else: st.warning("Uploaded empty list."); st.session_state[themes_data_state_key] = []; st.rerun()
        except json.JSONDecodeError: st.error); valid = isinstance(loaded, list); processed = []
            if valid:
                for idx, item in enumerate(loaded):
                    if not (isinstance(item, dict) and 'theme' in item and 'sub("Failed JSON parse."); logging.error(f"JSONDecodeError loading theme: {up_file.name}")
        except Exception as e: st.error(f"Load error: {e}"); logging.exception(f"Error_themes' in item): st.error(f"Invalid item {idx}: Missing keys."); valid = False; loading theme structure: {up_file.name}")


# --- Assignment Results Editor (REVERTED VALIDATION START break
                    if not isinstance(item.get('theme'), str) or not isinstance(item.get('sub) ---
def display_assignment_results_editable(df_state_key=ASSIGNMENT_DF_KEY_themes'), list): st.error(f"Invalid item {idx}: Wrong types."); valid = False; break, themes_state_key=EDITED_THEMES_KEY):
    """
    Displays the theme assignment
                    proc = {}; proc['theme'] = str(item.get('theme', '')).strip(); proc['description'] = str(item.get('description', '')).strip(); proc['sub_themes'] = [str(s). results in an editable st.data_editor.
    Includes robust checks to prevent error loops by reverting initial validation.
    """
    # --- Get Data and Validate (REVERTED LOGIC) ---
    if df_statestrip() for s in item.get('sub_themes', []) if str(s).strip()]
                    if not proc['theme']: st.error(f"Invalid item {idx}: Empty theme."); valid = False; break_key not in st.session_state or \
       not isinstance(st.session_state.get(df_state
                    processed.append(proc)
            if valid and processed: st.session_state[themes_data_state__key), pd.DataFrame) or \
       st.session_state.get(df_state_key, pd.key] = processed; st.success(f"Loaded {len(processed)} themes."); st.toast("Themes loaded!", icon="⬆️"); st.rerun()
            elif not valid: logging.warning(f"Invalid theme structureDataFrame()).empty:
        # If key missing, not DF, or DF empty, just show info and exit.
        st.info("No assignment results to display/edit. Run assignment first.")
        return # Exit function uploaded: {up_file.name}")
            else: st.warning("Uploaded empty list."); st.session gracefully

    # If we reach here, results_df_from_state should be a non-empty DataFrame
    results_df_from_state = st.session_state[df_state_key]
    #_state[themes_data_state_key] = []; st.rerun()
        except json.JSONDecodeError: st.error("Failed JSON parse."); logging.error(f"JSONDecodeError loading theme: { --- End of Reverted Validation ---

    # Work on a copy for cleaning/display
    results_df = results_dfup_file.name}")
        except Exception as e: st.error(f"Load error: {e_from_state.copy()

    st.info("Manually review/edit assignments below. Click 'Save Manual Changes}"); logging.exception(f"Error loading theme structure: {up_file.name}")


# --- Assignment Results' to update.")

    # --- DataFrame Preparation (Clean the COPY before config/editor) ---
    required_cols Editor (REVERTED INITIAL VALIDATION) ---
def display_assignment_results_editable(df_state_key= = ["response", "assigned_theme", "assigned_sub_theme", "assignment_confidence"]
    ifASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    """ not all(col in results_df.columns for col in required_cols):
        missing_cols = [
    Displays the theme assignment results in an editable st.data_editor.
    Includes robust checks to prevent error loops by reverting initial validation.
    """
    # --- Get Data and Validate (REVERTED LOGIC) ---
col for col in required_cols if col not in results_df.columns]
        st.error(f"Assignment DF missing required columns: {', '.join(missing_cols)}. Cannot display editor.")
        logging.error(    if df_state_key not in st.session_state or \
       not isinstance(st.session_state.get(df_state_key), pd.DataFrame) or \
       st.session_statef"Themer Assignment DF missing cols: {missing_cols}")
        return

    try:
        # Clean the DataFrame copy
        results_df['assigned_theme'] = results_df['assigned_theme'].fillna("Uncategor.get(df_state_key, pd.DataFrame()).empty:
        # If key missing, not DF, orized").astype(str)
        results_df['assigned_sub_theme'] = results_df['assigned_sub_theme'].fillna("N/A").astype(str)
        results_df['assignment_confidence DF empty, just show info and exit gracefully.
        st.info("No assignment results to display/edit. Run assignment first.")
        return # Exit function gracefully

    # If we reach here, results_df_from_state should'] = results_df['assignment_confidence'].fillna("Low").astype(str)
        results_df[' be a non-empty DataFrame
    results_df_from_state = st.session_state[df_response'] = results_df['response'].fillna("").astype(str)
        # Enforce 'Uncategorstate_key]
    # --- End of Reverted Validation ---

    # Work on a copy for cleaning/ized' rule *on the DataFrame being passed to editor*
        results_df.loc[results_df['display
    results_df = results_df_from_state.copy()

    st.info("Manassigned_theme'] == "Uncategorized", 'assigned_sub_theme'] = "N/A"
ually review/edit assignments below. Click 'Save Manual Changes' to update.")

    # --- DataFrame Preparation (Clean    except Exception as e:
        st.error(f"Failed preparing data for editor: {e}")
        logging. the COPY before config/editor) ---
    required_cols = ["response", "assigned_theme", "assigned_error(f"Themer: Error cleaning state DF for data editor: {e}")
        return

    # --- Preparesub_theme", "assignment_confidence"]
    if not all(col in results_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in results Theme Options (Based on cleaned DataFrame and structure) ---
    theme_structure = st.session_state.get(themes_state_key, [])
    defined_theme_labels = set()
    if isinstance(theme_df.columns]
        st.error(f"Assignment DF missing required columns: {', '.join(missing__structure, list) and theme_structure:
        defined_theme_labels = set(str(t.get('theme', '')).strip() for t in theme_structure if t.get('theme') and str(cols)}. Cannot display editor.")
        logging.error(f"Themer Assignment DF missing cols: {missing_cols}")
        return

    try:
        # Clean the DataFrame copy
        results_df['assigned_theme'] = results_df['assigned_theme'].fillna("Uncategorized").astype(str)
        results_df['t.get('theme','')).strip())
    else: st.warning("Theme structure missing/invalid. Themeassigned_sub_theme'] = results_df['assigned_sub_theme'].fillna("N/A").astype(str)
        results_df['assignment_confidence'] = results_df['assignment_confidence'].fillna(" dropdown options may be limited.")
    try: themes_in_data = set(results_df['assigned_theme'].unique()) # Use cleaned data
    except Exception as e: st.error(f"Error getting themesLow").astype(str)
        results_df['response'] = results_df['response'].fillna(""). from DF: {e}"); themes_in_data = set()
    all_theme_options_set = defined_theme_labels | themes_in_data | {"Uncategorized"}
    theme_options = sorted([opt for opt in all_theme_options_set if pd.notna(opt) and str(optastype(str)
        # Enforce 'Uncategorized' rule *on the DataFrame being passed to editor*
        results_df.loc[results_df['assigned_theme'] == "Uncategorized", 'assigned_sub_theme'] = "N/A"
    except Exception as e:
        st.error().strip()])
    if not theme_options: theme_options = ["Uncategorized"]

    # Ensure themes in the dataframe are actually in the options list before passing to editor
    results_df['assigned_theme'] = results_df['assigned_theme'].apply(lambda x: x if x in theme_options else "Unf"Failed preparing data for editor: {e}")
        logging.error(f"Themer: Error cleaning state DF for data editor: {e}")
        return

    # --- Prepare Theme Options (Based on cleaned DataFrame andcategorized")
    # Re-apply Uncategorized rule after potential reset above
    results_df.loc[results_df['assigned_theme'] == "Uncategorized", 'assigned_sub_theme'] = "N structure) ---
    theme_structure = st.session_state.get(themes_state_key, [])
    defined_theme_labels = set()
    if isinstance(theme_structure, list) and theme_/A"


    # --- Define Column Configurations ---
    column_config = {
        "response": st.column_config.TextColumn("Response", width="large", disabled=True),
        "assigned_theme": ststructure:
        defined_theme_labels = set(str(t.get('theme', '')).strip() for t in theme_structure if t.get('theme') and str(t.get('theme','')).strip.column_config.SelectboxColumn("Assigned Theme", width="medium", options=theme_options, required=True),
        "assigned_sub_theme": st.column_config.TextColumn("Assigned Sub())
    else: st.warning("Theme structure missing/invalid. Theme dropdown options may be limited.")
    try: themes-theme", width="medium", required=True),
        "assignment_confidence": st.column_config._in_data = set(results_df['assigned_theme'].unique()) # Use cleaned data
    except Exception as e: st.error(f"Error getting themes from DF: {e}"); themes_in_data = setSelectboxColumn("Confidence", width="small", options=["High", "Medium", "Low"], required=True),
    }

    # --- >>> REMOVED DEBUGGING OUTPUT <<< ---


    # --- Display Data Editor ---
    st()
    all_theme_options_set = defined_theme_labels | themes_in_data | {"Uncategorized"}
    theme_options = sorted([opt for opt in all_theme_options_set if.markdown("#### Edit Assignments")
    edited_df = None # Initialize edited_df to None
    try:
 pd.notna(opt) and str(opt).strip()])
    if not theme_options: theme_        edited_df = st.data_editor(
            results_df, # Pass the cleaned DataFrame
            options = ["Uncategorized"]

    # Ensure themes in the dataframe are actually in the options list before passing to editor
    key=ASSIGNMENT_EDITOR_WIDGET_KEY,
            use_container_width=True, column_config=column_config,
            column_order=required_cols, num_rows="fixed", hideresults_df['assigned_theme'] = results_df['assigned_theme'].apply(lambda x: x if_index=True,
            disabled=["response"]
        )
    except Exception as editor_err:
 x in theme_options else "Uncategorized")
    # Re-apply Uncategorized rule after potential reset above
    results_df.loc[results_df['assigned_theme'] == "Uncategorized", 'assigned         st.error(f"Data editor failed to render: {editor_err}. Try refreshing.")
         logging_sub_theme'] = "N/A"


    # --- Define Column Configurations ---
    column_config = {
        "response": st.column_config.TextColumn("Response", width="large", disabled=True),.exception("Themer: st.data_editor failed to render.")
         return # Stop function execution if editor fails
        "assigned_theme": st.column_config.SelectboxColumn("Assigned Theme", width="medium to render

    # --- Save Changes Button ---
    # Check if edited_df was successfully created by the editor
    if edited_df is not None and isinstance(edited_df, pd.DataFrame):
        if st.button("", options=theme_options, required=True),
        "assigned_sub_theme": st.column_config.TextColumn("Assigned Sub-theme", width="medium", required=True),
        "assignment_💾 Save Manual Assignment Changes", key=f"save_manual_assignments_btn_{df_state_key}"):

            # --- Validation Logic ---
            valid = True; validation_warnings = []
            for index, row in editedconfidence": st.column_config.SelectboxColumn("Confidence", width="small", options=["High", "Medium", "Low"], required=True),
    }

    # --- Display Data Editor ---
    st.markdown_df.iterrows():
                theme = row['assigned_theme']; sub_theme = row['assigned_sub("#### Edit Assignments")
    edited_df = None # Initialize edited_df to None
    try:
        edited_theme']
                if not theme or not sub_theme or not row['assignment_confidence']: valid=False_df = st.data_editor(
            results_df, # Pass the cleaned DataFrame
            key=ASSIGNMENT_; validation_warnings.append(f"Row {index+1}: Required fields empty.")
                if theme == "Uncategorized" and sub_theme != "N/A": validation_warnings.append(f"Row {indexEDITOR_WIDGET_KEY,
            use_container_width=True, column_config=column_config,
            column_order=required_cols, num_rows="fixed", hide_index=True,+1}: Uncategorized needs 'N/A' sub."); valid = False

            if valid:
                try:
                    # Ensure index consistency before saving back to state
                    edited_df_copy = edited_df.copy()
            disabled=["response"]
        )
    except Exception as editor_err:
         st.error(f"Data editor failed to render: {editor_err}. Try refreshing.")
         logging.exception("Themer: st
                    original_index = st.session_state[df_state_key].index
                    if len(original_index) == len(edited_df_copy):
                        edited_df_copy.index = original.data_editor failed to render.")
         return # Stop function execution if editor fails to render

    # --- Save Changes Button ---
    if edited_df is not None and isinstance(edited_df, pd.DataFrame):_index
                        st.session_state[df_state_key] = edited_df_copy
                        st.success("Manual assignment changes saved!"); st.toast("Assignments updated!", icon="💾")
                        logging.
        if st.button("💾 Save Manual Assignment Changes", key=f"save_manual_assignments_btn_{df_state_key}"):

            # --- Validation Logic ---
            valid = True; validation_warnings = []
            forinfo(f"Themer: Manual assignments saved '{df_state_key}'.")
                        time.sleep(0.5); st.rerun()
                    else:
                         st.error("Error saving: Row count mismatch after index, row in edited_df.iterrows():
                theme = row['assigned_theme']; sub_theme = editing.")
                         logging.error(f"Themer: Row count mismatch during save. Original: {len(original_index row['assigned_sub_theme']
                if not theme or not sub_theme or not row['assignment_confidence']: valid=False; validation_warnings.append(f"Row {index+1}: Required fields empty."))}, Edited: {len(edited_df_copy)}")

                except Exception as save_err: st.error(f"Error saving changes: {save_err}"); logging.error(f"Themer: Error saving edited DF: {save
                if theme == "Uncategorized" and sub_theme != "N/A": validation_warnings._err}")
            else:
                st.error("Validation Failed! Correct issues below before saving:");
                for iappend(f"Row {index+1}: Uncategorized needs 'N/A' sub."); valid = False, w in enumerate(validation_warnings): st.warning(w) if i<5 else st.warning(f"...{len(validation_warnings)-5} more issues."); break
                logging.warning(f"Themer:

            if valid:
                try:
                    # Ensure index consistency before saving back to state
                    edited_df_copy = edited_df.copy()
                    original_index = st.session_state[df_state_key].index
                    if len(original_index) == len(edited_df_copy):
                        edited Manual assignment save validation failed ({len(validation_warnings)} issues).")
    elif edited_df is None:_df_copy.index = original_index
                        st.session_state[df_state_key]
         # This case might be hit if the editor failed to render initially
         st.warning("Data editor did not render correctly. Cannot save changes.")


# Theme Examples Display
def display_theme_examples(df_state_ = edited_df_copy
                        st.success("Manual assignment changes saved!"); st.toast("Assignments updated!", icon="💾")
                        logging.info(f"Themer: Manual assignments saved '{df_state_key}'.key=ASSIGNMENT_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    """
    Allows selecting a theme and shows its description (if available)
    and example responses assigned to it.")
                        time.sleep(0.5); st.rerun()
                    else:
                         st.error("Error saving: Row count mismatch after editing.")
                         logging.error(f"Themer: Row count mismatch during save.
    """
    if df_state_key not in st.session_state or not isinstance(st.session_state.get(df_state_key), pd.DataFrame):
        st.info("Assign themes first.")
 Original: {len(original_index)}, Edited: {len(edited_df_copy)}")

                except Exception as save_err: st.error(f"Error saving changes: {save_err}"); logging.error(f        return
    results_df = st.session_state[df_state_key]
    if results_df.empty or 'assigned_theme' not in results_df.columns:
        st.info(""Themer: Error saving edited DF: {save_err}")
            else:
                st.error("ValidationNo assignment data.")
        return
    theme_structure = st.session_state.get(themes_state Failed! Correct issues below before saving:");
                for i, w in enumerate(validation_warnings): st.warning_key, [])
    theme_map = {}
    if isinstance(theme_structure, list) and theme(w) if i<5 else st.warning(f"...{len(validation_warnings)-5} more issues."); break
                logging.warning(f"Themer: Manual assignment save validation failed ({len(validation_warnings)}_structure:
        theme_map = {str(t.get('theme','')): t for t in theme_structure if t.get('theme')}
    else:
        st.caption("Theme structure missing.")
    try issues).")
    elif edited_df is None:
         st.warning("Data editor did not render correctly:
        valid_themes_in_data = sorted([str(th) for th in results_df['. Cannot save changes.")


# Theme Examples Display
def display_theme_examples(df_state_key=ASSIGNMENTassigned_theme'].unique() if pd.notna(th) and isinstance(th, str) and "Error" not in th])
    except Exception as e:
        st.error(f"Error processing themes:_DF_KEY, themes_state_key=EDITED_THEMES_KEY):
    """Allows selecting a theme and shows descriptions and example responses."""
    if df_state_key not in st.session_state or {e}")
        logging.error(f"Error getting unique themes: {e}")
        valid_themes not isinstance(st.session_state.get(df_state_key), pd.DataFrame):
        st.info_in_data = []
    if not valid_themes_in_data:
        st.warning("("No theme assignment results available to explore.")
        return
    if themes_state_key not in st.No valid themes in results.")
        return
    selected_theme = st.selectbox("Select Theme to Explore:",session_state or not st.session_state.get(themes_state_key):
         st.info valid_themes_in_data, key=f"explore_theme_select_{df_state_key}")
    if selected_theme:
        st.markdown(f"#### Exploring Theme: **{selected_theme}("No theme structure available.")
         return

    results_df = st.session_state[df_state_**")
        theme_details = theme_map.get(selected_theme)
        if theme_detailskey]
    theme_structure = st.session_state[themes_state_key]
    theme_map = {str(t.get('theme','')): t for t in theme_structure if t.get('theme:
            st.markdown("**Description:**")
            description = theme_details.get('description', '').strip()
            st.markdown(f"> {description}" if description else "_(N/A)_")
            st.markdown')}

    try: # Get unique assigned themes from results, excluding errors and ensuring string type
        valid_themes = sorted([ str(th) for th in results_df['assigned_theme'].unique() if pd.notna(("**Defined Sub-themes:**")
            sub_themes = theme_details.get('sub_themes', [])
            if sub_themes and isinstance(sub_themes, list):
                 subs_str = [str(s) for s inth) and isinstance(th, str) and "Error" not in th ])
    except Exception as e: st.error(f"Error processing assigned themes: {e}"); valid_themes = []

    if not valid sub_themes]
                 st.code(f"{', '.join(subs_str)}" if subs_str else "(None)", language=None)
            else: st.caption("(None)")
        elif selected_theme !=_themes: st.warning("No valid themes found in results."); return

    selected_theme = st.selectbox("Select Theme to Explore:", valid_themes, key=f"explore_theme_select_{df_state_ "Uncategorized":
             st.caption(f"Details not found for '{selected_theme}'.")
        st.markdown("**Example Responses (Sample):**")
        try:
            if 'assigned_theme' in results_df.key}") # Unique key

    if selected_theme:
        st.subheader(f"Exploring Theme: {columns:
                 theme_responses_df = results_df[results_df['assigned_theme'] == selectedselected_theme}")
        theme_details = theme_map.get(selected_theme)
        if theme_details:_theme].copy()
            else:
                 st.error("'assigned_theme' column missing.")
                 return
            if not theme_responses_df.empty:
                 num_examples = min(5, len(theme_responses_
            st.markdown("**Description:**"); description = theme_details.get('description', ''); st.markdown(f"> {description}" if description else "_(N/A)_")
            st.markdown("**Defined Sub-themes:**df))
                 examples = theme_responses_df.sample(n=num_examples, random_state=42)
                 for index, row in examples.iterrows():
                     st.markdown("---")
                     response = str"); sub_themes = theme_details.get('sub_themes', [])
            if sub_themes and isinstance(sub_themes, list): subs_str = [str(s) for s in sub_themes]; st(row.get('response', 'N/A'))
                     sub = str(row.get('assigned_sub_theme', 'N/A'))
                     conf = str(row.get('assignment_confidence', '.code(f"{', '.join(subs_str)}" if subs_str else "(None)", language=None)
            else: st.caption("(None defined)")
        elif selected_theme != "Uncategorized": st.captionN/A'))
                     st.markdown(f"> {response}")
                     st.caption(f"Sub-theme: `{sub}` | Confidence: `{conf}`")
            else:
                 st.info(f"No(f"Details not found for '{selected_theme}'.")

        st.markdown("**Example Responses (Sample):**")
        try: # Get Example Responses
            if 'assigned_theme' in results_df.columns: theme_responses_ responses found for '{selected_theme}'.")
        except KeyError as ke:
             st.error(f"Missing column: {ke}")
             logging.error(f"KeyError examples: {ke}")
        except Exception asdf = results_df[results_df['assigned_theme'] == selected_theme].copy()
            else: st.error("Column 'assigned_theme' missing."); theme_responses_df = pd.DataFrame()

 e:
             st.error(f"Error getting examples: {e}")
             logging.exception(f"Error displaying examples for '{selected_theme}'.")
