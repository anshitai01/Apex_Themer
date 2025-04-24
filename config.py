# config.py
"""
Stores configuration constants for the AI Themer application.
Uses the Phronesis Apex theme colors and fonts.
"""

from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Theme Configuration (Phronesis Apex Theme) ---
PRIMARY_ACCENT_COLOR = "#cd669b"  # Pinkish accent
PRIMARY_ACCENT_COLOR_RGB = "205, 102, 155" # RGB for the pinkish color

CARD_TEXT_COLOR = "#a9b4d2"       # Medium text (used for card descriptions, general body)
CARD_TITLE_TEXT_COLOR = PRIMARY_ACCENT_COLOR # Use Primary Accent for Card Title (or choose another)

MAIN_TITLE_COLOR = "#f0f8ff"      # Light text (used for main titles)
BODY_TEXT_COLOR = "#ffff"       # Medium text (duplicate of CARD_TEXT_COLOR, used generally)
SUBTITLE_COLOR = "#8b98b8"        # Darker medium text (used for captions, footer)

MAIN_BACKGROUND_COLOR = "#0b132b" # Dark blue background
CARD_BACKGROUND_COLOR = "#1c2541" # Slightly lighter card background
SIDEBAR_BACKGROUND_COLOR = "#121a35" # Slightly different dark shade for sidebar
HOVER_GLOW_COLOR = f"rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.4)" # Glow effect color

# Variables added previously to fix ImportError
CONTAINER_BG_COLOR = "rgba(11, 19, 43, 0.0)" # Transparent, rely on body background
CONTAINER_BORDER_RADIUS = "15px"

# Specific UI Element Colors (Adapt as needed)
INPUT_BG_COLOR = "#1c2541"        # Background for text inputs, select boxes
INPUT_BORDER_COLOR = "#3a506b"    # Border for inputs
INPUT_TEXT_COLOR = BODY_TEXT_COLOR

BUTTON_PRIMARY_BG = PRIMARY_ACCENT_COLOR
BUTTON_PRIMARY_TEXT = "#FFFFFF"
BUTTON_SECONDARY_BG = "transparent"
BUTTON_SECONDARY_TEXT = PRIMARY_ACCENT_COLOR
BUTTON_SECONDARY_BORDER = PRIMARY_ACCENT_COLOR

DATAFRAME_HEADER_BG = "#1c2541" # Match card background
DATAFRAME_HEADER_TEXT = MAIN_TITLE_COLOR
DATAFRAME_CELL_BG = MAIN_BACKGROUND_COLOR # Match main background
DATAFRAME_CELL_TEXT = BODY_TEXT_COLOR

# Chart / Alert Colors (Used by latest styling.py)
CHART_SUCCESS_COLOR = "#2ecc71" # Greenish
CHART_WARNING_COLOR = "#f39c12" # Orangish
CHART_ERROR_COLOR = "#e74c3c" # Reddish

# --- Font Families ---
TITLE_FONT = "'Montserrat', sans-serif" # Or 'Orbitron'
BODY_FONT = "'Roboto', sans-serif"
CARD_TITLE_FONT = "'Montserrat', sans-serif" # Often same as TITLE_FONT

# Safety Settings (Unchanged)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- App State Keys (Unchanged) ---
APP_PREFIX = "theming_"
API_KEY_STATE_KEY = f"{APP_PREFIX}api_key"
INITIALIZED_STATE_KEY = f"{APP_PREFIX}initialized"
API_KEY_SOURCE_STATE_KEY = f"{APP_PREFIX}api_key_source"
SURVEY_QUESTION_KEY = f"{APP_PREFIX}survey_question"
RESPONSES_RAW_KEY = f"{APP_PREFIX}responses_raw"
UPLOADED_DF_KEY = f"{APP_PREFIX}uploaded_df"
SELECTED_COLUMN_KEY = f"{APP_PREFIX}selected_column"
INPUT_METHOD_KEY = f"{APP_PREFIX}input_method"
CURRENT_FILE_NAME_KEY = f"{APP_PREFIX}current_file_name"
RESPONSES_INPUT_AREA_VAL_KEY = f"{APP_PREFIX}responses_input_area_val"
SELECTED_COLUMN_IDX_KEY = f"{APP_PREFIX}selected_column_idx"
UPLOADED_THEME_FILE_WIDGET_KEY = f"{APP_PREFIX}uploader_load_themes_json_widget"
GENERATED_THEMES_KEY = f"{APP_PREFIX}generated_themes"
EDITED_THEMES_KEY = f"{APP_PREFIX}edited_themes"
ASSIGNMENT_DF_KEY = f"{APP_PREFIX}assignment_df"
ASSIGNMENT_EDITOR_WIDGET_KEY = f"{APP_PREFIX}assignment_editor_widget_main"
AI_QA_HISTORY_KEY = f"{APP_PREFIX}ai_qa_history"
GEN_TEMP_KEY = f"{APP_PREFIX}gen_temp"
GEN_TOP_K_KEY = f"{APP_PREFIX}gen_top_k"
GEN_TOP_P_KEY = f"{APP_PREFIX}gen_top_p"
GEN_MAX_TOKENS_KEY = f"{APP_PREFIX}gen_max_tokens"
BATCH_SIZE_KEY = f"{APP_PREFIX}batch_size"
SAMPLE_SIZE_KEY = f"{APP_PREFIX}sample_size"
RESET_BUTTON_KEY = f"{APP_PREFIX}reset_all_btn"
INIT_BUTTON_KEY = f"{APP_PREFIX}init_button_manual"

# --- Logo Path (Relative to main_app.py location) ---
LOGO_FILENAME = "apexlogo.png" # Logo file name