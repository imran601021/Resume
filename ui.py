import streamlit as st


def apply_custom_styles():
    """Injects custom CSS for the 'Modern SaaS' visual theme."""
    st.markdown("""
    <style>
        /* Overall background */
        .stApp {
            background-color: #0f1117;
        }

        /* Card-style containers for metrics */
        div[data-testid="stMetric"] {
            background-color: #1a1d29;
            border: 1px solid #2a2e3d;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }

        /* Info/warning/success/error boxes get card treatment */
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid #2a2e3d;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #1a1d29;
            border-radius: 12px;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            color: #9ca3af;
        }
        .stTabs [aria-selected="true"] {
            background-color: #6366f1;
            color: white !important;
        }

        /* File uploader and text area as cards */
        div[data-testid="stFileUploader"], div[data-testid="stTextArea"] {
            background-color: #1a1d29;
            border: 1px solid #2a2e3d;
            border-radius: 12px;
            padding: 12px;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #14161f;
            border-right: 1px solid #2a2e3d;
        }

        /* Title accent */
        h1 {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* Buttons */
        .stButton button {
            background-color: #6366f1;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 20px;
            font-weight: 600;
        }
        .stButton button:hover {
            background-color: #4f46e5;
        }

        /* Code chips for detected skills */
        code {
            background-color: #1a1d29 !important;
            border: 1px solid #6366f1 !important;
            border-radius: 6px !important;
            color: #a5b4fc !important;
        }
    </style>
    """, unsafe_allow_html=True)
