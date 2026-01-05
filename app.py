import streamlit as st
import logging
from ui_components import init_session_state, setup_sidebar
from tabs import tab_data_upload, tab_optimization, tab_export, tab_compare_actuals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tab functions are now in tabs.py module

# All tab functions have been moved to tabs.py

def main() -> None:
    """Main application entry point."""
    st.set_page_config(layout="wide", page_title="Warehouse Optimizer", page_icon="🚛")
    init_session_state()
    services = setup_sidebar()
    st.title("🏭 Warehouse Location & Route Optimizer")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Data Upload",
        "2. Optimization",
        "3. Export",
        "4. Compare vs. Actuals"
    ])

    with tab1:
        tab_data_upload(services)
    with tab2:
        tab_optimization(services)
    with tab3:
        tab_export()
    with tab4:
        tab_compare_actuals(services)


if __name__ == "__main__":
    main()