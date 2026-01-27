import streamlit as st
import logging
from ui_components import init_session_state, setup_sidebar
from ui.tabs import tab_data_upload, tab_optimization, tab_static_routes, tab_compare_actuals, tab_export

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tab functions are now in tabs.py module

# All tab functions have been moved to tabs.py

def main() -> None:
    """Main application entry point."""
    st.set_page_config(layout="wide", page_title="Warehouse Optimizer")

    # Hide the "Link" anchor button on headers
    st.markdown("""
        <style>
        [data-testid="stHeaderAction"] {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

    init_session_state()
    services = setup_sidebar()
    st.title("Warehouse Location & Route Optimizer")

    tab1, tab2, tab3, tab4 = st.tabs([ #add tab3 and update tabs order to disable static routes tab
        "Data Upload",
        "Optimization",
#        "Static Routes",    #uncomment to enable static routes tab
        "Comparison",
        "Export"
    ])

    with tab1:
        tab_data_upload(services)
    with tab2:
        tab_optimization(services)
#    with tab3: #uncomment to enable static routes tab
#        tab_static_routes(services) #uncomment to enable static routes tab
    with tab3:
        tab_compare_actuals(services)
    with tab4:
        tab_export()


if __name__ == "__main__":
    main()