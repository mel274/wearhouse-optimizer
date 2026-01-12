"""
Export tab for the Warehouse Optimizer Streamlit app.
Handles exporting optimization results to Excel format.
"""
import streamlit as st
import pandas as pd
from io import BytesIO
import traceback


def tab_export() -> None:
    """Handle the export results tab functionality."""
    st.header("Export")
    if st.session_state.solution and st.session_state.solution['solution_found']:
        try:
            solution = st.session_state.solution

            if st.session_state.data is None:
                st.warning("No customer data available for export.")
                return

            # Build clean route list
            routes_data = []

            # Export only served customers (routes)
            for route_idx, route in enumerate(solution['routes']):
                # Skip empty routes (only depot)
                if len(route) <= 2:
                    continue

                # Process stops in route (skip depot at start and end)
                for stop_seq, node_idx in enumerate(route[1:-1], start=1):
                    if node_idx > 0 and node_idx - 1 < len(st.session_state.data):
                        customer = st.session_state.data.iloc[node_idx - 1]
                        routes_data.append({
                            'Route ID': route_idx + 1,
                            'Stop Sequence': stop_seq,
                            'Customer ID': str(customer.get("מס' לקוח", "")),
                            'Customer Name': customer.get('שם לקוח', ''),
                            'Address': customer.get('כתובת', '')
                        })

            if not routes_data:
                st.warning("No routes to export.")
                return

            # Create DataFrame
            routes_df = pd.DataFrame(routes_data)

            # Display preview
            st.dataframe(routes_df, width="stretch")

            # Export to Excel
            output = BytesIO()

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                routes_df.to_excel(writer, index=False, sheet_name='Routes')

            output.seek(0)

            # Download button
            st.download_button(
                "Download Routes (Excel)",
                output.getvalue(),
                "optimization_routes.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-excel'
            )

        except Exception as e:
            st.error(f"Export Error: {str(e)}")
            st.code(traceback.format_exc())
    else:
        st.warning("No solution to export.")
