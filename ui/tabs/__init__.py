"""
UI tab modules for the Warehouse Optimizer Streamlit app.
"""
from .data_upload import tab_data_upload
from .optimization import tab_optimization
from .comparison import tab_compare_actuals
from .export import tab_export

__all__ = [
    'tab_data_upload',
    'tab_optimization',
    'tab_compare_actuals',
    'tab_export'
]
