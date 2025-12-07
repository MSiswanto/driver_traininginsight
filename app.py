# DEBUG WRAPPER FOR STREAMLIT
# This file temporarily replaces your main app so we can see all runtime errors.
import traceback
import streamlit as st

st.set_page_config(page_title="DEBUG MODE — showing traceback", layout="wide")

st.title("🛠 DEBUG MODE — Showing Import & Runtime Errors")

try:
    import importlib

    st.write("Trying to import app_original.py ...")

    app_mod = importlib.import_module("app_original")

    st.success("Imported app_original successfully.")

    if hasattr(app_mod, "main"):
        st.info("Running main() of app_original.py ...")
        try:
            app_mod.main()
        except Exception:
            st.error("❌ Error while running main():")
            st.code(traceback.format_exc())
    else:
        st.warning("app_original has no main() function.")

except Exception:
    st.error("❌ Import error in app_original.py or its modules:")
    st.code(traceback.format_exc())
