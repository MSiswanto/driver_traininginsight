# ===========================
# DEBUG MODE (Single File)
# ===========================
import streamlit as st
import traceback
import importlib

st.set_page_config(page_title="DEBUG MODE", layout="wide")
st.title("🛠 DEBUG MODE – Showing errors from app_original.py")

try:
    # Try to load app_original.py
    st.write("Importing app_original.py...")
    app_mod = importlib.import_module("app_original")
    st.success("Imported app_original.py successfully!")

    # Try to run main()
    if hasattr(app_mod, "main"):
        st.write("Running app_original.main() ...")
        try:
            app_mod.main()
        except Exception:
            st.error("❌ Runtime error inside main():")
            st.code(traceback.format_exc())
    else:
        st.warning("app_original.py does NOT have a main() function!")

except Exception:
    st.error("❌ Import error in app_original.py:")
    st.code(traceback.format_exc())
