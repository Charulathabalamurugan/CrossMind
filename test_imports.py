
import sys
print("Python path:", sys.path)
try:
    import streamlit
    print("Streamlit imported successfully!")
except ImportError as e:
    print(f"Streamlit import error: {e}")
    print(f"Looking for streamlit in: {sys.modules}")

try:
    import utils.knowledge_essentials
    print("utils.knowledge_essentials imported successfully!")
except ImportError as e:
    print(f"utils.knowledge_essentials import error: {e}")
