
print("Hello from test_simple.py!")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
try:
    import utils.knowledge_essentials
    print("Successfully imported utils.knowledge_essentials!")
except Exception as e:
    print(f"Failed to import utils.knowledge_essentials: {e}")
    import traceback
    print("Stack trace:")
    traceback.print_exc()
