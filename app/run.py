
import subprocess
import sys

api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app"])
ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui.py"])
try:
    ui.wait()
except KeyboardInterrupt:
    pass
finally:
    api.terminate()
    ui.terminate()