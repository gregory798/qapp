import threading
import uvicorn
import subprocess

# Lancer le backend FastAPI dans un thread
def run_backend():
    uvicorn.run("api.main:app", host="0.0.0.0", port=7861, log_level="info")

threading.Thread(target=run_backend, daemon=True).start()

# Lancer Streamlit normalement
subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.port=7860"])