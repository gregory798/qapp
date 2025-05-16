import threading
import uvicorn
import subprocess

def run_backend():
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)

threading.Thread(target=run_backend, daemon=True).start()

# Lance Streamlit
subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.port=10000"])