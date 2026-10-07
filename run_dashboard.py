#!/usr/bin/env python3
"""
run_dashboard.py

Launcher for the Google VLA Robot Arm Autonomous Manipulation Studio.
Starts the FastAPI simulation server with real-time 60 FPS WebSocket streaming
and automatically launches the interactive Three.js 3D web dashboard.
"""

import os
import sys
import webbrowser
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import uvicorn
except ImportError:
    print("[!] Error: 'uvicorn' is required. Run: pip install uvicorn")
    sys.exit(1)

def open_browser_delayed(url: str, delay: float = 1.2):
    """Opens default browser after server initializes."""
    time.sleep(delay)
    print(f"[*] Opening {url} in your web browser...")
    webbrowser.open(url)

def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("========================================================================")
    print("[*] Google VLA Robot Arm Autonomous Manipulation Studio")
    print("    Combining Google Vision-Language-Action, 3D Semantic Spatial Mapping,")
    print("    Tabletop Frontier Exploration, and Rapier3D Kinematics")
    print(f"    Interactive Dashboard: {url}")
    print("========================================================================")

    # Launch browser in a background daemon thread
    threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    # Start Uvicorn server
    uvicorn.run("dashboard.server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
