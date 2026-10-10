#!/usr/bin/env python3
"""
run_dashboard.py

Launcher for the AERO Autonomous Manipulation Studio.
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
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    display_url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"

    print("========================================================================")
    print("[*] AERO: Agentic Execution for Robotic Operations")
    print("    Combining Vision-Language-Action, 3D Semantic Spatial Mapping,")
    print("    Tabletop Frontier Exploration, and Rapier3D Kinematics")
    print(f"    Interactive Dashboard: {display_url}")
    print("========================================================================")

    # Launch browser only when running in local desktop session (not cloud containers)
    is_cloud = bool(os.environ.get("RENDER") or os.environ.get("RAILWAY_STATIC_URL") or os.environ.get("DYNO"))
    if not is_cloud:
        threading.Thread(target=open_browser_delayed, args=(display_url,), daemon=True).start()

    # Start Uvicorn server
    uvicorn.run("dashboard.server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
