#!/usr/bin/env python3
"""
run_dashboard.py

Convenient launcher for the AERO Interactive Web Dashboard.
Starts the FastAPI server with live WebSocket streaming and opens the browser.
"""

import os
import sys
import webbrowser
import threading
import time

try:
    import uvicorn
except ImportError:
    print("Error: 'uvicorn' is required to run the dashboard. Run: pip install uvicorn")
    sys.exit(1)


def open_browser_delayed(url: str, delay: float = 1.2):
    """Opens default web browser after server initializes."""
    time.sleep(delay)
    print(f"Opening {url} in browser...")
    webbrowser.open(url)


def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("================================================================")
    print("⚡ AERO: Agentic Environment for Robotic Operations")
    print(f"Starting Interactive Dashboard at: {url}")
    print("================================================================")

    # Launch browser in a background daemon thread
    threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    # Start Uvicorn server
    uvicorn.run("dashboard.server:app", host=host, port=port, log_level="info")


if __name__ == '__main__':
    main()
