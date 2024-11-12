"""
Main entry point for the PPG analysis tool.
"""

from src.app import app

if __name__ == "__main__":
    app.run_server(debug=True)
