"""Launch HYPERION dashboard."""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/dashboard/app.py",
            "--server.headless",
            "true",
        ]
    )
