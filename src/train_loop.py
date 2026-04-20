import os
import subprocess
import time

INTERVAL_SECONDS = int(os.getenv("TRAIN_INTERVAL_SECONDS", "300"))
PYTHON_CMD = os.getenv("PYTHON_CMD", "python")


def main():
    print(f"Starting training loop with interval {INTERVAL_SECONDS}s")
    while True:
        print("[train_loop] Running training job...")
        result = subprocess.run([PYTHON_CMD, "src/training.py"], capture_output=True, text=True)
        print("[train_loop] Training stdout:\n", result.stdout)
        print("[train_loop] Training stderr:\n", result.stderr)
        print(f"[train_loop] Training exit code: {result.returncode}")
        print(f"[train_loop] Sleeping for {INTERVAL_SECONDS} seconds")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
