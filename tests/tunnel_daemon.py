import subprocess
import time
import sys

print("Starting persistent tunnel daemon on port 5173...")
while True:
    try:
        proc = subprocess.Popen(
            ["npx", "-y", "localtunnel", "--port", "5173"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in iter(proc.stdout.readline, ""):
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
    except Exception as e:
        print(f"Tunnel error: {e}", flush=True)
    print("Restarting localtunnel in 2 seconds...", flush=True)
    time.sleep(2)
