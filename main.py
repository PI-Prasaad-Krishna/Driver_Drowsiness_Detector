import argparse
import subprocess
import sys

def run_mode(mode):
    if mode == "haar":
        subprocess.run([sys.executable, "haar_mode.py"])
    elif mode == "dlib":
        subprocess.run([sys.executable, "dlib_mode.py"])
    else:
        print("❌ Invalid mode. Use --mode haar or --mode dlib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detector")
    parser.add_argument("--mode", type=str, required=True, help="haar or dlib")
    args = parser.parse_args()
    run_mode(args.mode)
