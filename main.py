import argparse
import tkinter as tk
from tkinter import messagebox

def run_mode(mode):
    if mode == "haar":
        import haar_mode
        haar_mode.run()
    elif mode == "dlib":
        import dlib_mode
        dlib_mode.run()
    elif mode == "huggingface":
        from hf_model import HuggingFaceDrowsinessDetector
        detector = HuggingFaceDrowsinessDetector()
        detector.run()
    else:
        print("[ERROR] Invalid mode selected")

def gui_mode():
    def start_haar():
        root.destroy()
        run_mode("haar")

    def start_dlib():
        root.destroy()
        run_mode("dlib")

    def start_hf():
        root.destroy()
        run_mode("huggingface")

    root = tk.Tk()
    root.title("Driver Drowsiness Detector")
    root.geometry("300x200")

    label = tk.Label(root, text="Choose Detection Mode", font=("Arial", 14))
    label.pack(pady=15)

    tk.Button(root, text="Haar Cascade", width=20, command=start_haar).pack(pady=5)
    tk.Button(root, text="Dlib (EAR Method)", width=20, command=start_dlib).pack(pady=5)
    tk.Button(root, text="Hugging Face (ViT)", width=20, command=start_hf).pack(pady=5)

    root.mainloop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None,
                        choices=["haar", "dlib", "huggingface"],
                        help="Choose detection mode")
    args = parser.parse_args()

    if args.mode:
        run_mode(args.mode)
    else:
        gui_mode()

if __name__ == "__main__":
    main()
