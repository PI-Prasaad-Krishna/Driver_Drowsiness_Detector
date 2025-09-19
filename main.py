import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
import argparse
import sys
import os

class DrowsinessDetectorApp:
    def __init__(self, root):
        self.root = root
        self.setup_styles()
        self.setup_window()
        self.setup_widgets()

    def setup_styles(self):
        # Colors & Fonts
        self.BG_COLOR = "#212121"
        self.BUTTON_COLOR = "#3c3c3c"
        self.BUTTON_HOVER_COLOR = "#0078D7"
        self.TEXT_COLOR = "#FFFFFF"
        self.ACCENT_COLOR = "#00aaff"
        self.title_font = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.button_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.desc_font = tkfont.Font(family="Segoe UI", size=9)

    def setup_window(self):
        self.root.title("Drowsiness Detector")
        self.root.configure(bg=self.BG_COLOR)
        # Center the window
        window_width, window_height = 450, 400
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        
    def setup_widgets(self):
        # --- Main Container ---
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        header_label = tk.Label(main_frame, text="Drowsiness Detection System",
                                 font=self.title_font, bg=self.BG_COLOR, fg=self.ACCENT_COLOR)
        header_label.pack(pady=(0, 25))
        sub_header_label = tk.Label(main_frame, text="Select a detection model to begin",
                                      font=self.desc_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        sub_header_label.pack(pady=(0, 20))

        # --- Button Frame ---
        button_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        button_frame.pack(pady=10)

        # --- Buttons (Text Only) ---
        self.create_mode_button(button_frame, "Haar Cascade", "haar",
                                 "Fast & lightweight, but less accurate. Good for ideal conditions.")
        self.create_mode_button(button_frame, "Dlib (EAR)", "dlib",
                                 "Highly accurate landmark detection. Best for reliability.")
        self.create_mode_button(button_frame, "Hugging Face", "huggingface",
                                 "Advanced deep learning model. High accuracy, needs powerful hardware.")

        # --- Footer / Status Bar ---
        self.status_label = tk.Label(main_frame, text="Hover over a mode for details",
                                       font=self.desc_font, bg=self.BG_COLOR, fg="#888888")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def create_mode_button(self, parent, text, mode, description):
        button = tk.Button(parent, text=text)
        
        button.configure(font=self.button_font, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR,
                         activebackground=self.BUTTON_HOVER_COLOR, activeforeground=self.TEXT_COLOR,
                         relief=tk.FLAT, borderwidth=0, width=25, anchor="center",
                         command=lambda: self.start_mode(mode))
        button.pack(pady=6, ipady=8)

        button.bind("<Enter>", lambda event: self.status_label.config(text=description))
        button.bind("<Leave>", lambda event: self.status_label.config(text="Hover over a mode for details"))

    def start_mode(self, mode):
        """Launches the selected detection mode using the 'import' method."""
        print(f"--- Starting {mode.upper()} Mode ---")
        self.root.destroy()

        try:
            if mode == "haar":
                import haar_mode
            elif mode == "dlib":
                import dlib_mode
            elif mode == "huggingface":
                from hf_model import HuggingFaceDrowsinessDetector
                detector = HuggingFaceDrowsinessDetector()
                detector.run()
        except ImportError as e:
            self.show_error(f"Failed to import script for '{mode}' mode.\n\n"
                            f"Ensure the file '{e.name}.py' exists and has no errors.")
        except Exception as e:
            self.show_error(f"An error occurred while starting '{mode}' mode:\n{e}")

    def show_error(self, message):
        error_root = tk.Tk()
        error_root.withdraw()
        tk.messagebox.showerror("Execution Error", message)
        error_root.destroy()

def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["haar", "dlib", "huggingface"],
                        help="Choose detection mode to run directly, skipping the GUI.")
    args = parser.parse_args()

    if args.mode:
        print(f"--- Starting {args.mode.upper()} Mode (CLI) ---")
        if args.mode == "haar": import haar_mode
        elif args.mode == "dlib": import dlib_mode
        elif args.mode == "huggingface":
            from hf_model import HuggingFaceDrowsinessDetector
            detector = HuggingFaceDrowsinessDetector()
            detector.run()
    else:
        root = tk.Tk()
        app = DrowsinessDetectorApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()