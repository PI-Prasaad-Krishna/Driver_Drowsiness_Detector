import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="haar",
                        choices=["haar", "dlib", "huggingface"],
                        help="Choose detection mode")
    args = parser.parse_args()

    if args.mode == "haar":
        import haar_mode
        haar_mode.run()
    elif args.mode == "dlib":
        import dlib_mode
        dlib_mode.run()
    elif args.mode == "huggingface":
        from hf_model import HuggingFaceDrowsinessDetector
        detector = HuggingFaceDrowsinessDetector()
        detector.run()

if __name__ == "__main__":
    main()
