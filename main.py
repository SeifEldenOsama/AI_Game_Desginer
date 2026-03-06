import argparse
from train import train
from app import ui

def main():
    parser = argparse.ArgumentParser(description="Anime Game Character Designer")
    parser.add_argument("--mode", type=str, choices=["train", "app"], default="app", help="Run mode: train or app")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        ui.launch(share=True)

if __name__ == "__main__":
    main()
