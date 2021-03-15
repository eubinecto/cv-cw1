from pathlib import Path

# directories
LIB_DIR = Path(__file__).resolve().parent
RESRC_DIR = LIB_DIR.joinpath("resources")

# the img to be used
KITTY_BMP_PATH = RESRC_DIR.joinpath("kitty.bmp")

