from pathlib import Path
from os import path

# directories
LIB_DIR = Path(__file__).resolve().parent
RESRC_DIR = path.join(LIB_DIR, '../resources')

# the img to be used
KITTY_BMP = path.join(RESRC_DIR, 'kitty.bmp')

