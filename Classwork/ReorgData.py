import os, shutil
import sys
from pathlib import Path

# Copy files matching name_ptn from src_dir to trg_dir
# Assume name_ptn is a template string with one numerical
# spot to fill.  Fill this with all values in range, copying
# all file names thus generated.
def files_to_dir(src_dir, trg_dir, name_ptn, range):
    Path.mkdir(trg_dir, parents=True, exist_ok=True)  # Make trg dir if it's not already there
    Path.mkdir(src_dir, parents=True, exist_ok=True)  # Ditto src dir
   
    for fname in [name_ptn.format(i) for i in range]:
        shutil.copyfile(str(src_dir.joinpath(fname)),
         str(trg_dir.joinpath(fname)))

src_dir = Path(sys.argv[1])  # Flat dir w/ images
trg_dir = Path(sys.argv[2])  # Home for structured train/vld/test dirs

# Training dirs
files_to_dir(src_dir, trg_dir.joinpath('train', 'cats'),
 'cat.{}.jpg', range(1000))

files_to_dir(src_dir, trg_dir.joinpath('train', 'dogs'),
 'dog.{}.jpg', range(1000))
 
# Validation dirs
files_to_dir(src_dir, trg_dir.joinpath('vld', 'cats'),
 'cat.{}.jpg', range(1000,1500))

files_to_dir(src_dir, trg_dir.joinpath('vld', 'dogs'),
 'dog.{}.jpg', range(1000,1500))

# Test dirs
files_to_dir(src_dir, trg_dir.joinpath('test', 'cats'),
 'cat.{}.jpg', range(1500,2000))

files_to_dir(src_dir, trg_dir.joinpath('test', 'dogs'),
 'dog.{}.jpg', range(1500,2000))
