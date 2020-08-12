import os
import argparse

parser = argparse.ArgumentParser("prepare_imagenet")
parser.add_argument('PATH_TO_IMAGENET', type=str, help="path to the prepared imagenet directory")
parser.add_argument('--val_only', action='store_true', default=False, help="whether only the val set of imagenet was prepared or not" )
args= parser.parse_args()

synset_to_label = {}
i=0
with open("src/synset_to_label.txt","r") as f:
    for line in f:
        key = line.split()[0]
        synset_to_label[key] = i
        i+= 1
if args.val_only is False:
    train_dir = os.path.join(args.PATH_TO_IMAGENET,"train")
    train_subfolders = [ f.name for f in os.scandir(train_dir) if f.is_dir() ]

    for fname in train_subfolders:
        name = os.path.join(train_dir, fname)
        os.rename(name, os.path.join(train_dir, str(synset_to_label[fname])))

val_dir = os.path.join(args.PATH_TO_IMAGENET,"val")
val_subfolders = [ f.name for f in os.scandir(val_dir) if f.is_dir() ]

for fname in val_subfolders:
    name = os.path.join(val_dir, fname)
    os.rename(name, os.path.join(val_dir, str(synset_to_label[fname])))
