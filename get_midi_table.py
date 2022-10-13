import numpy as np
import sys
from processor import ISSMidiFile

import argparse
import urllib.request
import shutil
import io

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--url", help="url of .mid file")
group.add_argument("--file", help="file of .mid file")
parser.add_argument('--type', default=None, type=int, help="output file")
args = parser.parse_args()
#print(args)

filename = 'tmp.txt'
if args.url:
    with urllib.request.urlopen(args.url) as response, open(filename, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

if args.file:
    filename = args.file
midi_file = ISSMidiFile(filename, mtype=args.type)

