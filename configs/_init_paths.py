import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
this_dir = os.path.dirname(__file__)


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


local_bin_paths=[
    r'C:\Users\lithiumice',
    r'C:\Users\lithiumice\code',
    os.path.join(this_dir, '..'),
    # sys.path.append(Path(__file__).parent.parent.__str__())
]

for path in local_bin_paths:
    add_path(path)
    
    
MPI_bin_paths=[
    '/is/cluster/scratch/hyi/ExpressiveBody',
    '/is/cluster/scratch/hyi/ExpressiveBody/speech2gesture_dataset',
]

for path in MPI_bin_paths:
    add_path(path)
    