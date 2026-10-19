import sys
import os

def connect_models_file():
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent)
