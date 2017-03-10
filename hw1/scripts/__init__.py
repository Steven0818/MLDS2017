# The hack to import something in the parent folder
import os
import sys
PARENT_DIR = os.path.abspath(__file__).split('/')[:-2]
sys.path.insert(0, '/'.join(PARENT_DIR))

