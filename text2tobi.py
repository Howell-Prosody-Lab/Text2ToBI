#!/usr/bin/env python3
# text2tobi.py — shebang entry point.
# Run directly:  python text2tobi.py "some text"
# Or after chmod +x: ./text2tobi.py "some text"

import sys
import os

# Ensure the repo root is on the path so the inner package is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text2tobi.__main__ import main

if __name__ == "__main__":
    main()
