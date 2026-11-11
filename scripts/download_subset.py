#!/usr/bin/env python3
"""
scripts/download_subset.py
--------------------------
CLI entry point to download or synthesize the 20-class Open Images subset.
Delegates to ml.data.download.
"""

from ml.data.download import main

if __name__ == "__main__":
    main()
