"""Hackathon harness. Run it with `python -m harness.run` from the repository root.
 
The trading environment lives in the sibling `env/` package. Putting the repository root
on sys.path guarantees that `from env.env import TradingEnv` finds it (and not this
package's own `harness/env.py`), whatever the working directory is.
"""
import sys
from pathlib import Path
 
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
 