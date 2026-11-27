"""RiskLadder: exact benchmarking of safe planning and learning under model uncertainty.

Subpackages:
    env      grid environment with an exact model interface and a sampling interface
    oracle   exact solvers and policy evaluation (value, failure probability, regret)
    agents   planners (A*, VI/PI, CMDP) and learners (Q, SARSA, linear, model-based)
    safety   safety shield and potential-based reward shaping
    harness  training and evaluation loop shared by experiments, the API and the demo
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("riskladder")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
