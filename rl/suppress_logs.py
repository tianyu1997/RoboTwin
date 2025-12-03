"""
Module to suppress verbose logging from CuRobo and other libraries.
Import this module BEFORE importing any CuRobo-related code.
"""

import os
import sys
import warnings
import logging

# ============== Environment variables (must be set before imports) ==============
# Suppress CuRobo verbose output
os.environ.setdefault("CUROBO_LOG_LEVEL", "ERROR")

# Disable JIT compilation messages (set before torch import)
os.environ.setdefault("CUROBO_TORCH_COMPILE", "0")

# Set CUDA arch to avoid the warning (use common architectures)
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"

# ============== Warning filters ==============
# Suppress PyTorch CUDA arch warnings
warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Suppress numexpr warnings
warnings.filterwarnings("ignore", message=".*NumExpr.*")
warnings.filterwarnings("ignore", message=".*NUMEXPR.*")

# ============== Logging level adjustments ==============
# Suppress verbose logging from external libraries
_VERBOSE_LOGGERS = [
    "curobo",
    "curobo.util.logger",
    "curobo.cuda_robot_model",
    "curobo.geom",
    "curobo.wrap",
    "curobo.rollout",
    "curobo.opt",
    "sapien",
    "warp",
    "nvdiffrast",
    "trimesh",
    "PIL",
    "numexpr",
]

for logger_name in _VERBOSE_LOGGERS:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


class CuRoboLogFilter(logging.Filter):
    """Filter out verbose CuRobo log messages."""
    
    FILTERED_PATTERNS = [
        "Environment variable for CUROBO",
        "USING EXISTING COLLISION CHECKER",
        "breaking reference",
        "Updating problem kernel",
        "ParallelMPPI:",
        "Updating state_seq buffer",
        "TrajOpt: solving",
        "Updating safety params",
        "Updating optimizer params",
        "Ran TO",
        "MG: running",
        "Cloning math.Pose",
        "Cloning JointState",
        "Solver was not initialized",
        "ParticleOptBase:",
        "Planning for Single Goal",
        "MG Iter:",
        "trajectory_smoothing package not found",
        "Self Collision",
        "Creating Obb cache",
        "JIT compiling",
        "jit compiling",
        "not found, JIT",
        "not found, jit",
        "Warmup",
        "USDParser failed to import",
        "NumExpr",
        "NUMEXPR",
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(pattern in msg for pattern in self.FILTERED_PATTERNS)


class StdoutFilter:
    """Filter stdout to suppress specific messages."""
    
    FILTERED_PATTERNS = [
        "kinematics_fused_cu not found",
        "geom_cu binary not found",
        "tensor_step_cu not found",
        "lbfgs_step_cu not found",
        "line_search_cu not found",
        "JIT compiling",
        "jit compiling",
        "TORCH_CUDA_ARCH_LIST",
        "pkg_resources is deprecated",
        "NumExpr",
        "NUMEXPR",
    ]
    
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
    
    def write(self, text):
        # Buffer multi-line outputs
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]  # Keep incomplete line in buffer
            for line in lines[:-1]:
                if all(pattern not in line for pattern in self.FILTERED_PATTERNS):
                    self.stream.write(line + '\n')
    
    def flush(self):
        if self.buffer:
            if all(pattern not in self.buffer for pattern in self.FILTERED_PATTERNS):
                self.stream.write(self.buffer)
            self.buffer = ""
        self.stream.flush()
    
    def __getattr__(self, name):
        return getattr(self.stream, name)


def suppress_curobo_logs():
    """Apply log filter to root logger to suppress CuRobo messages."""
    root_logger = logging.getLogger()
    root_logger.addFilter(CuRoboLogFilter())
    
    # Also add to specific loggers
    for logger_name in _VERBOSE_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.addFilter(CuRoboLogFilter())
        logger.setLevel(logging.ERROR)
    
    # Apply stdout filter to catch print statements from libraries
    if not isinstance(sys.stdout, StdoutFilter):
        sys.stdout = StdoutFilter(sys.stdout)
    if not isinstance(sys.stderr, StdoutFilter):
        sys.stderr = StdoutFilter(sys.stderr)


class SuppressStdout:
    """Context manager to temporarily suppress stdout."""
    
    def __init__(self, suppress: bool = True):
        self.suppress = suppress
        self._original_stdout = None
        self._devnull = None
    
    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            self._devnull = open(os.devnull, 'w')
            sys.stdout = self._devnull
        return self
    
    def __exit__(self, *args):
        if self.suppress and self._original_stdout is not None:
            sys.stdout = self._original_stdout
            if self._devnull:
                self._devnull.close()


# Auto-apply suppression when module is imported
suppress_curobo_logs()
