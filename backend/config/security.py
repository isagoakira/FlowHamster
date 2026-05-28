"""
Security configuration for FlowHamster backend.
"""
import os

# Development mode: allows raw Python code execution via /api/execute
# In production, this should ALWAYS be False.
ALLOW_RAW_EXECUTION = os.environ.get("FLOWHAMSTER_ALLOW_RAW_EXECUTION", "").lower() in (
    "1",
    "true",
    "yes",
)

# Allowed import modules for raw code execution (whitelist)
# Any import not in this list will be rejected.
ALLOWED_IMPORTS = {
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils.data",
    "torchvision",
    "torchvision.transforms",
    "torchvision.datasets",
    "numpy",
    "np",  # commonly aliased
    "math",
    "random",
    "json",
    "time",
    "typing",
    "collections",
    "functools",
    "itertools",
    "pathlib",
    "warnings",
    "abc",
    "numbers",
    "enum",
    "copy",
    "inspect",
    "types",
    "dataclasses",
    "contextlib",
    "io",
    "string",
    "re",
    "textwrap",
    "hashlib",
    "base64",
    "uuid",
    "datetime",
    "decimal",
    "fractions",
    "statistics",
}

# Whitelist of environment variables passed to sandboxed subprocess
ALLOWED_ENV_VARS = {
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "SYSTEMROOT",
    "WINDIR",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
    "COMSPEC",
    "PATHEXT",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PUBLIC",
    "USERNAME",
    "COMPUTERNAME",
    "USERDOMAIN",
}
