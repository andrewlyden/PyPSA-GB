"""
Replicate the conda activation scripts for runs launched without activation.

Windows conda packages (GDAL, PROJ, ecCodes) rely on environment variables that
are only set by ``conda activate``. When Snakemake is started directly through
``envs/pypsa-gb/python.exe -m snakemake`` those scripts never run, and every
job subprocess then emits:

    RuntimeWarning: Engine 'cfgrib' loading failed: Cannot find the ecCodes library
    Warning 3: Cannot find gdalvrt.xsd (GDAL_DATA is not defined)

The ecCodes DLL is found through ``ctypes.util.find_library``, which searches
``PATH``; GDAL and PROJ read ``GDAL_DATA`` and ``PROJ_DATA``. Setting these on
the parent process before any job is spawned makes every child inherit them.

Call ``ensure_conda_env_vars()`` as the first statement in each Snakefile.
On non-Windows platforms (or in an already-activated shell) it is a no-op.
"""

import os
import sys
from pathlib import Path


def ensure_conda_env_vars(prefix: str | Path | None = None) -> dict[str, str]:
    """Set PATH / GDAL_DATA / PROJ_DATA to the conda env's own directories.

    Parameters
    ----------
    prefix : str or Path, optional
        Conda environment root. Defaults to ``sys.prefix`` (the interpreter
        running this code), which is correct when Snakemake is invoked via the
        environment's ``python.exe``.

    Returns
    -------
    dict
        The variables that were changed (empty if nothing needed doing).
    """
    if os.name != "nt":
        return {}

    prefix = Path(prefix or sys.prefix)
    library = prefix / "Library"
    changed: dict[str, str] = {}

    # ecCodes (and other native DLLs) live in Library\bin; ctypes searches PATH.
    lib_bin = library / "bin"
    if lib_bin.is_dir():
        path_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        if not any(Path(p) == lib_bin for p in path_entries if p):
            os.environ["PATH"] = os.pathsep.join([str(lib_bin), *path_entries])
            changed["PATH"] = str(lib_bin)

    # Mirrors etc/conda/activate.d/gdal-activate.bat and proj4-activate.bat
    data_vars = {
        "GDAL_DATA": library / "share" / "gdal",
        "PROJ_DATA": library / "share" / "proj",
    }
    for var, directory in data_vars.items():
        if var not in os.environ and directory.is_dir():
            os.environ[var] = str(directory)
            changed[var] = str(directory)

    if "PROJ_NETWORK" not in os.environ and "PROJ_DATA" in changed:
        # proj-data is bundled when its licence file was copied over
        licence = data_vars["PROJ_DATA"] / "copyright_and_licenses.csv"
        os.environ["PROJ_NETWORK"] = "OFF" if licence.exists() else "ON"
        changed["PROJ_NETWORK"] = os.environ["PROJ_NETWORK"]

    return changed
