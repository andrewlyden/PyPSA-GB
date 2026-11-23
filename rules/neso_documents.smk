# NESO website documents the workflow reads but the repository does not redistribute
# (data/neso_documents.yaml). Where a document is already on disk the rule never runs.

import re as _neso_re

import yaml as _neso_yaml

_NESO_DOCUMENTS = [doc["path"] for doc in _neso_yaml.safe_load(
    open("data/neso_documents.yaml", encoding="utf-8"))["documents"]]


rule retrieve_neso_document:
    """Download a registered NESO website document and check its SHA-256."""
    output:
        document="{neso_document}"
    params:
        document=lambda wildcards: wildcards.neso_document
    wildcard_constraints:
        neso_document="|".join(_neso_re.escape(path) for path in _NESO_DOCUMENTS)
    retries: 2
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/utilities/retrieve_neso_document.py"
