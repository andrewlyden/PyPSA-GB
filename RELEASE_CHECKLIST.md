# PyPSA-GB v2.0.0 Release Checklist

## Pre-Release Checks

### Code Quality
- [ ] All tests pass: `pytest tests/`
- [ ] No critical linting errors
- [ ] Code coverage meets threshold
- [ ] All deprecation warnings addressed
- [ ] Performance benchmarks acceptable

### Documentation
- [ ] README.md up to date
- [ ] AGENTS.md reflects current architecture
- [ ] Release notes complete ([docs/source/development/release_notes.md](docs/source/development/release_notes.md))
- [ ] All docstrings current
- [ ] Tutorial notebooks working
- [ ] Sphinx docs build without errors: `cd docs && make html`
- [ ] Changelog up to date

### Configuration & Data
- [ ] `config/scenarios.yaml` validated
- [ ] `config/defaults.yaml` documented
- [ ] All example scenarios tested
- [ ] Data files version checked (DUKES, REPD, FES, etc.)
- [ ] Environment file current: `envs/pypsa-gb.yaml`

### Dependencies
- [ ] All dependencies pinned in `envs/pypsa-gb.yaml`
- [ ] Compatible with Python 3.10, 3.11, 3.12
- [ ] PyPSA version compatibility verified
- [ ] Solver requirements documented (Gurobi/HiGHS)

### Testing
- [ ] Smoke tests pass on all network models (ETYS, Reduced, Zonal)
- [ ] Historical scenario (2015-2024) tested
- [ ] Future scenario (2025+) tested
- [ ] Clustering functionality verified
- [ ] Network upgrades tested
- [ ] Cross-platform testing (Windows/Linux/macOS)

---

## Version Updates

### Version Numbers
- [ ] Update version in `setup.py` or `pyproject.toml` (if exists)
- [ ] Update version in `docs/source/conf.py`
- [ ] Update version in `README.md`
- [ ] Update version in `CITATION.cff` (if exists)
- [ ] Update release date in [docs/source/development/release_notes.md](docs/source/development/release_notes.md)

### Git Operations
- [ ] Create release branch: `git checkout -b release-2.0.0`
- [ ] Commit all version updates
- [ ] Push release branch
- [ ] Create pull request to main

---

## Release Process

### GitHub Release
- [ ] Merge release branch to main
- [ ] Tag release: `git tag -a v2.0.0 -m "Release v2.0.0"`
- [ ] Push tags: `git push origin v2.0.0`
- [ ] Create GitHub release from tag
- [ ] Add release notes to GitHub release
- [ ] Upload any release artifacts (if applicable)

### Documentation
- [ ] ReadTheDocs webhook configured (see ReadTheDocs section below)
- [ ] Documentation builds successfully on ReadTheDocs
- [ ] Version 2.0.0 appears in version selector
- [ ] Set v2.0.0 as default version on ReadTheDocs

### Distribution (Optional)
- [ ] Package for PyPI: `python -m build`
- [ ] Test on TestPyPI: `twine upload --repository testpypi dist/*`
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Verify installation: `pip install pypsa-gb==2.0.0`

### Conda Package (Optional)
- [ ] Update conda-forge recipe
- [ ] Test conda installation
- [ ] Submit PR to conda-forge

---

## Post-Release

### Communication
- [ ] Announce on GitHub Discussions
- [ ] Update project website (if exists)
- [ ] Social media announcement
- [ ] Notify key users/collaborators
- [ ] Update any related publications

### Maintenance
- [ ] Create milestone for v2.1.0
- [ ] Update project board
- [ ] Archive old branches
- [ ] Update development branch

### Monitoring
- [ ] Monitor issue tracker for bug reports
- [ ] Check ReadTheDocs build status
- [ ] Verify download statistics
- [ ] Check CI/CD pipelines

---

## Rollback Plan

If critical issues discovered:
- [ ] Document issue severity
- [ ] Create hotfix branch from v2.0.0 tag
- [ ] Apply fix and test
- [ ] Release v2.0.1 following checklist
- [ ] Update communication channels

---

## Notes

**Release Date**: January 2026
**Release Manager**: [Your Name]
**Target Platforms**: Windows, Linux, macOS
**Python Versions**: 3.10+

**Breaking Changes in 2.0.0**:
- Snakemake workflow (vs. notebook-based)
- YAML configuration (vs. Python)
- New file naming conventions
- Updated FES 2024 pathways

**Migration Support**:
- Upgrade guide in release notes
- Example migration scripts (if needed)
- Deprecation warnings for old APIs
