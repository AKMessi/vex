# Releasing Vex

This runbook is the authority for TestPyPI, PyPI, and GitHub Releases.

## One-Time Account Setup

1. Create PyPI and TestPyPI accounts and enable two-factor authentication.
2. In GitHub, create environments named `testpypi` and `pypi`.
3. Add a required reviewer to both environments. Keep the `pypi` environment
   restricted to the `main` branch and protected release tags.
4. Create a pending Trusted Publisher on TestPyPI:
   - PyPI project name: `vex-video`
   - GitHub owner: `AKMessi`
   - repository: `vex`
   - workflow: `release.yml`
   - environment: `testpypi`
5. Create the corresponding pending Trusted Publisher on PyPI with environment
   `pypi`.
6. Protect tags matching `v*` so only maintainers can create release tags.
7. Enable immutable GitHub Releases after the first successful release.

No PyPI API token is required or expected in GitHub secrets.

## Release Candidate

1. Confirm `main` is clean and synchronized with `origin/main`.
2. Set `vex_runtime.__version__` to the next release candidate.
3. Update `CHANGELOG.md` and remove `Unreleased` from that version's date.
4. Confirm the HyperFrames manifests match:

   ```bash
   cmp package.json vex_runtime/resources/hyperframes/package.json
   cmp package-lock.json vex_runtime/resources/hyperframes/package-lock.json
   ```

5. Run the local release gate:

   ```bash
   python -m pytest -q
   rm -rf build dist
   python -m build
   python -m twine check dist/*.whl dist/*.tar.gz
   python scripts/release_checks.py --dist-dir dist --write-checksums
   pipx install --backend pip --force dist/*.whl
   vex --version
   vex setup config --path .vex-release.env
   vex renderers doctor
   ```

6. Commit and push the release preparation. Wait for every CI job to pass.
7. Create and push an annotated tag:

   ```bash
   git tag -a v0.1.0rc1 -m "Vex 0.1.0rc1"
   git push origin v0.1.0rc1
   ```

The tag triggers TestPyPI publication, an isolated `pipx` installation check,
provenance attestation, and a GitHub prerelease. Environment approval remains a
human gate.

## Stable Release

Promote only after the release candidate has been installed and exercised on
real projects.

1. Set the version to the stable value and finalize the changelog.
2. Run the full local gate and wait for CI.
3. Create and push the stable annotated tag.
4. Approve the `pypi` environment deployment in GitHub.
5. Verify the PyPI install, GitHub Release assets, `SHA256SUMS`, and attestation.

The workflow does not publish from branch pushes. Only a version-matching `v*`
tag can reach a package index.

## Failure And Rollback

- Before publication: delete the local tag, fix the release commit, and create a
  new release-candidate version. Do not reuse a tag already pushed publicly.
- After TestPyPI publication: publish a new release-candidate version.
- After PyPI publication: never replace files for the same version. Yank a bad
  release on PyPI, document the reason, and publish a higher patch version.
- After an immutable GitHub Release: publish a corrective release; do not mutate
  the old tag or assets.

Yanking prevents ordinary dependency resolution from selecting the release while
preserving reproducibility for users who explicitly pin it.

## Verification

```bash
pipx install --force "vex-video==<version>"
vex --version
vex setup config --path /tmp/vex-release.env
vex renderers doctor
```

Verify downloaded GitHub artifacts from the directory containing them:

```bash
sha256sum -c SHA256SUMS
```

