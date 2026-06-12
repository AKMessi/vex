# Security Policy

## Supported Versions

Until Vex reaches 1.0, security fixes are provided for the latest published
release line only.

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting flow:

<https://github.com/AKMessi/vex/security/advisories/new>

Include the affected version, operating system, reproduction steps, impact,
and any proposed mitigation. Avoid including real API keys, private media, or
other secrets in the report.

## Release Security

Official Python packages are published by GitHub Actions through PyPI Trusted
Publishing. The release workflow uses short-lived OIDC credentials and does not
store a long-lived PyPI API token. GitHub Release artifacts include SHA-256
checksums and GitHub build provenance attestations.

