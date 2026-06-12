# Release Gate

A repair is monotonic only when:

- hard failures do not increase
- object and relation coverage do not regress
- render quality does not materially regress
- no new error or hard-failure category appears without resolving a harder failure
- hard failures decrease, or the composite score improves by the configured minimum

The independent final judge receives only final chronological frames, the signed
production contract, and the signed Scene Program V2. It never receives candidate
rankings, critic history, patch history, or previous scores.

In strict vision mode, missing final vision judgment is a failure. In hybrid mode,
the artifact-only gate remains available, but it still requires passing renderer QA,
passing structured critics, zero hard counterexamples, and a signed scene program.
