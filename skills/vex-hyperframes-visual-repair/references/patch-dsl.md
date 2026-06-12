# Visual Patch DSL

Allowed operations:

- move or resize an existing element
- change the layout family
- strengthen an existing grounded relation
- persist an existing object
- retime an existing reveal
- strengthen hierarchy
- reduce density without deleting required evidence
- bind a validated local source asset
- remove unsupported non-contract content
- swap proof encoding
- reroute the renderer

Every patch is signed against the current Scene Program V2 signature. Application must
preserve grounded copy, evidence IDs, graph and semantic signatures, required objects,
required relations, relation endpoints, and relation types.

Arbitrary HTML, CSS, JavaScript, shell execution, remote URLs, and generated replacement
facts are not patch operations.
