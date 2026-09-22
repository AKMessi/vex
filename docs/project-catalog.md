# Project catalog (architecture 1.0 foundation)

Each project has one authoritative `project.sqlite3` catalog in its working
directory. `project_revisions` stores complete, checksummed state snapshots with
monotonically increasing revision numbers. The existing `<project-id>.json` is
still written for older integrations, but is a **compatibility export**, not a
second source of truth. CLI and Studio both load through `ProjectState` and see
the same catalog revision.

On the first save of a legacy JSON project, Vex imports that JSON as revision 1
and commits the new state as revision 2 in the same SQLite transaction. An
unreadable or mismatched legacy file aborts migration. Existing JSON projects
are not rewritten just by listing or opening them. Deleting a damaged catalog
does not trigger automatic JSON recovery; doing so could silently discard newer
edits. Make a copy of the project directory and investigate the catalog first.

`ProjectState.save()` uses compare-and-swap: an editor whose loaded revision is
stale gets a conflict and must reload. The project mutation lock still
serializes normal tool execution; revision checking protects other writers that
do not share that lock. A rollback is saved as a new revision so the audit
history remains monotonic.

The catalog currently covers project state only. Jobs, assets, plans, and cache
indexes still use their existing stores; cross-store atomicity is **not yet
provided**. Later architecture checkpoints will migrate those records and add
recovery of interrupted promotion. Do not describe this first step as fully
transactional media editing.
