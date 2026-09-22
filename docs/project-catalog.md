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
history remains monotonic. If a catalog commit succeeds but the adjacent JSON
export fails, Vex warns and keeps the committed revision authoritative; the
next successful save repairs the export. The same rule applies to catalog-backed
CLI job records.

Studio tasks and CLI tool jobs now use one `executions` ledger in the catalog.
Studio stores status, a bounded event log, and the latest stream; CLI jobs store
attempts, result, stage, and progress. Older Studio rows in `studio_tasks` are
imported when the ledger first opens. Existing CLI `jobs/*.json` files remain
readable and are imported into the ledger on their next write. JSON job files
continue as compatibility exports; catalog rows win if both exist. The Studio
Activity page reads CLI job status from the same ledger.

The ledger enforces at most one active Studio task per project, including across
server processes. A CLI worker can save a cooperative stage/progress checkpoint
with `checkpoint_job`; only the current job process can do so, and progress is
monotonic. The CLI jobs table and Studio Activity view both expose those fields.

Task polling survives a Studio restart. A queued or running Studio task whose
owner process has exited is marked as interrupted and shown as failed. CLI jobs
retain their explicit `--force` recovery gate. This is **record recovery**, not
execution resumption: a retry must be initiated after checking the project's
result. The ledger tracks execution but does not yet checkpoint FFmpeg or model
work itself.

The catalog now also stores asset and cache indexes. Existing `assets.json` and
`cache/cache_index.json` records are imported on first catalog write; their JSON
files become repairable exports. `promote_working_file` stages an immutable,
checksum-verified cache object, then commits the new asset, cache record, and
project timeline revision in **one SQLite transaction**. A failure before the
commit leaves no partial metadata and restores in-memory project state. An
unreferenced cache object may remain after a failed transaction; it is safe to
retain and can be garbage-collected later. Studio Activity displays registered
media lineage from the catalog.

This transaction boundary currently covers tools that call
`promote_working_file` (not every legacy edit path). Plans, some generated
artifacts, and external render processes still have separate lifecycles. Do not
describe Vex as fully transactional or its media jobs as resumable yet.
