# Adding a project or release

Each project has a top-level directory and a short guide to its model generations. Add new generations alongside existing ones, as in `decisions/v1/` and `embeddings/v2/`. Keep package metadata, optional dependencies, tests and examples with the code they describe. Shared benchmarks live in `benchmarks/`.

Link new projects from the root README. Record changes that affect users under **Unreleased** in the [changelog](CHANGELOG.md). At release time, give the entry a date, project name and version, and link the release tag. Use project-prefixed tags for new releases, such as `decisions-vX.Y.Z` or `embeddings-vX.Y.Z`. Feature major releases in the root README’s News section. The historical `v0.x` tags identify the original Decisions package.
