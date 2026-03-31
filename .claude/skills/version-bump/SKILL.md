---
name: version-bump
description: Bump the package version (major, minor, or patch), then sync CITATION.cff, src/_version.py, and CHANGELOG.md. Use when asked to cut a release or bump the version.
---

# Version Bump

Bump the package version and keep all version references in sync.

## Arguments

The user must specify the bump type: `major`, `minor`, or `patch`.

If no bump type is given, ask the user before proceeding.

## Steps

### 0. Create a new branch

Create and switch to a new branch named `version-bump-<new_version>` (determine the new version by inspecting the current version first, then applying the bump type):

```bash
git checkout -b version-bump-<new_version>
```

### 1. Bump the package version

Run `uv version --bump <bump_type>` where `<bump_type>` is `major`, `minor`, or `patch`.

```bash
uv version --bump <bump_type>
```

Capture the new version from the output (e.g. `0.0.15`). Strip the leading `v` if present.

### 2. Update `CITATION.cff`

In `CITATION.cff`, replace the `version:` line with the new version:

```yaml
version: "<new_version>"
```

### 3. Update `src/llm_agents_from_scratch/_version.py`

Replace the `VERSION` value with the new version string:

```python
VERSION = "<new_version>"
```

### 4. Update `CHANGELOG.md`

The changelog follows [Keep a Changelog](https://keepachangelog.com/) format.

- Find the `## Unreleased` heading near the top of the file.
- Replace it with `## [<new_version>] - <today_date>` where `<today_date>` is today's date in `YYYY-MM-DD` format.
- Insert a new `## Unreleased` section immediately above the versioned heading, with an empty `### Added` subsection:

```markdown
## Unreleased

### Added

## [<new_version>] - <today_date>
```

### 5. Verify changed files

Run `git diff --name-only` and confirm that only the following files are modified:

- `CHANGELOG.md`
- `CITATION.cff`
- `pyproject.toml`
- `src/llm_agents_from_scratch/_version.py`
- `uv.lock`

If any other files are modified, stop and report them to the user before proceeding.

### 6. Commit

Stage the five files above and commit:

```bash
git add CHANGELOG.md CITATION.cff pyproject.toml src/llm_agents_from_scratch/_version.py uv.lock
git commit -m "v<new_version>"
```

### 7. Open a pull request

Push the branch and open a PR:

```bash
git push -u origin version-bump-<new_version>
gh pr create --title "version: bump version to v<new_version>" --body ""
```

### 8. Confirm

Print a summary of all changes made:
- New version
- Files updated: `pyproject.toml`, `CITATION.cff`, `src/llm_agents_from_scratch/_version.py`, `CHANGELOG.md`
- Branch created and PR URL
