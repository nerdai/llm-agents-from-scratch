# ⚠️ `examples/` has moved to [`chapter-notebooks/`](../chapter-notebooks/)

The chapter notebooks now live in
[`chapter-notebooks/`](../chapter-notebooks/). The files here are symlinks,
kept so links in the book and in older clones keep working.

**This directory will be removed on February 28, 2027.** Please update
bookmarks, scripts and clones to use `chapter-notebooks/`.

| Old path | New path |
|---|---|
| `examples/chXX.ipynb` | [`chapter-notebooks/chXX.ipynb`](../chapter-notebooks/) |
| `examples/inspector_demo.py` | [`chapter-notebooks/inspector_demo.py`](../chapter-notebooks/inspector_demo.py) |

On Windows, symlinks are only checked out as links when Developer Mode or
`git config core.symlinks true` is enabled; otherwise they appear as small
text files. Use `chapter-notebooks/` directly.
