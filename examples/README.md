# ⚠️ `examples/` has moved to [`chapter-notebooks/`](../chapter-notebooks/)

The chapter notebooks now live in
[`chapter-notebooks/`](../chapter-notebooks/). The files here are symlinks,
kept so links in the book and in older clones keep working.

**This directory will be removed on February 28, 2027.** Please update
bookmarks, scripts and clones to use `chapter-notebooks/`.

| Old path | New path |
|---|---|
| `examples/chXX.ipynb` | [`chapter-notebooks/chXX/chXX.ipynb`](../chapter-notebooks/) |
| `examples/inspector_demo.py` | [`chapter-notebooks/ch08/inspector_demo.py`](../chapter-notebooks/ch08/inspector_demo.py) |

On Windows, symlinks are only checked out as links when Developer Mode or
`git config core.symlinks true` is enabled; otherwise they appear as small
text files. Use `chapter-notebooks/` directly.

Ch05 and Ch10 start helper servers from `extra/`, which they locate
relative to their own folder (`chapter-notebooks/chXX/`). Opened through
the links here they look one level too high, so open those two from
`chapter-notebooks/` instead.
