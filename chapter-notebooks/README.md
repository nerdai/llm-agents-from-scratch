# Chapter notebooks

One folder per chapter: `chXX/chXX.ipynb`, alongside any files that
chapter reads (for example `ch09/hailstone_known_sequences.json`).

Run each notebook from its own folder; Jupyter does this when you open the
notebook. The folder is also where an agent finds *project* skills: the
library scans `<working directory>/.agents/skills`.

- **Ch06–Ch10** use the example skill `stop-at-one`. It lives once in
  [`shared/.agents/skills/`](shared/.agents/skills/), and each of those
  chapter folders links to it with an `.agents` symlink.
- **Ch02–Ch05** have no `.agents` folder, so their agents see no project
  skills, matching the book, which introduces skills in Chapter 6.

On Windows, the `.agents` symlinks only work if you clone with symlink
support (Developer Mode, or `git config core.symlinks true`). Otherwise
copy `shared/.agents` into `ch06/` through `ch10/`.

On Google Colab, uploading a notebook on its own leaves out the files next
to it, including `.agents`. Clone the repository in the notebook instead
(`!git clone https://github.com/nerdai/llm-agents-from-scratch`), then
`%cd` into the chapter folder, install from PyPI, and set
`os.environ["OLLAMA_API_KEY"]`, since Colab has no local Ollama server.
