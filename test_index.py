# Tests for indexing decisions: what gets read, what gets refused, and how the
# result is presented. No API key, no network.
#
# These exist because the bugs in this area are all SILENT. A file type that is
# not on the allow-list is not an error, it is an absence: the app indexes what
# is left, reports a number, and reads as working. Notebooks were missing for
# exactly that reason, and on a data science repository that meant indexing the
# README and the .gitignore while the four notebooks holding the entire project
# were counted as "not a text file".
#
# Usage:
#   pytest test_index.py -v
#   python test_index.py

import json
import sys
import types

import repo_index as R


def make_notebook(cells) -> str:
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4})


def code_cell(source, outputs=None):
    return {"cell_type": "code", "execution_count": 1,
            "source": source, "outputs": outputs or []}


def markdown_cell(source):
    return {"cell_type": "markdown", "source": source}


# ============================================================================
# 1. Notebooks: the case that was silently broken
# ============================================================================


def test_notebooks_are_indexable_at_all():
    """REGRESSION. .ipynb was on neither extension list, so every notebook was
    rejected and counted as "not a text file"."""
    assert R._is_text_file("analysis.ipynb"), "notebooks are refused again"
    assert R._is_notebook("a/b/analysis.ipynb")
    assert not R._is_notebook("analysis.py")


def test_notebook_outputs_are_discarded():
    """A committed notebook is mostly output. Embedding a base64 PNG spends the
    bill on a plot and retrieves punctuation."""
    huge_png = "iVBORw0KGgoAAAANSUhEUgAA" + ("A" * 5000)
    raw = make_notebook([
        code_cell(["df = pd.read_csv('books.csv')\n"],
                  outputs=[{"data": {"image/png": huge_png},
                            "output_type": "display_data"}]),
    ])
    text = R.notebook_to_python("n.ipynb", raw)

    assert "pd.read_csv" in text, "the actual code was lost"
    assert huge_png[:40] not in text, "output image was embedded"
    assert "output_type" not in text and "execution_count" not in text, \
        "notebook JSON metadata leaked into the index"
    assert len(text) < 500, f"extracted {len(text)} chars from one line of code"


def test_notebook_markdown_is_kept_as_comments():
    """The prose is often the only explanation of what the code is for."""
    raw = make_notebook([
        markdown_cell(["# Loading the data\n", "We drop rows with no description.\n"]),
        code_cell(["df = df.dropna()\n"]),
    ])
    text = R.notebook_to_python("n.ipynb", raw)
    assert "no description" in text, "markdown was dropped"
    for line in text.splitlines():
        if "no description" in line:
            assert line.lstrip().startswith("#"), "prose is not commented out"


def test_notebook_magics_do_not_break_the_parser():
    """%matplotlib and !pip are not Python. Left as-is they fail ast.parse, and
    the file silently falls back to character splitting."""
    raw = make_notebook([
        code_cell(["%matplotlib inline\n", "!pip install pandas\n",
                   "%%time\n", "import pandas as pd\n"]),
        code_cell(["def clean(frame):\n", "    return frame.dropna()\n"]),
    ])
    text = R.notebook_to_python("n.ipynb", raw)

    import ast
    ast.parse(text)          # the assertion: this must not raise

    assert "pip install pandas" in text, "commented out, not deleted"
    pieces = R.chunk_python_ast("n.ipynb", text)
    symbols = [symbol for _, symbol in pieces]
    assert "clean" in symbols, f"the AST chunker found {symbols}, expected 'clean'"


def test_notebook_source_may_be_a_string_or_a_list():
    """The standard says list of lines; plenty of tools write one string."""
    as_list = R.notebook_to_python("n.ipynb", make_notebook([code_cell(["a = 1\n", "b = 2\n"])]))
    as_string = R.notebook_to_python("n.ipynb", make_notebook([code_cell("a = 1\nb = 2\n")]))
    assert "a = 1" in as_list and "b = 2" in as_list
    assert "a = 1" in as_string and "b = 2" in as_string


def test_malformed_notebook_is_a_skip_not_a_crash():
    """One bad file must not fail a whole repository."""
    for bad in ("", "not json at all", "[]", "null", '{"cells": "nonsense"}',
                '{"cells": [null, 3, "x"]}', '{"no_cells": true}'):
        assert R.notebook_to_python("n.ipynb", bad) == "", f"{bad!r} did not skip cleanly"


def test_notebooks_get_a_higher_size_ceiling():
    """Most of a committed .ipynb is output that we discard, so judging it on
    file size rejects a 40 KB program for carrying a 3 MB plot."""
    assert R.MAX_NOTEBOOK_BYTES > R.MAX_FILE_BYTES * 4


# ============================================================================
# 2. What must NEVER be indexed
# ============================================================================


def test_credential_files_are_never_indexed():
    """A private key is plain text, so nothing else here would stop it.

    Once a chunk is in the store the agent can be asked to quote it, and the app
    would read a key file aloud on request.
    """
    for path in (".env", ".env.production", "tests/certs/server.key",
                 "fixtures/ca.pem", "id_rsa", "id_ed25519", "a/b/client.p12",
                 ".netrc", ".npmrc", "secrets.toml"):
        assert not R._is_text_file(path), f"{path} would be indexed"
        assert R._skip_reason(path) == R.SKIP_CREDENTIAL, f"{path} mislabelled"


def test_generated_files_are_refused_even_with_a_source_extension():
    """jquery.min.js really is JavaScript, and it is still noise."""
    for path in ("static/jquery.min.js", "a/styles.min.css", "yarn.lock",
                 "bundle.js.map", "__snapshots__/render.snap"):
        assert not R._is_text_file(path), f"{path} would be indexed"
        assert R._skip_reason(path) == R.SKIP_GENERATED, f"{path} mislabelled"


def test_module_javascript_is_indexed():
    """REGRESSION. .mjs and .cjs were missing, which dropped 16 real source
    files from React alone."""
    for path in ("src/index.mjs", "src/util.cjs", "types/a.mts", "types/b.cts"):
        assert R._is_text_file(path), f"{path} is being skipped"


# ============================================================================
# 3. How the result is presented
# ============================================================================


def make_contents():
    contents = R.RepoContents(owner="me", repo="proj")
    contents.tree = ["README.md", "app/main.py", "app/util.py",
                     "data/books.csv", "notes.ipynb", "img/logo.png"]
    contents.files = {"README.md": "# hi", "app/main.py": "x = 1",
                      "app/util.py": "y = 2", "notes.ipynb": "# In [1]:\nz = 3"}
    contents.skipped = {R.SKIP_DATA: 1, R.SKIP_BINARY: 1}
    return contents


def test_skip_summary_is_english_not_reason_keys():
    """It used to render "7 not a text file, 1 too large"."""
    contents = make_contents()
    summary = R.describe_skipped(contents)
    assert summary == "1 data file, 1 image or binary", summary

    contents.skipped = {R.SKIP_DATA: 3, R.SKIP_LARGE: 1}
    summary = R.describe_skipped(contents)
    assert "3 data files" in summary, summary
    assert "1 file over" in summary and "KBs" not in summary, \
        f"pluralisation is wrong: {summary}"

    assert R.describe_skipped(R.RepoContents(owner="a", repo="b")) == "", \
        "an empty skip set must produce no text at all"


def test_the_tree_has_no_question_marks():
    """REGRESSION. Every line used to carry an indexed/not-indexed marker, so on
    a data science repo the answer to "show me the structure" was a wall of "?".
    """
    tree = R.build_tree(make_contents())

    body = tree.split("Listed above")[0]
    assert "?" not in body, f"marker characters are back:\n{body}"
    assert "├──" in body or "└──" in body, \
        "not rendered as a tree"


def test_the_tree_names_what_it_could_not_read():
    """Dropping the markers must not drop the honesty."""
    tree = R.build_tree(make_contents())
    assert "books.csv" in tree and "logo.png" in tree
    assert "Listed above but not indexed" in tree
    assert "Reason: 1 data file, 1 image or binary" in tree


def test_the_tree_puts_directories_before_files():
    contents = make_contents()
    lines = [line for line in R.build_tree(contents).splitlines()
             if line.startswith(("├", "└"))]
    directories = [i for i, line in enumerate(lines) if line.rstrip().endswith("/")]
    files = [i for i, line in enumerate(lines) if not line.rstrip().endswith("/")]
    assert not directories or not files or max(directories) < min(files), \
        f"directories and files are interleaved:\n" + "\n".join(lines)


def test_the_tree_survives_an_empty_repo():
    contents = R.RepoContents(owner="me", repo="proj")
    assert "proj/" in R.build_tree(contents)


# ============================================================================
# Runner
# ============================================================================


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and isinstance(v, types.FunctionType)]
    print("=" * 72)
    print(f"index tests ({len(tests)}), no key, no network")
    print("=" * 72)
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
        except Exception as error:
            failures += 1
            print(f"  [FAIL] {test.__name__}: {type(error).__name__}: {error}")
    print("=" * 72)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
