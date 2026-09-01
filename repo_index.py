# Repo Index: fetch a GitHub repository and make it searchable
#
# This is the same three steps as 4_rag/1a_rag_basics.py in the langchain-course -
# load documents, split them, embed them into a vector store, with two changes
# that both exist because this app gets DEPLOYED rather than run on a laptop.
#
# CHANGE 1: WE DOWNLOAD A ZIP, WE DO NOT `git clone`
# A hosted app cannot rely on a `git` binary being installed, and shelling out
# to a subprocess on a shared host is something to avoid if there is a simpler
# route. GitHub will hand you the whole repository as a zip over plain HTTP:
#
#     https://api.github.com/repos/{owner}/{repo}/zipball
#
# One request, no subprocess, no binary, works anywhere Python runs.
#
# CHANGE 2: THE VECTOR STORE IS IN MEMORY, NOT CHROMA
# The langchain-course uses Chroma, and locally that is the right choice. On a
# hosted app it is not, for two concrete reasons:
#
#   - Chroma needs SQLite 3.35+. Streamlit Community Cloud ships an older
#     SQLite, so Chroma there needs a `pysqlite3-binary` shim and a module-swap
#     hack at the top of the app. That is a lot of machinery to import a
#     database we do not need.
#   - The filesystem on these hosts is ephemeral and shared. Persisting an index
#     buys nothing when the container can be recycled at any moment, and every
#     visitor indexes a different repository anyway.
#
# `InMemoryVectorStore` is a first-class LangChain vector store with the same
# `.as_retriever()` interface the course teaches, so nothing about the lesson
# changes, only where the vectors live. One index per visitor, discarded when
# they leave, which is exactly the lifetime we want.

import ast
import io
import os
import re
import zipfile
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Load THIS directory's .env, explicitly.
#
# load_dotenv() with no argument walks UP the directory tree until it finds a
# .env, so an app sitting inside a larger project silently picks up the parent's
# key. That is wrong twice over: locally you cannot tell whether the app is
# configured or is borrowing someone else's credentials, and it hides a missing
# key that would fail on deployment. Load only our own file.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Pinned to the model allow-list on the OpenAI project. See guard.py.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Directories that are never worth indexing: dependency trees, build output,
# caches. Skipping them is most of what keeps an index small and useful.
IGNORE_DIRS = {
    ".git", ".github", "node_modules", "dist", "build", "__pycache__", ".venv",
    "venv", "env", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".idea",
    ".vscode", "vendor", "target", ".next", ".nuxt", "coverage", "htmlcov",
    "site-packages", ".eggs", "migrations",
}

# File extensions we index, mapped to the LangChain splitter language where one
# exists. A language-aware splitter tries to break at function and class
# boundaries rather than mid-statement, so a retrieved chunk is far more likely
# to be a complete thought.
LANGUAGE_BY_EXTENSION = {
    ".py": Language.PYTHON,
    ".js": Language.JS, ".jsx": Language.JS,
    ".ts": Language.TS, ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".cs": Language.CSHARP,
    ".cpp": Language.CPP, ".cc": Language.CPP, ".hpp": Language.CPP,
    ".c": Language.C, ".h": Language.C,
    ".kt": Language.KOTLIN,
    ".swift": Language.SWIFT,
    ".scala": Language.SCALA,
    ".md": Language.MARKDOWN, ".markdown": Language.MARKDOWN,
    ".html": Language.HTML,
    ".sol": Language.SOL,
}

# Text files worth indexing that have no language-aware splitter
PLAIN_EXTENSIONS = {
    ".txt", ".rst", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json",
    ".sh", ".bash", ".zsh", ".env.example", ".sql", ".css", ".scss",
    ".dockerfile", ".makefile", ".gradle", ".properties",
}

# Filenames with no extension that are still worth reading
NOTABLE_FILENAMES = {
    "Dockerfile", "Makefile", "LICENSE", "README", "CHANGELOG",
    "Procfile", "Pipfile", ".gitignore",
}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Guards, because a visitor can paste a link to anything. Without these, one
# person pointing at a monorepo hangs the app and spends the whole budget.
MAX_ZIP_BYTES = 60 * 1024 * 1024      # 60 MB download ceiling
MAX_FILE_BYTES = 400 * 1024           # skip any single file over 400 KB
MAX_FILES = 1200                      # stop indexing beyond this many files
MAX_CHUNKS = 4000                     # ceiling on the embedding bill


# ============================================================================
# PART 1: The URL
# ============================================================================


def parse_repo_url(url: str) -> tuple[str, str]:
    """Turn anything that looks like a GitHub link into (owner, repo).

    Accepts what people actually paste: full URLs, URLs with a trailing slash,
    `.git` suffixes, SSH remotes, `/tree/main/...` deep links, and bare
    `owner/repo`.
    """
    text = url.strip()
    if not text:
        raise ValueError("Please paste a GitHub repository link.")

    text = re.sub(r"^git@([^:]+):", r"https://\1/", text)      # SSH form
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"\.git$", "", text)
    text = text.strip("/")

    parts = [p for p in text.split("/") if p]

    # Drop the host if there is one. A GitHub username cannot contain a dot, so
    # a dot in the first segment means it is a hostname, not an owner. Checking
    # the segment COUNT instead (">= 3") lets "github.com/onlyowner" through as
    # owner="github.com", repo="onlyowner", which then 404s confusingly.
    if parts and "." in parts[0]:
        parts = parts[1:]

    if len(parts) < 2:
        raise ValueError(
            f"Could not read owner/repo from {url!r}. "
            f"Try something like https://github.com/pypa/sampleproject"
        )

    # Deep links like /tree/main/src, so keep only the first two segments
    return parts[0], parts[1]


# ============================================================================
# PART 1b: Which files to keep when a repo is bigger than the caps
# ============================================================================
# THE BUG THIS FIXES, and it was invisible until measured.
#
# The caps used to be applied in ZIP ORDER, which is roughly alphabetical, so
# WHICH files survived was arbitrary:
#
#   facebook/react   -> chunk languages came out rust:3332, ts:104.
#                       React's repo has 4,210 files under compiler/ (a Rust
#                       compiler) against 2,154 under packages/. Reading in
#                       alphabetical order, the 1,200-file cap filled up inside
#                       compiler/ and never reached React itself. Asking about
#                       hooks retrieved Rust.
#   tiangolo/fastapi -> markdown:3756 against js:25, because the cap filled up
#                       on the translated documentation tree.
#
# The repositories were never the problem. The selection was.
#
# So: work out what the project is mostly WRITTEN IN, then keep files in
# priority order, primary-language source first, the top-level README next,
# other source after that, documentation and test trees last.

# Path fragments that mark content as real but secondary
DOC_PATH_HINTS = ("docs/", "doc/", "website/", "site/", "translations/",
                  "i18n/", "locale/", "changelog", "news/")
TEST_PATH_HINTS = ("test/", "tests/", "__tests__/", "spec/", "e2e/", "fixtures/",
                   "benchmark", "examples/", "example/", "demo/", "samples/",
                   "scripts/")

# Markdown lives in LANGUAGE_BY_EXTENSION (it has its own splitter), but it is
# prose, not source, so it must be excluded when deciding the primary language.
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
PROSE_EXTENSIONS = MARKDOWN_EXTENSIONS | {".txt", ".rst"}
SOURCE_EXTENSIONS = set(LANGUAGE_BY_EXTENSION) - MARKDOWN_EXTENSIONS


def _primary_extensions(entries: list) -> set:
    """Which extensions is this project mostly WRITTEN in, weighted by bytes.

    By bytes rather than file count: a project can have 2,000 tiny generated
    files and 50 substantial modules, and the modules are the thing.

    Documentation and test trees are excluded from the vote, or a project with
    enormous docs elects Markdown as its primary language and we are back where
    we started.
    """
    weight = {}
    for path, size in entries:
        lowered = path.lower()
        if any(h in lowered for h in DOC_PATH_HINTS + TEST_PATH_HINTS):
            continue
        extension = os.path.splitext(path)[1].lower()
        if extension in SOURCE_EXTENSIONS:
            weight[extension] = weight.get(extension, 0) + size

    if not weight:
        return set(SOURCE_EXTENSIONS)

    total = sum(weight.values())
    ranked = sorted(weight.items(), key=lambda kv: -kv[1])

    # Take extensions until 80% of source bytes are covered, so a genuinely
    # bilingual project (a .ts front end with a .py back end) keeps both.
    primary, running = set(), 0
    for extension, size in ranked:
        primary.add(extension)
        running += size
        if running / total >= 0.80:
            break
    return primary


def _file_priority(path: str, primary: set) -> int:
    """Lower is kept first. See the comment at the top of this section."""
    lowered = path.lower()
    extension = os.path.splitext(path)[1].lower()
    depth = path.count("/")

    is_doc = any(h in lowered for h in DOC_PATH_HINTS)
    is_test = any(h in lowered for h in TEST_PATH_HINTS)
    is_source = extension in SOURCE_EXTENSIONS

    # 0. the actual project: primary-language source, outside docs and tests
    if is_source and extension in primary and not is_doc and not is_test:
        return 0
    # 1. the top-level README, short, and often the best answer to
    #    "what is this project"
    if depth == 0 and extension in MARKDOWN_EXTENSIONS:
        return 1
    # 2. primary-language source inside tests (still shows how the API is used)
    if is_source and extension in primary:
        return 2
    # 3. source in another language (React's Rust compiler lands here)
    if is_source:
        return 3
    # 4. documentation
    if extension in PROSE_EXTENSIONS:
        return 4
    return 5


# ============================================================================
# PART 2: Download
# ============================================================================


@dataclass
class RepoContents:
    """Everything we pulled out of the zip, before any splitting."""
    owner: str
    repo: str
    files: dict = field(default_factory=dict)     # path -> text
    tree: list = field(default_factory=list)      # every path, including skipped
    skipped: dict = field(default_factory=dict)   # reason -> count
    total_bytes: int = 0
    truncated: bool = False
    primary_extensions: list = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def fetch_repo(owner: str, repo: str, token: str = None) -> RepoContents:
    """Download the repository as a zip and read the text files out of it.

    A GitHub token is optional and only raises the rate limit (60 requests per
    hour anonymous, 5,000 authenticated). Nothing here needs write access.
    """
    headers = {"Accept": "application/vnd.github+json"}
    token = token or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    response = requests.get(url, headers=headers, timeout=120, stream=True)

    if response.status_code == 404:
        raise ValueError(
            f"GitHub has no public repository at {owner}/{repo}. "
            f"Check the spelling, and note that private repos need a GITHUB_TOKEN."
        )
    if response.status_code == 403:
        raise ValueError(
            "GitHub rate limit reached (60 requests/hour without a token). "
            "Add a GITHUB_TOKEN to raise it to 5,000, or wait a few minutes."
        )
    response.raise_for_status()

    # Read with a ceiling rather than trusting Content-Length, which is absent
    # on a streamed zipball
    buffer = io.BytesIO()
    for block in response.iter_content(chunk_size=1 << 16):
        buffer.write(block)
        if buffer.tell() > MAX_ZIP_BYTES:
            raise ValueError(
                f"{owner}/{repo} is larger than "
                f"{MAX_ZIP_BYTES // (1024 * 1024)} MB, which is too big for this app."
            )

    buffer.seek(0)
    contents = RepoContents(owner=owner, repo=repo)

    with zipfile.ZipFile(buffer) as archive:
        all_entries = [e for e in archive.infolist() if not e.is_dir()]
        # GitHub wraps everything in one directory named owner-repo-sha
        prefix = all_entries[0].filename.split("/")[0] + "/" if all_entries else ""

        # ---- Pass 1: decide what is eligible, WITHOUT reading anything ----
        eligible = []
        for entry in all_entries:
            path = entry.filename[len(prefix):] if entry.filename.startswith(prefix) \
                else entry.filename
            if not path:
                continue

            contents.tree.append(path)

            if _in_ignored_dir(path):
                contents.skipped["ignored directory"] = \
                    contents.skipped.get("ignored directory", 0) + 1
                continue
            if not _is_text_file(path):
                contents.skipped["not a text file"] = \
                    contents.skipped.get("not a text file", 0) + 1
                continue
            if entry.file_size > MAX_FILE_BYTES:
                contents.skipped["too large"] = contents.skipped.get("too large", 0) + 1
                continue
            eligible.append((path, entry, entry.file_size))

        # ---- Pass 2: sort by priority, so the caps keep the RIGHT files ----
        primary = _primary_extensions([(p, s) for p, _, s in eligible])
        contents.primary_extensions = sorted(primary)
        # Within a tier, shallower paths first: fastapi/main.py before
        # fastapi/deeply/nested/helper.py
        eligible.sort(key=lambda item: (_file_priority(item[0], primary),
                                        item[0].count("/"), item[0]))

        # ---- Pass 3: read in that order, up to the cap ----
        for path, entry, size in eligible:
            if len(contents.files) >= MAX_FILES:
                contents.truncated = True
                break

            try:
                raw = archive.read(entry)
            except Exception:
                continue

            # A binary file that slipped through the extension check will be
            # full of replacement characters; skip it rather than embed noise.
            text = raw.decode("utf-8", errors="replace")
            if text.count(chr(0xFFFD)) > len(text) * 0.02:
                contents.skipped["looks binary"] = \
                    contents.skipped.get("looks binary", 0) + 1
                continue

            contents.files[path] = text
            contents.total_bytes += entry.file_size

        if contents.truncated:
            contents.skipped["beyond the file cap"] = len(eligible) - len(contents.files)

    return contents


def _in_ignored_dir(path: str) -> bool:
    return any(part in IGNORE_DIRS for part in path.split("/")[:-1])


def _is_text_file(path: str) -> bool:
    name = path.split("/")[-1]
    extension = os.path.splitext(name)[1].lower()
    if extension in LANGUAGE_BY_EXTENSION or extension in PLAIN_EXTENSIONS:
        return True
    # Files like Dockerfile and Makefile have no extension
    return name in NOTABLE_FILENAMES or name.split(".")[0] in NOTABLE_FILENAMES


# ============================================================================
# PART 3a: Python, chunked with the AST
# ============================================================================
# Python files go through the real parser; everything else uses the
# language-aware character splitter in PART 3b.
#
# WHY THE DIFFERENCE IS WORTH THE EXTRA CODE
# `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` sounds
# structural, but its separator list is just text to search for:
#
#     ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
#
# It never parses anything, so three things go wrong. Measured on psf/requests
# with this app's own settings: only 189 of 467 Python chunks (40%) parsed as
# valid Python on their own, and 9 chunks ended ON a decorator line, stranding
# it from the function it belongs to.
#
#   1. It cuts inside strings and docstrings. A code generator holding a
#      template, or a docstring showing an example, contains "\ndef ", and the
#      splitter happily cuts there, sometimes tearing an unterminated triple
#      quote in half.
#   2. It strands decorators. '\ndef ' matches immediately BEFORE the def, so
#      `@app.get("/health")` is left at the tail of the previous chunk. A
#      health_check function without its route is unrecognisable.
#   3. It abandons boundaries under pressure. Once a function exceeds
#      chunk_size, it falls through to '\n\n', then '\n', then ' '.
#
# The parser has none of those problems, because it knows what things ARE. A
# `def` inside a string is a Constant node, not a definition. A decorator is an
# ATTRIBUTE of the function node (`node.decorator_list`), which is the only
# reason we can know the two belong together. And `end_lineno` gives the exact
# extent, so a chunk is a complete unit by construction.
#
# The cost, stated plainly: `ast` is Python-only. Doing this for JS, TS and Go
# would mean a parser per language (tree-sitter or similar), which is a much
# bigger commitment than this app justifies. So Python, usually the bulk of a
# Python project, gets exact chunks, and the rest gets good-enough ones.


def _split_large_node(node, lines: list) -> list:
    """Break one oversized definition into its child definitions, via the AST.

    For a class this yields a header chunk (the `class` line, its docstring and
    any class-level attributes) followed by one chunk per method. Every piece is
    still a complete syntactic unit, and every piece carries `ClassName.method`
    so a retrieved method announces which class it belongs to.

    A single enormous FUNCTION has no child definitions to split at, so that one
    genuinely does fall back to the character splitter, but it is now one known
    function being divided, not an arbitrary window over the file.
    """
    pieces = []
    children = [c for c in getattr(node, "body", [])
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    if not children:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        start = node.lineno
        if node.decorator_list:
            start = min(start, min(d.lineno for d in node.decorator_list))
        text = "\n".join(lines[start - 1:getattr(node, "end_lineno", node.lineno)])
        return [(part, node.name) for part in splitter.split_text(text)]

    outer_start = node.lineno
    if node.decorator_list:
        outer_start = min(outer_start, min(d.lineno for d in node.decorator_list))

    child_lines = set()
    for child in children:
        child_start = child.lineno
        if child.decorator_list:
            child_start = min(child_start, min(d.lineno for d in child.decorator_list))
        child_lines.update(range(child_start, getattr(child, "end_lineno", child.lineno) + 1))

    # The class header: everything in the class that is not one of its methods
    header_lines = [i for i in range(outer_start, getattr(node, "end_lineno", outer_start) + 1)
                    if i not in child_lines and lines[i - 1].strip()]
    if header_lines:
        header = "\n".join(lines[i - 1] for i in header_lines)
        if len(header.strip()) >= 40:
            pieces.append((header[:CHUNK_SIZE * 2], f"{node.name} (class header)"))

    for child in children:
        child_start = child.lineno
        if child.decorator_list:
            child_start = min(child_start, min(d.lineno for d in child.decorator_list))
        child_end = getattr(child, "end_lineno", child.lineno)
        text = "\n".join(lines[child_start - 1:child_end])
        symbol = f"{node.name}.{child.name}"

        if len(text) > CHUNK_SIZE * 2:
            # A giant method. Recurse once more, then give up to characters
            for part, _ in _split_large_node(child, lines):
                pieces.append((part, symbol))
        else:
            pieces.append((text, symbol))

    return pieces


def chunk_python_ast(path: str, source: str) -> list:
    """Split a Python file at real definition boundaries, using the AST.

    Returns a list of (text, symbol) pairs. Falls back to the character splitter
    if the file does not parse, a Python 2 file, or syntax newer than this
    interpreter, should still be searchable rather than silently vanishing.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [(piece, "") for piece in _split_with_separators(path, source)]

    lines = source.splitlines()
    pieces = []
    covered = set()

    # Only TOP-LEVEL definitions, so a class arrives whole rather than being
    # shredded into its methods. ast.walk() would return both the class and each
    # method, duplicating every method's text inside the class chunk.
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # START AT THE FIRST DECORATOR. This is the whole reason to use the AST
        # rather than text matching, see point 2 in the comment above.
        start = node.lineno
        if node.decorator_list:
            start = min(start, min(d.lineno for d in node.decorator_list))
        end = getattr(node, "end_lineno", node.lineno)

        text = "\n".join(lines[start - 1:end])
        covered.update(range(start, end + 1))

        # A unit that fits stays whole, the common case, and the point of all this.
        if len(text) <= CHUNK_SIZE * 2:
            pieces.append((text, node.name))
            continue

        # Too big to keep whole. Recurse with the PARSER rather than falling back
        # to characters: a large class becomes its methods, each still a complete
        # unit, each labelled `ClassName.method`. Splitting a 6 KB class on
        # character count instead would produce fragments that end mid-`if`.
        pieces.extend(_split_large_node(node, lines))

    # Everything not inside a top-level definition: imports, constants, the
    # module docstring, `if __name__ == "__main__"`. Grouped into one chunk so
    # the file's setup is retrievable as a unit.
    leftover = [i for i in range(1, len(lines) + 1)
                if i not in covered and lines[i - 1].strip()]
    if leftover:
        module_text = "\n".join(lines[i - 1] for i in leftover)
        if len(module_text.strip()) >= 40:
            pieces.insert(0, (module_text[:CHUNK_SIZE * 2], "(module level)"))

    return pieces or [(piece, "") for piece in _split_with_separators(path, source)]


def _split_with_separators(path: str, source: str) -> list:
    """The character-splitter path, used for non-Python files and as a fallback."""
    extension = os.path.splitext(path)[1].lower()
    language = LANGUAGE_BY_EXTENSION.get(extension)
    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    return splitter.split_text(source)


# ============================================================================
# PART 3b: Split
# ============================================================================


def split_repo(contents: RepoContents) -> list:
    """Turn the files into Documents, split with a language-aware splitter.

    Every chunk carries its `source` path in metadata, and that is what makes
    citation possible: the agent can only say "this is in src/app.py" because
    the chunk it retrieved knows it came from there.
    """
    documents = []

    # contents.files is already in priority order (see fetch_repo), so if the
    # chunk cap bites, it bites on the least important files.
    for path, text in contents.files.items():
        if not text.strip():
            continue

        extension = os.path.splitext(path)[1].lower()
        language = LANGUAGE_BY_EXTENSION.get(extension)

        # Python gets the real parser; everything else gets separators.
        if extension == ".py":
            pieces = chunk_python_ast(path, text)
        else:
            pieces = [(piece, "") for piece in _split_with_separators(path, text)]

        for index, (piece, symbol) in enumerate(pieces):
            if not piece.strip():
                continue
            # The symbol goes in the header too, so a retrieved chunk announces
            # which function or class it came from without the model guessing.
            header = f"File: {path}"
            if symbol:
                header += f"  ({symbol})"
            documents.append(Document(
                page_content=f"{header}\n\n{piece}",
                metadata={
                    "source": path,
                    "chunk": index,
                    "symbol": symbol,
                    "language": language.value if language else "text",
                },
            ))

            if len(documents) >= MAX_CHUNKS:
                contents.truncated = True
                return documents

    return documents


# ============================================================================
# PART 4: Index
# ============================================================================


@dataclass
class RepoIndex:
    """A fully indexed repository, ready to be asked questions."""
    contents: RepoContents
    store: InMemoryVectorStore
    documents: list

    @property
    def retriever(self):
        """The retriever the agent's search tool uses.

        Same interface as the Chroma retriever in 4_rag/7_rag_conversational.py -
        `search_kwargs={"k": n}` and all.
        """
        return self.store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def summary(self) -> str:
        return (f"{self.contents.full_name}: {len(self.contents.files)} files indexed "
                f"({len(self.documents)} chunks) out of {len(self.contents.tree)} total")


def build_index(url: str, token: str = None, progress=None,
                api_key: str = None) -> RepoIndex:
    """Fetch, split and embed a repository. This is the whole pipeline.

    `progress` is an optional callback taking a status string, so the Streamlit
    page can show what is happening during the slow part.
    """
    owner, repo = parse_repo_url(url)

    def say(message: str):
        if progress:
            progress(message)

    say(f"Downloading {owner}/{repo}...")
    contents = fetch_repo(owner, repo, token=token)
    if not contents.files:
        raise ValueError(
            f"{owner}/{repo} downloaded, but it contains no text files this app "
            f"can read (it found {len(contents.tree)} files in total)."
        )

    say(f"Splitting {len(contents.files)} files...")
    documents = split_repo(contents)
    if not documents:
        raise ValueError("Nothing left to index after splitting.")

    say(f"Embedding {len(documents)} chunks...")
    # The key is passed EXPLICITLY, never read from the process environment.
    # A visitor's own key must not become the process default for every other
    # visitor, which is exactly what setting os.environ inside a cached
    # function would do.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,
                                  **({"api_key": api_key} if api_key else {}))
    store = InMemoryVectorStore.from_documents(documents, embeddings)

    say("Ready.")
    return RepoIndex(contents=contents, store=store, documents=documents)


# ============================================================================
# PART 5: Structure, without the model
# ============================================================================
# The agent is asked about structure constantly, and a directory tree is
# something a loop can produce exactly. Letting the model guess at it would be
# both slower and wrong.


def build_tree(contents: RepoContents, max_entries: int = 300) -> str:
    """Render the repository layout as an indented tree."""
    interesting = [p for p in sorted(contents.tree) if not _in_ignored_dir(p)]
    truncated = len(interesting) > max_entries
    interesting = interesting[:max_entries]

    lines = [f"{contents.full_name}/"]
    seen_dirs = set()

    for path in interesting:
        parts = path.split("/")
        # Emit any parent directories we have not printed yet
        for depth in range(len(parts) - 1):
            directory = "/".join(parts[:depth + 1])
            if directory not in seen_dirs:
                seen_dirs.add(directory)
                lines.append("  " * (depth + 1) + parts[depth] + "/")
        # A file sits one level deeper than its parent directory. The marker is
        # a single character so indexed and non-indexed files stay aligned.
        marker = " " if path in contents.files else "?"
        lines.append("  " * len(parts) + marker + " " + parts[-1])

    if truncated:
        lines.append(f"  ... and {len(contents.tree) - max_entries} more entries")
    lines.append("")
    lines.append("(entries marked '?' exist in the repo but were not indexed)")
    return "\n".join(lines)
