# 💬 Talk to GitHub

Paste a public GitHub repository link into the box in the middle of the page, and ask it
anything — the folder structure, what a specific file contains, where something is
handled, how big the project is.

Built with **LangChain** (a tool-calling agent, `AgentExecutor`), following the patterns
in `langchain-course`.

## Quick start

```bash
cd talk-github
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add your OPENAI_API_KEY
streamlit run streamlit_app.py
```

You can also paste a key straight into the sidebar instead of using a `.env` file, which
is what a visitor to the deployed app does.

## Files

| File | What it does |
|---|---|
| `streamlit_app.py` | The UI: the centred repo field, then the chat |
| `repo_index.py` | Download the repo as a zip, split it, embed it into a vector store |
| `agent.py` | The LangChain agent and its four tools |
| `requirements.txt` | Deployment dependencies — deliberately short |

## How it works

```
        paste a repo link
                |
                v
    download zipball from GitHub      (one HTTPS request, no `git` needed)
                |
                v
    chunk:  .py  -> real AST parser   (complete functions and classes)
            rest -> language splitter (RecursiveCharacterTextSplitter)
                |
                v
    embed into InMemoryVectorStore    (per visitor, discarded on exit)
                |
                v
    LangChain agent with 4 tools  <--->  your questions
```

### Chunking: a real parser for Python

`RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` sounds structural, but
its separator list is just text to search for:

```python
['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
```

It never parses anything, and three things go wrong as a result:

1. **It cuts inside strings.** A code generator holding a template, or a docstring showing
   an example, contains `\ndef ` — and it happily cuts there, sometimes tearing an
   unterminated `"""` in half.
2. **It strands decorators.** `\ndef ` matches immediately *before* the `def`, so
   `@app.get("/health")` is left at the tail of the previous chunk. A `health_check`
   function without its route is unrecognisable.
3. **It abandons boundaries under pressure.** Once a unit exceeds `chunk_size` it falls
   through to `\n\n`, then `\n`, then `' '`.

So `.py` files go through Python's own `ast` module instead. A `FunctionDef` node knows
its exact extent (`end_lineno`), and a decorator is an *attribute* of that node
(`node.decorator_list`) — which is the only reason you can know the two belong together.
A `def` inside a string is a `Constant`, not a definition.

An oversized class is split at its **methods**, again via the parser, each labelled
`ClassName.method`. Only a single oversized function with no child definitions falls back
to characters.

Measured on `psf/requests` (the app's real settings):

| | separators | AST |
|---|---|---|
| Definition chunks that are complete units | 40% of chunks parsed standalone | **94%** |
| Chunks stranding a decorator | 9 | **0** |
| Chunks that know their symbol name | 0 | **all of them** |

**The cost, stated plainly:** `ast` is Python-only. Doing this for JS, TS and Go means a
parser per language (tree-sitter or similar), which is a bigger commitment than this app
justifies — so Python gets exact chunks and the other ~20 languages get good-enough ones.

### Why an agent instead of a retrieval chain

`4_rag/7_rag_conversational.py` in the course builds a conversational retrieval chain,
which retrieves then answers on every single turn. That is right for "what does this
document say about X", but questions about a repository do not all have that shape:

| Question | What it actually needs |
|---|---|
| "what's the folder structure?" | the directory tree — no retrieval |
| "show me the code in `app.py`" | that file's contents — no retrieval |
| "where is authentication handled?" | retrieval, definitely |
| "how many Python files?" | counting — no retrieval |

So the agent picks the tool that suits the question, and **three of its four tools never
touch the vector store**:

| Tool | Exact or fuzzy? |
|---|---|
| `search_code` | fuzzy — semantic search over the chunks |
| `repo_structure` | **exact** — the real directory tree |
| `read_file` | **exact** — the real file contents |
| `repo_stats` | **exact** — real counts, languages, sizes |

That split matters. A tree and a file's text are things a loop can produce perfectly, so
the model is never asked to guess at them — it is handed the real thing and asked to
explain it. `read_file` is also forgiving about paths: ask for `app.py` when the file is
`src/app.py` and it matches on the suffix and tells you the real path, rather than
failing and making the agent guess again.

## Two decisions taken for deployment

**No `git clone`.** Repositories are downloaded as a zip from
`api.github.com/repos/{owner}/{repo}/zipball` — one HTTPS request, no subprocess, and no
dependency on a `git` binary existing on the host.

**No Chroma.** The course uses Chroma, and locally that is the right call. On a hosted
app it is not:

- Chroma needs SQLite 3.35+, and Streamlit Community Cloud ships an older SQLite. Using
  it there means adding `pysqlite3-binary` plus a module-swap hack at the top of the app.
- The filesystem on these hosts is ephemeral, and every visitor indexes a *different*
  repo, so persisting an index buys nothing.

`InMemoryVectorStore` is a first-class LangChain vector store with the same
`.as_retriever()` interface, so nothing about the lesson changes — only where the vectors
live. One index per visitor, discarded when they leave.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo (its own repo, or this one with `talk-github` as the
   app path).
2. On [share.streamlit.io](https://share.streamlit.io) create an app pointing at
   `streamlit_app.py`.
3. In **Settings → Secrets**, paste:

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

   **The quotes matter.** That box takes TOML, so an unquoted `KEY=value` line loads
   nothing at all — which from the outside looks identical to "no secrets configured".

`GITHUB_TOKEN` is optional and only raises the GitHub rate limit from 60 to 5,000
requests an hour. Public repositories work without it.

Visitors without a configured key can paste their own into the sidebar, so the app is
still usable if the shared key runs out.

## Limits, and why each exists

| Limit | Value | Why |
|---|---|---|
| Download size | 60 MB | one person pasting a monorepo would otherwise hang the app |
| Single file size | 400 KB | a generated lockfile or minified bundle is noise, not code |
| Files indexed | 1,200 | keeps indexing to a few seconds |
| Chunks | 4,000 | caps the embedding bill per repo |
| File shown to the model | 12,000 chars | a 4,000-line file would eat the whole context window |
| Chat history | last 12 messages | the full history is resent every turn, so cost grows with length |

Binary files, dependency directories (`node_modules`, `.venv`, `dist`, …) and anything
that is not text are skipped. **The app tells you what it skipped**, in the sidebar and in
the agent's own context, so it says "that file was not indexed" instead of inventing an
answer about it.

## What is verified

Tested end-to-end through Streamlit's own `AppTest`, driving it as a user would:

- **Landing screen** renders the centred field, Start button and example repos, with no
  chat input until a repo is chosen.
- **Typing a URL and pressing Start** indexed `pypa/sampleproject` (9 files, 18 chunks),
  swapped to the chat view, and populated the sidebar metrics.
- **A structure question** returned the exact directory tree.
- **A follow-up that depends on history** — "show me the code in the file that has
  `add_one`" — resolved to `src/sample/simple.py` and showed the real code.
- Transcript, LangChain history and rendered bubbles all stayed in step (4 / 4 / 4).

On a real mid-size repository (`psf/requests`, 79 files → 709 chunks in 5.7s), asked "how
are HTTP redirects followed?", the agent found `src/requests/sessions.py` and correctly
described `get_redirect_target`, `rebuild_method` and `rebuild_auth`.

URL parsing accepts full URLs, trailing slashes, `.git` suffixes, SSH remotes,
`/tree/main/...` deep links and bare `owner/repo`; it rejects `github.com/onlyowner` and
non-URLs with a readable message. A missing repo, a rate limit, a bad API key and an
exhausted quota each produce a message a stranger can act on rather than a stack trace.

## Known limitations

- **Retrieval is plain vector search.** Docs, tests and implementation compete in one
  space, so "how does X work" sometimes surfaces a test or a changelog entry first. The
  agent usually recovers by calling `read_file`, but a reranker would be the real fix.
- **One repository per conversation.** Switching repos clears the chat.
- **The index is in memory**, so it is rebuilt when the host recycles the process, and
  the server holds at most four repos at once (`max_entries=4`).
- **Read-only.** The agent cannot commit, open issues, or change anything.
- **No authentication on the page.** Fine for a personal deployment; anyone with the URL
  can spend the configured key, which is why the sidebar accepts a visitor's own.
