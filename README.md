# 💬 Talk to GitHub

Paste a public GitHub repository link into the box in the middle of the page, and ask it
anything, the folder structure, what a specific file contains, where something is
handled, how big the project is.

Built with **LangChain** (a tool-calling agent, `AgentExecutor`), following the patterns
in `langchain-course`.

> **On the shared demo key:** the in-app message limit stops one visitor draining the
> demo allowance in a sitting. **The hard spend cap on the OpenAI project is what stops
> money being lost.** The two are different things and are not confused in the code. See [Running this safely in public](#running-this-safely-in-public).

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
| `guard.py` | Keys, the usage gate, and failure classification |
| `test_guard.py` | Tests for all of that, no key or network needed |
| `requirements.txt` | Deployment dependencies, deliberately short |

## How it works

```
        paste a repo link
                |
                v
    download zipball from GitHub      (one HTTPS request, no `git` needed)
                |
                v
    chunk: .py/.ipynb -> real AST parser   (complete functions and classes)
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
   an example, contains `\ndef `, and it happily cuts there, sometimes tearing an
   unterminated `"""` in half.
2. **It strands decorators.** `\ndef ` matches immediately *before* the `def`, so
   `@app.get("/health")` is left at the tail of the previous chunk. A `health_check`
   function without its route is unrecognisable.
3. **It abandons boundaries under pressure.** Once a unit exceeds `chunk_size` it falls
   through to `\n\n`, then `\n`, then `' '`.

So `.py` files go through Python's own `ast` module instead. A `FunctionDef` node knows
its exact extent (`end_lineno`), and a decorator is an *attribute* of that node
(`node.decorator_list`), which is the only reason you can know the two belong together.
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
justifies, so Python gets exact chunks and the other ~20 languages get good-enough ones.

### Why an agent instead of a retrieval chain

`4_rag/7_rag_conversational.py` in the course builds a conversational retrieval chain,
which retrieves then answers on every single turn. That is right for "what does this
document say about X", but questions about a repository do not all have that shape:

| Question | What it actually needs |
|---|---|
| "what's the folder structure?" | the directory tree, no retrieval |
| "show me the code in `app.py`" | that file's contents, no retrieval |
| "where is authentication handled?" | retrieval, definitely |
| "how many Python files?" | counting, no retrieval |

So the agent picks the tool that suits the question, and **three of its four tools never
touch the vector store**:

| Tool | Exact or fuzzy? |
|---|---|
| `search_code` | fuzzy, semantic search over the chunks |
| `repo_structure` | **exact**, the real directory tree |
| `read_file` | **exact**, the real file contents |
| `repo_stats` | **exact**, real counts, languages, sizes |

That split matters. A tree and a file's text are things a loop can produce perfectly, so
the model is never asked to guess at them, it is handed the real thing and asked to
explain it. `read_file` is also forgiving about paths: ask for `app.py` when the file is
`src/app.py` and it matches on the suffix and tells you the real path, rather than
failing and making the agent guess again.

## Two decisions taken for deployment

**No `git clone`.** Repositories are downloaded as a zip from
`api.github.com/repos/{owner}/{repo}/zipball`, one HTTPS request, no subprocess, and no
dependency on a `git` binary existing on the host.

**No Chroma.** The course uses Chroma, and locally that is the right call. On a hosted
app it is not:

- Chroma needs SQLite 3.35+, and Streamlit Community Cloud ships an older SQLite. Using
  it there means adding `pysqlite3-binary` plus a module-swap hack at the top of the app.
- The filesystem on these hosts is ephemeral, and every visitor indexes a *different*
  repo, so persisting an index buys nothing.

`InMemoryVectorStore` is a first-class LangChain vector store with the same
`.as_retriever()` interface, so nothing about the lesson changes, only where the vectors
live. One index per visitor, discarded when they leave.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io) create an app pointing at
   `streamlit_app.py`.
3. In **Settings → Secrets**, paste:

   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   ```

   **The quotes matter.** That box takes TOML, so an unquoted `KEY=value` line loads
   nothing at all, which from the outside looks identical to "no secrets configured".
   When no key is found the app prints the secret **names** it can see (never values),
   which distinguishes those two cases.

`GITHUB_TOKEN` is optional, operator-only, and just raises the GitHub rate limit from 60
to 5,000 requests an hour. There is deliberately **no visitor-facing input** for it: every
repository this app can reach is public, so a visitor never needs one, and asking a
stranger for a GitHub credential they do not need is worse than the small rate-limit
benefit.

## Running this safely in public

This app holds an API key that costs real money, behind a public URL with no login.
Five layers, and **only one of them is a security boundary**.

### Layer 1, the OpenAI project. This is the real protection.

Enforced by OpenAI, so it holds even if this code is wrong or bypassed entirely.

| Setting | Value |
|---|---|
| Monthly spend limit | **$5.00**, with **"Enforce a hard limit" ON** |
| Alerts | 50% and 80% |
| Model allow-list | exactly `gpt-5.4-mini-2026-03-17` and `text-embedding-3-small` |
| Rate limits | 10 RPM on the default row, 20 RPM chat, 20 RPM embeddings |
| Key | project-scoped, not account-wide |

The hard-limit toggle is the whole step: off, the $5 is only a notification and requests
keep going; on, requests actually stop.

**A consequence of that RPM ceiling.** One agent turn costs at least *two* requests, one
for the model to choose a tool, one to answer. At 20 RPM shared across everyone, that is
roughly **ten conversation turns per minute for the whole app**. So HTTP 429 from
throttling is a *normal operating condition* here, which is why the code classifies it
separately from a spent budget.

### Layer 2, getting the key in without leaking it

`.env` and `.streamlit/secrets.toml` are gitignored, and full git history has been
scanned for key-shaped strings (zero found). The key is read via `st.secrets`
**explicitly**, with an `os.environ` fallback, because Streamlit only copies secrets
into the environment *after* something touches `st.secrets`, so an app that reads
`os.environ` directly finds it empty on Cloud and crashes before rendering anything.

### Layer 3, the in-app usage gate

There are **two** allowances, because the two operations cost very different amounts. A
chat turn is roughly **$0.001**. Loading a large repository is roughly **$0.024** of
embeddings, about twenty chat turns of cost in a single click.

| Allowance | Per visitor | Guards |
|---|---|---|
| Messages | **3** | each agent turn |
| Repositories | **2** | each *new* index build, the expensive call |

- Metering only the messages was a real hole, and it guarded the cheap operation. With the
  chat gate fully closed, a visitor could return to the landing screen and index repository
  after repository forever. Driven through `AppTest`, `build_index` ran on **four attempts
  out of four** while the sidebar read `0`. `test_gate.py` now asserts against the number of
  index builds, not against the displayed number.
- **A spent message allowance also blocks indexing.** Loading a repo you cannot then ask
  about is embedding budget spent for nothing.
- **Reopening a repo you already loaded is free.** "Use a different repo" returns to the
  landing screen, so returning to something you just had on screen must not cost a second
  slot, and while the index is cached it costs nothing to serve.
- Both counters live in `@st.cache_resource`, **not session state**, a per-session counter
  resets on reload and in a new tab, which is no limit at all.
- A visitor on **their own key is never counted** against either allowance.
- Both are charged **only after success**, so a 404, an oversized repo, a bad folder path or
  a failed turn costs nothing and there is never anything to refund.
- With no key configured the input is **disabled with a plain sentence**, not a traceback.
- The numbers are shown as **bare figures**, not `3 / 3`, which was easy to misread as three
  already used out of three.

Both counters live in the app process, so **restarting or redeploying the app resets them**,
and Community Cloud sleeps idle apps. That is by design rather than an oversight: the gate is
cost smoothing, and the $5 hard cap is the layer that actually bounds the loss.

### Layer 4, failing without leaking or crashing

OpenAI answers two completely different situations with HTTP 429: *your budget is gone*
(terminal) and *you are going too fast* (transient). The terminal case is checked
**first**, because a quota error also contains the string "429", get that order wrong and
every spent budget reads as "try again shortly", inviting a visitor to hammer a wall.

A throttle **never** closes the gate or spends a turn. The raw exception goes to the
server log only; the page gets a sentence written by us. `test_guard.py` verifies all of
this against real OpenAI error strings, including the quota-error-containing-429 case.

### Layer 5, what the repository publishes

No dataset ships, so there is nothing to scrub. Deployment dependencies contain only what
the app imports (74 packages, verified by installing `requirements.txt` alone in a clean
environment).

### What none of this protects against

Stated plainly, because each item is a reason Layer 1 is the one that matters:

- **IP is spoofable.** Streamlit's own docs say `st.context.ip_address` "should not be used
  for security measures because it can easily be spoofed."
- **IPv6 makes per-IP counting weak** even without spoofing, one user gets a huge range.
- **Everyone behind one NAT shares an entry**, so an office shares three messages.
- **The counter resets** when the app sleeps or redeploys.
- **Nothing stops prompt-directed misuse.** The model allow-list bounds how expensive it
  can get; the system prompt's topic restriction is politeness, not enforcement.

### Rotating the key

Because the key is project-scoped, rotation is cheap: create a new key in the same
project, update the Streamlit secret, delete the old one. Rotate immediately if it has
ever appeared in a screenshot, a commit, a pasted log, or a chat with a tool.

## The example repositories

`facebook/react`, `langchain-ai/langchain` and `tiangolo/fastapi`, the ones people
actually want to poke at.

They are only usable because of **file prioritisation**, and that fix came out of a bug
worth describing. The caps used to be applied in *zip order*, roughly alphabetical, so
which files survived was arbitrary:

| repo | chunk languages before | after |
|---|---|---|
| `facebook/react` | `rust:3332, ts:104` | `js:4000` |
| `tiangolo/fastapi` | `markdown:3756, js:25` | `python:3968` |
| `langchain-ai/langchain` | `python:3881` (arbitrary files) | `python:4000`, all of `libs/` |

React's repo carries a large Rust compiler under `compiler/`, 4,210 files against 2,154
under `packages/`. Reading in alphabetical order, the 1,200-file cap filled up inside the
compiler and **never reached React itself**, so asking about hooks retrieved Rust. FastAPI's repo is dominated by its translated documentation tree.

The repositories were never the problem; the selection was. `repo_index.py` now works out
what the project is mostly *written in*, weighted by bytes, excluding docs and test trees
so a docs-heavy project cannot elect Markdown as its language, then keeps files in
priority order:

```
0  primary-language source, outside docs and tests   <- the actual project
1  the top-level README
2  primary-language source inside tests              <- shows how the API is used
3  source in another language                        <- React's Rust compiler
4  documentation
```

Retrieval for *"useState hook"* now returns `packages/react-reconciler/src/ReactFiberHooks.js`,
and the agent names `mountState` / `updateState` correctly.

**What this costs.** All three truncate, the sidebar says so, and the agent is instructed
to admit a partial view rather than describe the whole project. Each indexes in 15-20
seconds and costs roughly **$0.024**, about sixteen times a small repository, which is
worth knowing against a $5 cap. Small repos are unaffected by the change: `psf/requests`
still indexes 79 files to 933 chunks with no truncation.

## Large repositories

This app is built for small to medium projects, where it indexes everything. On a large
repository it indexes a prioritised slice and says so. Measured file coverage:

| repo | files | indexed | coverage |
|---|---|---|---|
| `psf/requests` | 130 | 79 | **100%** |
| `pallets/click` | 166 | 148 | **100%** |
| `langchain-ai/langchain` | 3,044 | 1,200 | 43% |
| `django/django` | 7,086 | 1,200 | 31% |
| `facebook/react` | 7,205 | 1,200 | 17% |
| `vercel/next.js` | . | 0 | refused, over 60 MB |

### Point at a folder for complete coverage

Paste a folder link and only that subtree is indexed, completely:

```
github.com/facebook/react/tree/main/packages/react
```

| target | files | chunks | coverage | cost |
|---|---|---|---|---|
| `facebook/react` | 1,200 | 4,000 | 17% | $0.024 |
| `react/tree/main/packages/react` | 84 | 589 | **100%** | $0.004 |
| `django/tree/main/django/db` | 104 | 3,104 | **100%** | $0.019 |

The branch or tag in the link is honoured, so `/tree/master/libs/core` indexes `master`.
The agent is also told when its view is folder-scoped, so it refuses to speak for the rest
of the project. Asked about the DOM renderer while scoped to `packages/react` it replied:
*"The DOM renderer is not in `packages/react`. In this folder, I can only see the public
React package implementation and exports."*

### What this does not fix

Being honest about the ceiling, because folder scoping is a workaround rather than a
solution:

- **Cross-cutting questions break.** "How does the reconciler talk to the DOM renderer?"
  spans two packages; scope to one and the other is invisible.
- **You have to know where to look**, which is often the very thing you wanted to ask.
- **The 60 MB download ceiling is unchanged.** GitHub serves an archive of the whole tree,
  so the zip is fetched before any filtering. `vercel/next.js` is still out of reach.
- **A big folder still truncates.** `packages/react-dom` needs 4,272 chunks against a
  4,000 cap.

Full coverage of a large repository is blocked by three separate limits, not one. Measured
for `facebook/react`: **37,709 chunks, $0.226 per index, 137 seconds of embedding, and
927 MB of RAM** as Python-list vectors. Memory is the easy one, a numpy float16 array is
eight times smaller. Cost is the hard one: at $0.226 an index, a $5 budget buys 22 of
them. Doing this properly needs a persistent vector store so a repo is indexed once and
shared, not per session, which means paid hosting rather than a free tier.

For a $5 public demo, truncation with good file prioritisation plus honest reporting is
the right trade.

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

Two categories are refused **on purpose**, and they beat the extension allow-list:

- **Generated files.** A lockfile is thousands of lines of hashes, a source map is one line
  of coordinates, a snapshot is serialised output. They match almost any query and answer
  nothing. `jquery.min.js` is genuinely JavaScript and is still noise, so `.min.` is
  refused by name.
- **Certificates, keys and credential files** (`.pem`, `.key`, `.env`, `id_rsa`, `.netrc`,
  …). A private key is plain text, so nothing else here would have stopped one being
  embedded, and once a chunk is in the store the agent can be asked to quote it. The keys
  in a public repository's fixtures are throwaway, but an app that will read any key file
  aloud on request is the wrong habit to build. `psf/requests` alone has 12.

### Notebooks

`.ipynb` files are indexed as the Python inside them. This is not a matter of adding an
extension, because the raw file looks like this:

```json
{"cell_type": "code", "execution_count": 12,
 "outputs": [{"data": {"image/png": "iVBORw0KGgoAAA... 400 KB of base64 ..."}}],
 "source": ["df = pd.read_csv('books.csv')\n"]}
```

One line of real code, wrapped in metadata and a rendered PNG. So the JSON is parsed and
only `source` is kept: markdown cells become comments, because a notebook's prose is often
the only explanation of what the code is for, and `%matplotlib` and `!pip` lines are
commented out rather than deleted, so the reader still sees that the notebook installs
pandas while `ast.parse` sees valid Python. The result goes through the same AST chunker as
a `.py` file.

Notebooks also get a much higher size ceiling (8 MB against 400 KB), because most of a
committed `.ipynb` is output that is about to be discarded, and judging one by its file size
rejects a 40 KB program for carrying a 3 MB plot.

**Why this mattered more than it sounds.** In a data science repository the notebooks *are*
the project. Before this, `Semantic_Book_Recommender` indexed 5 of 13 files: a README, a
`requirements.txt`, a `.gitignore`, a LICENSE and one `.py`. The four notebooks holding the
entire pipeline were counted as "not a text file" and the app reported, accurately, that it
could not see much. It now indexes 9 files and 29 chunks, and can quote the emotion
classifier out of `sentiment-analysis.ipynb`.

### Saying what was skipped, in English

The sidebar used to read `Not indexed: 7 not a text file, 1 too large`. Every word of that
is either jargon or, for the four notebooks it was counting, false. Skips are now bucketed
and phrased: `Skipped 4 of 13 files: 3 data files, 1 file over 400 KB`.

The structure tool had the same problem in a different form. It marked every line of the
tree with `?` for "not indexed" and explained the convention in a footnote, so on a data
science repo the answer to "show me the structure" was a wall of question marks, and the
agent repeated the convention back. The tree is now a plain tree, and the unreadable files
are **named once underneath**. Same information, and it reads like output rather than an
apology.

## What is verified

Tested end-to-end through Streamlit's own `AppTest`, driving it as a user would:

- **Landing screen** renders the centred field, Start button and example repos, with no
  chat input until a repo is chosen.
- **Typing a URL and pressing Start** indexed `pypa/sampleproject` (9 files, 18 chunks),
  swapped to the chat view, and populated the sidebar metrics.
- **A structure question** returned the exact directory tree.
- **A follow-up that depends on history**, "show me the code in the file that has
  `add_one`", resolved to `src/sample/simple.py` and showed the real code.
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
