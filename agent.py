# The Agent: a LangChain tool-calling agent that answers questions about one repo
#
# This follows the langchain-course pattern from
# 5_agents_and_tools/tools_deep_dive/1_tool_constructor.py exactly:
#
#     prompt   = ChatPromptTemplate.from_messages([...])
#     agent    = create_tool_calling_agent(llm=..., tools=..., prompt=...)
#     executor = AgentExecutor.from_agent_and_tools(agent=..., tools=...)
#     executor.invoke({"input": question, "chat_history": history})
#
# WHY AN AGENT AND NOT A PLAIN RETRIEVAL CHAIN
# 4_rag/7_rag_conversational.py builds a conversational retrieval chain, and for
# "what does this document say about X" that is the right shape: retrieve, then
# answer, every single time.
#
# Questions about a repository do not all have that shape:
#
#   "what's the folder structure?"      -> no retrieval needed, list the tree
#   "show me the code in app.py"        -> no retrieval needed, read that file
#   "where is authentication handled?"  -> retrieval, definitely
#   "how many Python files are there?"  -> counting, not retrieval
#
# A fixed chain would embed the question and hope for every one of those. An
# agent picks the tool that suits the question, which is why three of the four
# tools below do not touch the vector store at all.
#
# The three non-retrieval tools are also EXACT. A directory tree and a file's
# contents are things a loop can produce perfectly, so the model is never asked
# to guess at them, it is handed the real thing and asked to explain it.

import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

import repo_index

# Load THIS directory's .env, explicitly.
#
# load_dotenv() with no argument walks UP the directory tree until it finds a
# .env, so an app sitting inside a larger project silently picks up the parent's
# key. That is wrong twice over: locally you cannot tell whether the app is
# configured or is borrowing someone else's credentials, and it hides a missing
# key that would fail on deployment. Load only our own file.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# The chat model, pinned to the DATED SNAPSHOT that the OpenAI project's model
# allow-list names. An alias would silently change behaviour when the provider
# repoints it; a mismatch with the allow-list fails at the first request.
MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini-2026-03-17")

# How much of a file we will paste into the model's context at once. A 4,000-line
# file would otherwise eat the whole window and crowd out the conversation.
MAX_FILE_CHARS = 12000


# ============================================================================
# PART 1: The system prompt
# ============================================================================
# Two halves: what the agent is for, and what it must not do. The second half is
# the important one, an agent with a "read the code" tool sounds trustworthy,
# and will still confidently describe a file it never opened if you let it.

SYSTEM_PROMPT = """You are a helpful engineer who has just read through the GitHub \
repository {repo_name} and is explaining it to a colleague.

WHAT YOU KNOW
{repo_facts}

YOUR TOOLS
- search_code: find relevant code or docs by meaning. Use it for "where is X", \
"how does Y work", "why is Z done this way".
- repo_structure: the exact directory tree. Use it for any question about layout, \
folders, or what files exist. Never guess at structure.
- read_file: the full contents of one file, by its exact path. Use it whenever \
someone asks about a specific file or wants to see actual code.
- repo_stats: file counts, languages, and sizes across the whole repository.

HOW TO ANSWER
- Always name the file path you are talking about, like `src/app.py`.
- When you show code, show what read_file actually returned. Never retype it \
from memory and never fill in gaps.
- If a tool returns nothing useful, say so plainly. Do not fall back on what you \
happen to know about a popular library that shares this name.
- Some files in this repository were not indexed (binaries, very large files, \
dependency folders). If a question is about one of those, say it was not indexed \
rather than guessing.
- Keep answers conversational and reasonably short. Use a code block when showing \
code, and plain prose otherwise.
- Write with ordinary punctuation. Do not use em dashes or a spaced hyphen as a \
sentence break; use a comma, a colon, or a full stop instead. Hyphens inside \
compound words like "read-only" and inside file names are fine.
- You are read-only. You cannot modify, commit to, or open issues on this \
repository."""


def build_repo_facts(index) -> str:
    """A short factual block for the prompt, so the agent knows what it has.

    This is injected on every turn rather than being retrieved, because it is
    small, always relevant, and stops the commonest failure: the agent talking
    about "the codebase" when it has only seen a fraction of it.
    """
    contents = index.contents
    languages = {}
    for document in index.documents:
        language = document.metadata.get("language", "text")
        languages[language] = languages.get(language, 0) + 1

    top = sorted(languages.items(), key=lambda item: -item[1])[:6]

    lines = [
        f"- Repository: {contents.full_name}",
    ]
    if contents.subpath:
        # Without this the agent happily answers as though it had seen the whole
        # project, when it has only seen one folder of it.
        lines.append(
            f"- IMPORTANT: you are looking at ONLY the folder "
            f"'{contents.subpath}' inside {contents.owner}/{contents.repo}. You "
            f"have NOT seen the rest of the repository. If a question is about "
            f"code outside this folder, say that you can only see this folder "
            f"and suggest the visitor point the app at the relevant one."
        )
    lines += [
        f"- {len(contents.files)} text files indexed, out of "
        f"{len(contents.tree)} files in the repository",
        f"- {len(index.documents)} searchable chunks",
        f"- Mostly: {', '.join(f'{name} ({count})' for name, count in top)}",
    ]
    if contents.skipped:
        reasons = ", ".join(f"{count} {reason}" for reason, count in contents.skipped.items())
        lines.append(f"- NOT indexed: {reasons}")
    if contents.truncated:
        lines.append("- This repository was large enough that indexing stopped early, "
                     "so your view of it is incomplete. Say so if it matters.")
    return "\n".join(lines)


# ============================================================================
# PART 2: The tools
# ============================================================================


def build_tools(index) -> list:
    """Build the four tools, closed over one indexed repository.

    Written with `Tool(...)` and single-string inputs, the same shape as
    5_agents_and_tools/1_agent_and_tools_basics.py. Single-string arguments are
    deliberate here: every one of these takes one obvious thing (a query, a
    path, nothing), so a schema would add ceremony without adding clarity.
    """
    retriever = index.retriever

    def search_code(query: str) -> str:
        """Semantic search over the indexed repository."""
        query = (query or "").strip()
        if not query:
            return "Please give me something to search for."

        documents = retriever.invoke(query)
        if not documents:
            return "No matching code found in the index."

        blocks = []
        for document in documents:
            source = document.metadata.get("source", "unknown")
            blocks.append(f"--- {source} ---\n{document.page_content}")
        return "\n\n".join(blocks)

    def repo_structure(_: str = "") -> str:
        """The exact directory tree. Takes no meaningful argument."""
        return repo_index.build_tree(index.contents)

    def read_file(path: str) -> str:
        """Return one file's full text, by exact path.

        Being forgiving here matters. The model will often ask for `app.py` when
        the file is `src/app.py`, or guess a path from a search result. Rather
        than returning "not found" and making it try again, match on the suffix
        and tell it the real path.
        """
        path = (path or "").strip().strip("`'\" ")
        if not path:
            return "Please give me a file path."

        files = index.contents.files

        if path in files:
            matched = path
        else:
            # Suffix match, then basename match
            candidates = [p for p in files if p.endswith(path)]
            if not candidates:
                candidates = [p for p in files if p.split("/")[-1] == path.split("/")[-1]]
            if not candidates:
                near = [p for p in files if path.lower() in p.lower()][:5]
                hint = f" Did you mean: {', '.join(near)}?" if near else ""
                return (f"No indexed file at '{path}'.{hint} "
                        f"Use repo_structure to see what exists.")
            if len(candidates) > 1:
                return (f"'{path}' matches several files: {', '.join(candidates[:8])}. "
                        f"Ask again with the full path.")
            matched = candidates[0]

        text = files[matched]
        line_count = len(text.splitlines())

        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n\n... (file truncated for length)"

        return f"File: {matched} ({line_count} lines)\n\n{text}"

    def repo_stats(_: str = "") -> str:
        """Counts and sizes across the whole repository."""
        contents = index.contents

        by_extension = {}
        for path in contents.files:
            extension = os.path.splitext(path)[1].lower() or "(no extension)"
            by_extension[extension] = by_extension.get(extension, 0) + 1

        top_directories = {}
        for path in contents.files:
            directory = path.split("/")[0] if "/" in path else "(root)"
            top_directories[directory] = top_directories.get(directory, 0) + 1

        largest = sorted(contents.files.items(), key=lambda kv: -len(kv[1]))[:5]

        lines = [
            f"Repository: {contents.full_name}",
            f"Files indexed: {len(contents.files)} of {len(contents.tree)} total",
            f"Indexed size: {contents.total_bytes / 1024:.0f} KB",
            f"Searchable chunks: {len(index.documents)}",
            "",
            "By extension: " + ", ".join(
                f"{extension} x{count}"
                for extension, count in sorted(by_extension.items(), key=lambda kv: -kv[1])[:12]
            ),
            "Top-level entries: " + ", ".join(
                f"{directory} ({count})"
                for directory, count in sorted(top_directories.items(), key=lambda kv: -kv[1])[:10]
            ),
            "",
            "Largest indexed files:",
        ]
        for path, text in largest:
            lines.append(f"  {path}: {len(text.splitlines())} lines")

        if contents.skipped:
            lines.append("")
            lines.append("Not indexed: " + ", ".join(
                f"{count} {reason}" for reason, count in contents.skipped.items()
            ))
        return "\n".join(lines)

    return [
        Tool(
            name="search_code",
            func=search_code,
            description=(
                "Search the repository by meaning. Input: a description of what you "
                "are looking for, e.g. 'where user login is validated'. Use for "
                "'where is X', 'how does Y work', 'why is Z like this'."
            ),
        ),
        Tool(
            name="repo_structure",
            func=repo_structure,
            description=(
                "The exact directory tree of the repository. Input: nothing (pass an "
                "empty string). Use for any question about folders, layout, or which "
                "files exist. This is exact, never guess at structure instead."
            ),
        ),
        Tool(
            name="read_file",
            func=read_file,
            description=(
                "Read one file's full contents. Input: the file path, e.g. "
                "'src/sample/simple.py'. Use whenever someone asks about a specific "
                "file or wants to see real code. Always prefer this over recalling "
                "what a file 'probably' contains."
            ),
        ),
        Tool(
            name="repo_stats",
            func=repo_stats,
            description=(
                "Counts, languages and sizes across the whole repository. Input: "
                "nothing (pass an empty string). Use for 'how big is this', 'what "
                "languages', 'how many files'."
            ),
        ),
    ]


# ============================================================================
# PART 3: Assembling the agent
# ============================================================================


def build_agent(index, model_name: str = MODEL, verbose: bool = False,
                api_key: str = None) -> AgentExecutor:
    """Build the AgentExecutor for one indexed repository.

    The prompt has FOUR slots and all four are required by
    create_tool_calling_agent:
      system            - the instructions, with repo facts baked in
      chat_history      - previous turns, so follow-ups work
      input             - this turn's question
      agent_scratchpad  - where the agent keeps its tool calls and results
                          while it works. Omit it and construction fails.
    """
    tools = build_tools(index)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]).partial(
        repo_name=index.contents.full_name,
        repo_facts=build_repo_facts(index),
    )

    # Key passed explicitly, for the reason in repo_index.build_index
    llm = ChatOpenAI(model=model_name, temperature=0,
                     **({"api_key": api_key} if api_key else {}))

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        # A question about structure plus two file reads is a realistic chain;
        # beyond about six steps the agent is usually going in circles.
        max_iterations=6,
        # Return whatever it has rather than raising when it hits the ceiling
        early_stopping_method="force",
    )


def ask(executor: AgentExecutor, question: str, chat_history: list) -> str:
    """Run one turn and return the answer text.

    `chat_history` is a list of LangChain message objects, exactly as in
    4_rag/7_rag_conversational.py. We pass the whole history in and the caller
    appends to it afterwards.
    """
    result = executor.invoke({"input": question, "chat_history": chat_history})
    output = result.get("output", "")

    # A tool-calling model can return its answer as a LIST OF CONTENT BLOCKS
    # rather than a string. Rendering that directly puts raw dictionaries on the
    # user's screen, so pull the text out wherever it might appear.
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        output = "\n".join(p for p in parts if p).strip()

    return output or "I could not produce an answer for that."
