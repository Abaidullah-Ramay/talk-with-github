# Talk to GitHub: the Streamlit app
#
# Paste a repository link in the box in the middle of the page. The app downloads
# it, indexes it, and then you can ask it anything: the folder structure, what a
# specific file contains, where something is handled, how big the project is.
#
# Run locally:
#   streamlit run streamlit_app.py
#
# THE EXECUTION MODEL, WHICH SHAPES EVERY DECISION BELOW
# Streamlit re-runs this WHOLE FILE from top to bottom on every interaction.
# Local variables do not survive between runs, only `st.session_state` and the
# caches do. Three consequences:
#
#   1. The conversation lives in st.session_state, never in a local list.
#   2. Indexing is wrapped in @st.cache_resource, keyed by URL. Without that,
#      every message you send would re-download and re-embed the whole repo.
#   3. After the agent answers we call st.rerun(), because the chat bubbles
#      rendered earlier in THIS pass have already gone to the browser and cannot
#      be repainted retroactively. Without the rerun the user sits one turn
#      behind.

import os
import re

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import agent as agent_module
import guard
import repo_index

st.set_page_config(page_title="Talk to GitHub", page_icon="💬", layout="centered")

# ============================================================================
# Styling
# ============================================================================
# Streamlit's default text input and button are sized for forms, and the landing
# page here is one field that deserves to be the focus of the screen. This CSS
# enlarges just those two.
#
# Targeted by WIDGET KEY, which is the documented way to do this. Giving a widget
# `key="foo"` makes Streamlit put `st-key-foo` on its container as a real CSS
# class, so `.st-key-foo input` hits exactly that one widget and nothing else.
#
# The obvious-looking alternative does NOT work: wrapping widgets in
# `st.markdown("<div class='hero'>")` does not nest them, because each element
# renders into its own container, the div ends up a sibling, and a descendant
# selector never matches. Keys avoid that entirely.
#
# Nothing here is load-bearing: if a future version changed the class scheme the
# app would simply look default.

BIG_INPUT_CSS = """
<style>
/* --- the URL field --- */
.st-key-repo_url_input input {
    font-size: 1.15rem !important;
    height: 3.4rem !important;
    padding: 0.5rem 1.1rem !important;
    border-radius: 12px !important;
    border: 2px solid #d7dce5 !important;
    text-align: center;
}
.st-key-repo_url_input input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12) !important;
}
.st-key-repo_url_input input::placeholder {
    color: #9aa3b2 !important;
    font-size: 1.05rem;
}

/* --- the Start button --- */
.st-key-start_button button {
    height: 3.2rem !important;
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

/* --- the example buttons: quieter, card-like, two lines of text --- */
[class*="st-key-example_"] button {
    height: auto !important;
    min-height: 3.4rem !important;
    padding: 0.6rem 0.9rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    border: 1px solid #e2e6ee !important;
    background: #fbfcfe !important;
    line-height: 1.3 !important;
    white-space: normal !important;
}
[class*="st-key-example_"] button:hover {
    border-color: #2563eb !important;
    background: #f4f7ff !important;
}
</style>
"""

# Example repositories offered on the landing page.
#
# These are the well-known ones people actually want to poke at. They are only
# usable because of the file prioritisation in repo_index.py, before that, the
# 1,200-file cap filled up in zip order and produced nonsense:
#
#   repo                     before prioritisation      after
#   facebook/react           rust:3332, ts:104          js:4000
#   tiangolo/fastapi         markdown:3756, js:25       python:3968
#   langchain-ai/langchain   python:3881 (arbitrary)    python:4000, all of libs/
#
# React's repo carries a large Rust compiler under compiler/ (4,210 files against
# 2,154 under packages/), so the index used to be Rust and asking about hooks
# retrieved a compiler. FastAPI's is dominated by its translated docs tree.
#
# All three are large enough that the index is TRUNCATED, the sidebar says so,
# and the agent is told to admit it rather than describe the whole project. Each
# costs roughly $0.024 to index, about sixteen times a small repo, which matters
# against a $5 cap. They index in 15-20 seconds.
EXAMPLE_REPOS = [
    ("facebook/react", "the UI library · 4,000 chunks of JS"),
    ("langchain-ai/langchain", "LLM framework · what this app is built with"),
    ("tiangolo/fastapi", "the Python web framework"),
]


# ============================================================================
# PART 1: Keys and the usage gate
# ============================================================================
# All of the reasoning for this lives in guard.py. The short version:
#
#   THE IN-APP GATE IS COST SMOOTHING. THE SPEND CAP ON THE OPENAI PROJECT IS
#   THE ACTUAL PROTECTION.
#
# Streamlit documents st.context.ip_address as spoofable and explicitly unsuitable
# for security, so the free-turn counter is here to stop one ordinary visitor
# draining the demo in a sitting, nothing more.


def active_credentials() -> tuple[str, bool]:
    """Return (key to use, whether it is the visitor's own).

    A visitor on their own key is never counted against the shared allowance.
    """
    own = st.session_state.get("user_key", "").strip()
    if own:
        return own, True
    return guard.demo_api_key(), False


# ============================================================================
# PART 2: Caching the expensive part
# ============================================================================


@st.cache_resource(show_spinner=False, max_entries=4)
def load_repo(url: str, _key: str, _token: str):
    """Download, split and embed one repository.

    Cached by URL, so asking a second question does not re-index. `max_entries`
    caps how many repos one server process holds at once, the index lives in
    RAM, so this is the memory ceiling for the whole app.

    The key is passed EXPLICITLY into build_index rather than being written to
    os.environ. An earlier version set os.environ["OPENAI_API_KEY"] here, which
    in a process-wide cache meant one visitor's own key silently became the
    default for every other visitor in the container. Never put a per-visitor
    credential into process-global state.

    Leading underscores keep Streamlit from hashing the credentials while still
    busting the cache when they change.
    """
    return repo_index.build_index(url, token=_token or None, api_key=_key)


@st.cache_resource(show_spinner=False, max_entries=4)
def load_agent(url: str, _key: str, _token: str):
    """Build the AgentExecutor for a repo. Cached alongside the index."""
    index = load_repo(url, _key, _token)
    return agent_module.build_agent(index, api_key=_key)


# ============================================================================
# PART 3: Session state
# ============================================================================

st.session_state.setdefault("repo_url", "")       # the repo we are talking about
st.session_state.setdefault("history", [])        # LangChain message objects
st.session_state.setdefault("transcript", [])     # what we show on screen
st.session_state.setdefault("user_key", "")

# The gate, computed ONCE per rerun and used by both the sidebar and the input
api_key, own_key = active_credentials()
# Read from secrets/env only. There is deliberately no visitor-facing input for
# this: every repository this app can reach is public, so a visitor never needs a
# token, and asking a stranger for a GitHub credential they do not need is worse
# than the small rate-limit benefit. The operator can still set GITHUB_TOKEN to
# raise the limit from 60 to 5,000 requests an hour.
github_token = guard.github_token()
can_talk = bool(api_key) and (own_key or guard.turns_left(own_key=False) > 0)


# ============================================================================
# PART 4: Sidebar
# ============================================================================

with st.sidebar:
    st.header("Talk to GitHub")
    st.caption(
        "Paste any public GitHub repository and ask questions about it, "
        "the structure, a specific file's code, where something is handled."
    )

    st.divider()

    # ---- The usage gate ----
    if own_key:
        st.success("Using your own API key, no message limit.")
    elif api_key:
        left = guard.turns_left(own_key=False)
        st.metric("Free messages left", f"{left} / {guard.FREE_TURNS}")
        if left == 0:
            st.warning(guard.ALLOWANCE_SPENT)
        st.caption(
            "This is a shared demo key with a small monthly budget, so the "
            "allowance is per visitor and deliberately low."
        )
    else:
        st.error("This demo has no API key configured.")
        # NAMES ONLY, never values, and only shown in the already-broken state
        st.caption(guard.secrets_diagnostic())

    with st.expander("Use your own OpenAI key"):
        st.session_state.user_key = st.text_input(
            "OpenAI API key", type="password", placeholder="sk-...",
            value=st.session_state.user_key,
            help="Used only for this browser session. Never stored, written to "
                 "disk, or logged, it goes to OpenAI and nowhere else.",
        ).strip()
        st.caption(f"The app calls `{guard.CHAT_MODEL}` and "
                   f"`{guard.EMBEDDING_MODEL}`.")


    if st.session_state.repo_url:
        st.divider()
        try:
            index = load_repo(st.session_state.repo_url, api_key, github_token)
            contents = index.contents
            st.subheader(contents.full_name)
            files_col, chunks_col = st.columns(2)
            files_col.metric("Files indexed", len(contents.files))
            chunks_col.metric("Chunks", len(index.documents))

            if contents.skipped:
                st.caption("Not indexed: " + ", ".join(
                    f"{count} {reason}" for reason, count in contents.skipped.items()
                ))
            if contents.subpath:
                st.info(f"Indexing only `{contents.subpath}` from "
                        f"{contents.owner}/{contents.repo}.")
            if contents.truncated:
                st.warning("This was large enough that indexing stopped early, "
                           "so the answers cover only part of it. Paste a folder "
                           "link to index one area completely.")
        except Exception:
            pass

        if st.button("Use a different repo", use_container_width=True):
            for key in ("repo_url", "history", "transcript"):
                st.session_state[key] = "" if key == "repo_url" else []
            st.rerun()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.transcript = []
            st.rerun()


# ============================================================================
# PART 5: The landing screen, with the text field in the middle of the page
# ============================================================================

if not st.session_state.repo_url:
    st.markdown(BIG_INPUT_CSS, unsafe_allow_html=True)

    # ---- The note about what this works on, at the top ----
    st.markdown(
        "<div style='text-align:center; margin-top:0.6rem; margin-bottom:1.8rem;'>"
        "<span style='display:inline-block; padding:0.35rem 0.9rem; "
        "border-radius:999px; background:#eef4ff; color:#1d4ed8; "
        "font-size:0.86rem; font-weight:500; border:1px solid #dbe6ff;'>"
        "Works with any <strong>public</strong> or <strong>open-source</strong> "
        "repository. No sign-in, nothing installed."
        "</span></div>",
        unsafe_allow_html=True,
    )

    # ---- Hero ----
    st.markdown(
        "<h1 style='text-align:center; font-size:3rem; margin:0 0 0.35rem 0; "
        "letter-spacing:-0.02em;'>💬 Talk to GitHub</h1>"
        "<p style='text-align:center; color:#6b7280; font-size:1.15rem; "
        "margin:0 0 2rem 0;'>Paste a repository link and ask it anything: "
        "the structure, a file's code, or how something works.</p>",
        unsafe_allow_html=True,
    )

    # ---- The field and button, wrapped so the CSS above scopes to them ----
    left, middle, right = st.columns([1, 6, 1])
    with middle:
        url = st.text_input(
            "GitHub repository URL",
            placeholder="github.com/owner/repo",
            label_visibility="collapsed",
            key="repo_url_input",          # -> .st-key-repo_url_input in the DOM
        )
        start = st.button("Analyse this repo  →", type="primary",
                          use_container_width=True, key="start_button")

    # ---- Examples, three across ----
    st.markdown(
        "<p style='text-align:center; color:#9aa3b2; font-size:0.85rem; "
        "margin:2rem 0 0.6rem 0; text-transform:uppercase; "
        "letter-spacing:0.06em;'>or try one of these</p>",
        unsafe_allow_html=True,
    )
    # Honest hint about large projects. The examples above are all truncated, so
    # this is the route to complete coverage rather than a nicety.
    st.markdown(
        "<p style='text-align:center; color:#9aa3b2; font-size:0.82rem; "
        "margin:1.4rem 0 0 0; line-height:1.5;'>"
        "Big project? Paste a <strong>folder</strong> link to index just that "
        "part, completely:<br>"
        "<code style='font-size:0.78rem;'>"
        "github.com/facebook/react/tree/main/packages/react</code></p>",
        unsafe_allow_html=True,
    )

    example_columns = st.columns(len(EXAMPLE_REPOS))
    for column, (name, blurb) in zip(example_columns, EXAMPLE_REPOS):
        with column:
            # The key becomes a CSS class, so it must contain no slashes or dots
            safe_key = "example_" + re.sub(r"[^a-zA-Z0-9]+", "_", name)
            if st.button(f"{name.split('/')[-1]}\n{blurb}",
                         key=safe_key, use_container_width=True,
                         help=f"github.com/{name}"):
                url, start = f"https://github.com/{name}", True

    if start:
        if not api_key:
            st.error("Add an OpenAI API key in the sidebar first.")
        elif not url.strip():
            st.error("Paste a GitHub repository link.")
        else:
            # Validate the URL BEFORE the slow download, so a typo fails fast
            try:
                owner, repo, ref, subpath = repo_index.parse_repo_url(url)
                target = f"{owner}/{repo}" + (f"/{subpath}" if subpath else "")
            except ValueError as error:
                st.error(str(error))
            else:
                status = st.status(f"Indexing {target}...", expanded=True)
                try:
                    with status:
                        st.write("Downloading the repository...")
                        index = load_repo(url.strip(), api_key, github_token)
                        st.write(f"Indexed {len(index.contents.files)} files "
                                 f"into {len(index.documents)} chunks.")
                        load_agent(url.strip(), api_key, github_token)
                    status.update(label=f"Ready: {index.contents.full_name}",
                                  state="complete", expanded=False)
                    st.session_state.repo_url = url.strip()
                    st.rerun()
                except ValueError as error:
                    # Our own, human-readable failures: 404, rate limit, too big.
                    # These are safe to show verbatim because we wrote them.
                    status.update(label="Could not index that repo", state="error")
                    st.error(str(error))
                except Exception as error:
                    # Anything from the provider goes through the classifier, so
                    # the page never shows a raw provider exception.
                    status.update(label="Could not index that repo", state="error")
                    message, should_close = guard.explain_failure(error, own_key)
                    if should_close:
                        guard.close_gate()
                    st.error(message)
                    print(f"[index error] {type(error).__name__}: {error}")

    st.stop()      # nothing below this belongs on the landing screen


# ============================================================================
# PART 6: The conversation
# ============================================================================

index = load_repo(st.session_state.repo_url, api_key, github_token)
executor = load_agent(st.session_state.repo_url, api_key, github_token)

st.title(f"💬 {index.contents.full_name}")
st.caption(f"{len(index.contents.files)} files indexed · ask about structure, "
           f"specific files, or how something works")

if not st.session_state.transcript:
    st.info(
        "Try: **What's the folder structure?** · **Show me the main entry "
        "point** · **How does the test suite work?** · **What does "
        "`README.md` say this project is for?**"
    )

for entry in st.session_state.transcript:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# The input is DISABLED rather than allowed to throw. A visitor who has run out
# should see a plain sentence, not a traceback from a client constructor.
if not api_key:
    st.warning(guard.NO_KEY)
    st.caption(guard.secrets_diagnostic())
elif not can_talk:
    st.warning(guard.ALLOWANCE_SPENT)

question = st.chat_input(
    f"Ask about {index.contents.full_name}..." if can_talk
    else "Add your own API key in the sidebar to continue",
    disabled=not can_talk,
)

if question:
    st.session_state.transcript.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reading the code..."):
            try:
                answer = agent_module.ask(executor, question, st.session_state.history)

                # Keep the LangChain history in step with the visible transcript
                st.session_state.history.append(HumanMessage(content=question))
                st.session_state.history.append(AIMessage(content=answer))

                # Cap the history so a long session cannot grow the prompt (and
                # the per-turn cost) without bound. Twelve messages is six turns.
                st.session_state.history = st.session_state.history[-12:]

                # COUNT THE TURN ONLY NOW, after success, and only if the shared
                # key paid for it. A failed turn costs the visitor nothing, so
                # there is never anything to refund.
                if not own_key:
                    guard.spend_one_turn()

            except Exception as error:
                # guard.explain_failure checks the TERMINAL case (budget gone)
                # before the TRANSIENT one (going too fast), because a quota
                # error is also delivered as a 429. Get that order wrong and
                # every spent budget reads as "try again shortly".
                #
                # A throttle deliberately does NOT spend a turn or close the
                # gate: with a 20 RPM project limit, a busy minute must not
                # destroy a visitor's whole allowance.
                answer, should_close = guard.explain_failure(error, own_key)
                if should_close:
                    guard.close_gate()
                # The real exception goes to the SERVER LOG only. Provider errors
                # can carry request details and internal identifiers.
                print(f"[agent error] {type(error).__name__}: {error}")

    st.session_state.transcript.append({"role": "assistant", "content": answer})

    # See note 3 in the header: the bubbles above have already been sent, so this
    # rerun is what actually paints the finished turn.
    st.rerun()
