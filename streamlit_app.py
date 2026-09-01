# Talk to GitHub - the Streamlit app
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
# Local variables do not survive between runs - only `st.session_state` and the
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

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import agent as agent_module
import guard
import repo_index

st.set_page_config(page_title="Talk to GitHub", page_icon="💬", layout="centered")

EXAMPLE_REPOS = [
    "https://github.com/pypa/sampleproject",
    "https://github.com/psf/requests",
    "https://github.com/tiangolo/fastapi",
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
# draining the demo in a sitting - nothing more.


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
    caps how many repos one server process holds at once - the index lives in
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
st.session_state.setdefault("user_gh_token", "")

# The gate, computed ONCE per rerun and used by both the sidebar and the input
api_key, own_key = active_credentials()
github_token = st.session_state.get("user_gh_token", "").strip() or guard.github_token()
can_talk = bool(api_key) and (own_key or guard.turns_left(own_key=False) > 0)


# ============================================================================
# PART 4: Sidebar
# ============================================================================

with st.sidebar:
    st.header("Talk to GitHub")
    st.caption(
        "Paste any public GitHub repository and ask questions about it - "
        "the structure, a specific file's code, where something is handled."
    )

    st.divider()

    # ---- The usage gate ----
    if own_key:
        st.success("Using your own API key - no message limit.")
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
        # NAMES ONLY, never values - and only shown in the already-broken state
        st.caption(guard.secrets_diagnostic())

    with st.expander("Use your own OpenAI key"):
        st.session_state.user_key = st.text_input(
            "OpenAI API key", type="password", placeholder="sk-...",
            value=st.session_state.user_key,
            help="Used only for this browser session. Never stored, written to "
                 "disk, or logged - it goes to OpenAI and nowhere else.",
        ).strip()
        st.caption(f"The app calls `{guard.CHAT_MODEL}` and "
                   f"`{guard.EMBEDDING_MODEL}`.")

    with st.expander("GitHub token (optional)"):
        st.caption(
            "Only raises the GitHub rate limit from 60 to 5,000 requests per "
            "hour. Public repositories work without it."
        )
        st.session_state.user_gh_token = st.text_input(
            "GitHub token", type="password", value=st.session_state.user_gh_token
        )

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
            if contents.truncated:
                st.warning("This repo was large enough that indexing stopped early, "
                           "so the answers cover only part of it.")
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
# PART 5: The landing screen - the text field in the middle of the page
# ============================================================================

if not st.session_state.repo_url:
    # Vertical breathing room, so the input sits near the middle rather than
    # jammed under the title
    st.write("")
    st.write("")

    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0.2em;'>💬 Talk to GitHub</h1>"
        "<p style='text-align:center; color:#888; font-size:1.05em;'>"
        "Paste a repository link and ask it anything.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Centre the field by putting it in the middle of three columns
    left, middle, right = st.columns([1, 3, 1])
    with middle:
        url = st.text_input(
            "GitHub repository URL",
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed",
        )
        start = st.button("Start", type="primary", use_container_width=True)

        st.write("")
        st.caption("Or try one of these:")
        for example in EXAMPLE_REPOS:
            name = "/".join(example.split("/")[-2:])
            if st.button(name, key=f"eg_{name}", use_container_width=True):
                url, start = example, True

    if start:
        if not api_key:
            st.error("Add an OpenAI API key in the sidebar first.")
        elif not url.strip():
            st.error("Paste a GitHub repository link.")
        else:
            # Validate the URL BEFORE the slow download, so a typo fails fast
            try:
                owner, repo = repo_index.parse_repo_url(url)
            except ValueError as error:
                st.error(str(error))
            else:
                status = st.status(f"Indexing {owner}/{repo}...", expanded=True)
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

                # COUNT THE TURN ONLY NOW - after success, and only if the shared
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
