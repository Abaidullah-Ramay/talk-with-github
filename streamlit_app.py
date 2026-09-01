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
import repo_index

st.set_page_config(page_title="Talk to GitHub", page_icon="💬", layout="centered")

EXAMPLE_REPOS = [
    "https://github.com/pypa/sampleproject",
    "https://github.com/psf/requests",
    "https://github.com/tiangolo/fastapi",
]


# ============================================================================
# PART 1: The API key
# ============================================================================


def get_api_key() -> str:
    """Find an OpenAI key: the visitor's own, then secrets, then the environment.

    WHY st.secrets IS READ EXPLICITLY. Streamlit copies secrets into the process
    environment only once something touches the secrets object. An app that never
    does can find the environment empty on a hosted platform, and constructing
    the OpenAI client then raises before the page can render - which looks
    identical to "no secrets configured".

    Reading it here, in a try/except (there is no secrets file locally), makes
    both paths work.
    """
    if st.session_state.get("user_key"):
        return st.session_state.user_key

    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass                        # no secrets.toml - normal when running locally

    return os.getenv("OPENAI_API_KEY", "")


def get_github_token() -> str:
    """Optional. Only raises the GitHub rate limit; never needed for public repos."""
    if st.session_state.get("user_gh_token"):
        return st.session_state.user_gh_token
    try:
        if "GITHUB_TOKEN" in st.secrets:
            return st.secrets["GITHUB_TOKEN"]
    except Exception:
        pass
    return os.getenv("GITHUB_TOKEN", "")


# ============================================================================
# PART 2: Caching the expensive part
# ============================================================================


@st.cache_resource(show_spinner=False, max_entries=4)
def load_repo(url: str, _key: str, _token: str):
    """Download, split and embed one repository.

    Cached by URL, so asking a second question does not re-index. `max_entries`
    caps how many repos a single server process keeps in memory at once - the
    index lives in RAM, so this is the memory ceiling for the whole app.

    The key and token are passed with a leading underscore so Streamlit does not
    hash them (they are credentials) while still busting the cache if they
    change.
    """
    os.environ["OPENAI_API_KEY"] = _key
    return repo_index.build_index(url, token=_token or None)


@st.cache_resource(show_spinner=False, max_entries=4)
def load_agent(url: str, _key: str, _token: str):
    """Build the AgentExecutor for a repo. Cached alongside the index."""
    index = load_repo(url, _key, _token)
    os.environ["OPENAI_API_KEY"] = _key
    return agent_module.build_agent(index)


# ============================================================================
# PART 3: Session state
# ============================================================================

st.session_state.setdefault("repo_url", "")       # the repo we are talking about
st.session_state.setdefault("history", [])        # LangChain message objects
st.session_state.setdefault("transcript", [])     # what we show on screen
st.session_state.setdefault("user_key", "")
st.session_state.setdefault("user_gh_token", "")

api_key = get_api_key()


# ============================================================================
# PART 4: Sidebar
# ============================================================================

with st.sidebar:
    st.header("Talk to GitHub")
    st.caption(
        "Paste any public GitHub repository and ask questions about it - "
        "the structure, a specific file's code, where something is handled."
    )

    if not api_key:
        st.warning("An OpenAI API key is needed.")
        st.session_state.user_key = st.text_input(
            "OpenAI API key", type="password",
            help="Stored only in this browser session, never written to disk.",
        )
        api_key = get_api_key()

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
            index = load_repo(st.session_state.repo_url, api_key, get_github_token())
            contents = index.contents
            st.subheader(contents.full_name)
            left, right = st.columns(2)
            left.metric("Files indexed", len(contents.files))
            right.metric("Chunks", len(index.documents))

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
                        index = load_repo(url.strip(), api_key, get_github_token())
                        st.write(f"Indexed {len(index.contents.files)} files "
                                 f"into {len(index.documents)} chunks.")
                        load_agent(url.strip(), api_key, get_github_token())
                    status.update(label=f"Ready: {index.contents.full_name}",
                                  state="complete", expanded=False)
                    st.session_state.repo_url = url.strip()
                    st.rerun()
                except ValueError as error:
                    # Our own, human-readable failures: 404, rate limit, too big
                    status.update(label="Could not index that repo", state="error")
                    st.error(str(error))
                except Exception as error:
                    status.update(label="Could not index that repo", state="error")
                    message = str(error).lower()
                    if any(cue in message for cue in
                           ("api key", "authentication", "invalid_api_key", "401")):
                        st.error("That OpenAI API key was rejected. Check it in the sidebar.")
                    elif any(cue in message for cue in ("quota", "rate limit", "429")):
                        st.error("The OpenAI account has hit its usage limit. "
                                 "Add your own key in the sidebar to continue.")
                    else:
                        st.error(f"Something went wrong: {type(error).__name__}")
                    print(f"[error] {type(error).__name__}: {error}")

    st.stop()      # nothing below this belongs on the landing screen


# ============================================================================
# PART 6: The conversation
# ============================================================================

index = load_repo(st.session_state.repo_url, api_key, get_github_token())
executor = load_agent(st.session_state.repo_url, api_key, get_github_token())

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

question = st.chat_input(f"Ask about {index.contents.full_name}...")

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

            except Exception as error:
                message = str(error).lower()
                if any(cue in message for cue in ("quota", "rate limit", "429",
                                                  "insufficient_quota")):
                    answer = ("I have hit the usage limit for this key. Add your own "
                              "OpenAI key in the sidebar to keep going.")
                elif any(cue in message for cue in ("api key", "401", "invalid_api_key")):
                    answer = "That OpenAI API key was rejected. Check it in the sidebar."
                else:
                    answer = ("Something went wrong answering that. Try rephrasing, "
                              "or ask about a specific file.")
                print(f"[error] {type(error).__name__}: {error}")

    st.session_state.transcript.append({"role": "assistant", "content": answer})

    # See note 3 in the header: the bubbles above have already been sent, so this
    # rerun is what actually paints the finished turn.
    st.rerun()
