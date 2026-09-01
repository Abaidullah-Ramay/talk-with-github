# Guard: keys, usage limits and failure handling for a public demo
#
# READ THIS FIRST, because it frames everything below.
#
#   THE IN-APP GATE IS COST SMOOTHING. THE SECURITY BOUNDARY IS THE OPENAI
#   PROJECT'S SPEND CAP.
#
# Streamlit's own documentation says of `st.context.ip_address`:
#
#   "This should not be used for security measures because it can easily be
#    spoofed."
#
# That is correct. Everything in this file stops one ordinary visitor from
# draining the demo allowance in a sitting. None of it stops a determined person,
# and it is not trying to. What stops you losing money is the hard spend cap set
# on the OpenAI project, which OpenAI enforces whatever this code does.
#
# Both layers are built. They are not confused with each other.
#
# What is configured on the provider side (enforced by OpenAI, not by this code):
#   - hard monthly spend cap of $5, ENFORCED, with alerts at 50% and 80%
#   - model allow-list of exactly two models, matching CHAT_MODEL and
#     EMBEDDING_MODEL below
#   - requests-per-minute overrides: 10 on the default row, 20 for chat,
#     20 for embeddings
#
# A consequence of that RPM ceiling worth knowing: one agent turn costs at least
# TWO requests (one for the model to pick a tool, one to answer). At 20 RPM
# shared across everyone, that is roughly ten conversation turns per minute for
# the whole app. So HTTP 429 from throttling is a NORMAL operating condition
# here, not a rare fault, which is exactly why it is classified separately from
# a spent budget below.

import os

import streamlit as st

# ============================================================================
# PART 1: Models, pinned to the allow-list
# ============================================================================
# Pinned to the exact DATED SNAPSHOT, not a moving alias, and the allow-list on
# the OpenAI project names the same snapshot.
#
# Why pin both sides. If you allow-list an alias like `gpt-5.4-mini` and the
# provider later repoints it at a newer snapshot, an app that asked for the alias
# silently changes behaviour on a date nobody chose. Pinning the snapshot in code
# AND in the allow-list keeps them in step, and a mismatch fails loudly at the
# first request instead of drifting.
#
# These are overridable by secret or environment variable so the model can be
# changed without a code edit, but the default is the pinned snapshot.

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini-2026-03-17")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Three messages. Enough to demonstrate the app, not enough to matter
# financially. Deliberately low.
FREE_TURNS = 3

ALLOWANCE_SPENT = (
    "The free allowance for this demo is used up. You can add your own OpenAI "
    "API key in the sidebar to keep going, it stays in your browser session."
)
TOO_BUSY = (
    "The demo is being rate limited right now (it runs on a small shared quota). "
    "Wait a few seconds and send that again, this did not use up one of your "
    "free messages."
)
NO_KEY = (
    "This demo has no OpenAI API key configured, so it cannot answer. "
    "Add your own key in the sidebar to use it."
)


# ============================================================================
# PART 2: Reading the key
# ============================================================================


def demo_api_key() -> str:
    """The shared key, read from st.secrets EXPLICITLY, then the environment.

    THE LAZY-SECRETS TRAP. Streamlit copies secrets into os.environ, but only
    once something has touched `st.secrets`. An app that reads os.environ
    directly and never touches st.secrets finds the environment empty on
    Community Cloud, and constructing the OpenAI client then raises before the
    page can render a single element, which looks exactly like a broken app
    rather than a missing secret.

    Touching st.secrets first is the fix. The try/except is because there is no
    secrets.toml at all when running locally, which is normal.
    """
    from_secrets = ""
    try:
        from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        from_secrets = ""
    return str(from_secrets or os.environ.get("OPENAI_API_KEY", "")).strip()


def github_token() -> str:
    """Optional. Only raises the GitHub rate limit; never needed for public repos."""
    from_secrets = ""
    try:
        from_secrets = st.secrets.get("GITHUB_TOKEN", "")
    except Exception:
        from_secrets = ""
    return str(from_secrets or os.environ.get("GITHUB_TOKEN", "")).strip()


def secrets_diagnostic() -> str:
    """Why no key was found, in terms an operator can act on. NAMES, NEVER VALUES.

    This exists because of a second trap: the Secrets box on Community Cloud
    takes TOML, so an unquoted line loads NOTHING AT ALL rather than partially:

        OPENAI_API_KEY=sk-proj-...      wrong. Invalid TOML, zero secrets loaded
        OPENAI_API_KEY="sk-proj-..."    correct

    From the outside, "invalid file" and "no secrets configured" look identical.
    This line distinguishes them. It is shown only when the app is already
    broken, and it never prints a value.
    """
    try:
        names = sorted(st.secrets.keys())
    except Exception as error:
        return (f"st.secrets could not be read ({type(error).__name__}). "
                f"If you are on Streamlit Cloud, check the Secrets box parses as "
                f"TOML. Values must be quoted.")
    if not names:
        return "No secrets are configured for this app."
    return "Secrets this app can see: " + ", ".join(names) + "."


# ============================================================================
# PART 3: The usage counter
# ============================================================================


@st.cache_resource(ttl="12h")
def free_turns_used() -> dict:
    """Turns spent per visitor, shared across every session in this app process.

    DELIBERATELY NOT IN SESSION STATE. A per-session counter resets on page
    reload and in a new tab, which makes it no limit at all. cache_resource
    returns the SAME object to every session in the container, so it survives
    both.

    Its honest limits, none of which are secrets:
      - it resets when the app sleeps or redeploys (Community Cloud sleeps idle
        apps), which zeroes this dictionary
      - everyone behind one NAT shares an entry, so an office shares three
        messages between hundreds of people, a usability cost, not a security one
      - IP addresses are spoofable, and IPv6 hands one user an enormous range

    ttl="12h" bounds the growth: one entry per distinct IP until it expires.
    """
    return {}


def visitor_id() -> str:
    """A best-effort per-visitor key. Not an identity, and not trustworthy.

    Two fallbacks, and the second one was a real bug worth keeping a note on.

    The obvious one: on localhost the IP is None, so without a fallback local
    development keys everything under None.

    The one that actually bit: `getattr(..., None) or "local"` is NOT enough,
    because it only guards against None and falsiness. Under Streamlit's own
    AppTest harness `st.context.ip_address` returns a MagicMock, truthy, so the
    fallback never fired, and a DIFFERENT instance every script run, so every
    turn landed under a new dictionary key and the counter never got past 1.

    The gate still rendered a number the whole time. It looked like it was
    working while enforcing nothing at all, which for a cost control is the
    worst way to fail. So insist on an actual string: anything else is not a
    usable key and must collapse to one shared bucket rather than silently
    becoming unlimited.
    """
    raw = getattr(st.context, "ip_address", None)
    if not isinstance(raw, str) or not raw.strip():
        return "local"
    return raw.strip()


def turns_left(own_key: bool) -> int:
    """How many shared-key turns this visitor has left. Unlimited on their own key."""
    if own_key:
        return FREE_TURNS
    return max(0, FREE_TURNS - free_turns_used().get(visitor_id(), 0))


def spend_one_turn() -> None:
    """Count a turn. Called ONLY after a successful call on the shared key.

    Counting after success rather than before means a failed turn costs the
    visitor nothing, so there is never anything to refund.
    """
    used = free_turns_used()
    key = visitor_id()
    used[key] = used.get(key, 0) + 1


def close_gate() -> None:
    """Force the allowance to zero.

    Used when the shared budget is GONE rather than merely throttled. Further
    attempts would just fail again, so there is no point letting the visitor
    spend their remaining turns discovering that.
    """
    free_turns_used()[visitor_id()] = FREE_TURNS


# ============================================================================
# PART 4: Classifying failures
# ============================================================================
# THE TRAP THIS SOLVES: OpenAI answers two completely different situations with
# HTTP 429.
#
#   "your budget is gone"      -> terminal. Nothing will work until billing changes.
#   "you are going too fast"   -> transient. Works again in seconds.
#
# Treating them the same means a visitor who arrives during a busy minute is told
# the demo is over, and if you also close their gate, a fifteen-second queue has
# destroyed their entire free allowance. With a 20 RPM project limit that is not
# hypothetical, it is the common case.
#
# ORDER MATTERS. A quota error is ALSO delivered as a 429 and often contains the
# string "429", so the terminal check must run FIRST or every spent budget is
# misread as a speed problem and the visitor is invited to retry into a wall.

BUDGET_GONE = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing_hard_limit",
    "billing hard limit",
    "check your plan and billing",
)

TOO_FAST = (
    "rate limit",
    "rate_limit",
    "ratelimiterror",
    "too many requests",
    "429",
)

BAD_KEY = (
    "invalid_api_key",
    "incorrect api key",
    "invalid authentication",
    "authenticationerror",
    "401",
)

MODEL_BLOCKED = (
    "model_not_found",
    "does not exist or you do not have access",
    "do not have access to model",
)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__} {error}".lower()


def out_of_credit(error: BaseException) -> bool:
    """Terminal: the budget is spent. MUST be checked before rate_limited()."""
    return any(marker in _error_text(error) for marker in BUDGET_GONE)


def rate_limited(error: BaseException) -> bool:
    """Transient: too fast. MUST be checked AFTER out_of_credit()."""
    return any(marker in _error_text(error) for marker in TOO_FAST)


def bad_key(error: BaseException) -> bool:
    """The key itself was rejected."""
    return any(marker in _error_text(error) for marker in BAD_KEY)


def model_blocked(error: BaseException) -> bool:
    """The model is not on the project's allow-list, or the pin has drifted.

    Worth its own message: this one is a configuration mistake by the operator,
    not anything the visitor did, and the fix is in the OpenAI dashboard.
    """
    return any(marker in _error_text(error) for marker in MODEL_BLOCKED)


def explain_failure(error: BaseException, own_key: bool) -> tuple[str, bool]:
    """Turn an exception into (message for the visitor, should_close_gate).

    NEVER returns the raw exception text. Provider errors can carry request
    details, internal identifiers and occasionally fragments of the request
    itself. The real exception is logged server-side by the caller.

    The order of these branches is the whole point, see the comment above.
    """
    # 1. TERMINAL first, because a quota error also contains "429"
    if out_of_credit(error):
        if own_key:
            return ("That key has no credit left. Check its billing on the "
                    "OpenAI dashboard.", False)
        return (ALLOWANCE_SPENT, True)        # close the gate: it will not recover

    # 2. Transient. Deliberately does NOT close the gate or spend a turn.
    if rate_limited(error):
        if own_key:
            return ("That key is being rate limited. Wait a moment and try "
                    "again.", False)
        return (TOO_BUSY, False)

    # 3. The key was rejected
    if bad_key(error):
        if own_key:
            return ("That API key was rejected. Check you pasted it fully.", False)
        return ("The demo's API key is not working. The operator needs to "
                "check it.", False)

    # 4. Operator misconfiguration, worth naming precisely
    if model_blocked(error):
        return (f"This app is configured to use `{CHAT_MODEL}`, which the API "
                f"key is not allowed to call. If you are the operator: the "
                f"model allow-list on the OpenAI project must include exactly "
                f"this model id.", False)

    # 5. Anything else stays vague on the page and detailed in the log
    return ("Something went wrong on my end. Try rephrasing that, or ask about "
            "a specific file.", False)
