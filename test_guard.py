# Tests for guard.py. No API key, no network, no services.
#
# The security brief is explicit about why these exist:
#
#   "Test the classifier against real message strings rather than trusting the
#    patterns by eye. The case that matters is a quota error that also contains
#    '429': it must come out terminal."
#
# That case is the whole reason this file exists. Reading the patterns and
# nodding is not a test, because the failure is an ORDERING bug that both
# patterns match.
#
# Usage:
#   pytest test_guard.py -v
#   python test_guard.py

import sys
import types

# guard.py imports streamlit at module level, and importing streamlit outside a
# script run is fine, but @st.cache_resource needs a runtime to actually call.
# The classifier functions are pure, so we test those directly and stub the
# cache-backed ones.
import guard


# ============================================================================
# Real error strings, copied from what OpenAI actually returns
# ============================================================================

QUOTA_ERROR = (
    "Error code: 429 - {'error': {'message': 'You exceeded your current quota, "
    "please check your plan and billing details.', 'type': 'insufficient_quota', "
    "'param': None, 'code': 'insufficient_quota'}}"
)

RATE_LIMIT_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for "
    "gpt-5.4-mini-2026-03-17 in organization org-xxx on requests per min (RPM): "
    "Limit 20, Used 20, Requested 1.', 'type': 'requests', 'param': None, "
    "'code': 'rate_limit_exceeded'}}"
)

BAD_KEY_ERROR = (
    "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-xxx. "
    "You can find your API key at https://platform.openai.com/account/api-keys.', "
    "'type': 'invalid_request_error', 'code': 'invalid_api_key'}}"
)

MODEL_BLOCKED_ERROR = (
    "Error code: 404 - {'error': {'message': 'The model `gpt-4o` does not exist "
    "or you do not have access to it.', 'type': 'invalid_request_error', "
    "'code': 'model_not_found'}}"
)

HARD_LIMIT_ERROR = (
    "Error code: 429 - {'error': {'message': 'Billing hard limit has been "
    "reached', 'type': 'billing_hard_limit_reached'}}"
)

RANDOM_ERROR = "ConnectionError: Max retries exceeded with url: /v1/chat/completions"


class FakeError(Exception):
    """Stands in for an openai.RateLimitError etc, only the text is classified."""


# ============================================================================
# 1. THE CASE THAT MATTERS: a quota error also contains "429"
# ============================================================================


def test_quota_error_is_terminal_not_transient():
    """A spent budget must NOT be read as 'you are going too fast'.

    Both classifiers match this string, it contains 'insufficient_quota' AND
    '429'. The ordering in explain_failure is what makes it terminal. Get it
    backwards and every visitor whose budget is gone is told to retry shortly,
    inviting them to hammer a wall forever.
    """
    error = FakeError(QUOTA_ERROR)

    assert guard.out_of_credit(error), "quota error not detected as terminal"
    assert guard.rate_limited(error), (
        "this string DOES contain 429, that is the whole trap, and why order matters"
    )

    message, should_close = guard.explain_failure(error, own_key=False)
    assert message == guard.ALLOWANCE_SPENT, "did not take the terminal branch"
    assert should_close is True, "a spent budget must close the gate"


def test_hard_billing_limit_is_terminal():
    error = FakeError(HARD_LIMIT_ERROR)
    assert guard.out_of_credit(error)
    _, should_close = guard.explain_failure(error, own_key=False)
    assert should_close is True


# ============================================================================
# 2. A throttle must be transient and must cost the visitor nothing
# ============================================================================


def test_rate_limit_is_transient_and_does_not_close_the_gate():
    """With a 20 RPM project limit this is the COMMON case, not a rare fault.

    A fifteen-second queue must not destroy a visitor's whole allowance.
    """
    error = FakeError(RATE_LIMIT_ERROR)

    assert not guard.out_of_credit(error), "a throttle is not a spent budget"
    assert guard.rate_limited(error)

    message, should_close = guard.explain_failure(error, own_key=False)
    assert message == guard.TOO_BUSY
    assert should_close is False, "a throttle must NEVER close the gate"
    assert "did not use up" in message, "tell the visitor it was free"


# ============================================================================
# 3. The other branches
# ============================================================================


def test_bad_key_is_named_as_such():
    error = FakeError(BAD_KEY_ERROR)
    assert guard.bad_key(error)
    assert not guard.out_of_credit(error)

    own, _ = guard.explain_failure(error, own_key=True)
    shared, _ = guard.explain_failure(error, own_key=False)
    assert "rejected" in own.lower()
    assert own != shared, "whose key failed changes what the visitor should do"


def test_blocked_model_names_the_configuration_problem():
    """This is an OPERATOR mistake, so the message should say what to fix."""
    error = FakeError(MODEL_BLOCKED_ERROR)
    assert guard.model_blocked(error)
    message, should_close = guard.explain_failure(error, own_key=False)
    assert guard.CHAT_MODEL in message, "name the model that is actually configured"
    assert "allow-list" in message
    assert should_close is False


def test_unknown_error_stays_vague_on_the_page():
    """The page must never carry a raw provider exception."""
    error = FakeError(RANDOM_ERROR)
    message, should_close = guard.explain_failure(error, own_key=False)
    assert "Max retries" not in message
    assert "ConnectionError" not in message
    assert should_close is False


def test_no_message_ever_leaks_the_raw_exception():
    """Every branch, checked at once: no provider text reaches the visitor."""
    leaky = [
        QUOTA_ERROR, RATE_LIMIT_ERROR, BAD_KEY_ERROR,
        MODEL_BLOCKED_ERROR, HARD_LIMIT_ERROR, RANDOM_ERROR,
    ]
    for raw in leaky:
        for own in (True, False):
            message, _ = guard.explain_failure(FakeError(raw), own_key=own)
            assert "Error code:" not in message
            assert "'error'" not in message
            assert "org-" not in message
            assert "sk-" not in message, "a key fragment must never be echoed back"


# ============================================================================
# 4. Model pins must match the OpenAI project's allow-list
# ============================================================================


def test_models_are_pinned_to_dated_snapshots():
    """An alias silently changes behaviour when the provider repoints it.

    The allow-list on the project names these exact ids, so a drift here fails
    at the first request rather than quietly becoming a different model.
    """
    assert guard.CHAT_MODEL == "gpt-5.4-mini-2026-03-17"
    assert guard.EMBEDDING_MODEL == "text-embedding-3-small"

    # And the modules that actually make the calls must agree with guard.py
    import agent
    import repo_index
    assert agent.MODEL == guard.CHAT_MODEL, "agent.py drifted from the allow-list"
    assert repo_index.EMBEDDING_MODEL == guard.EMBEDDING_MODEL, \
        "repo_index.py drifted from the allow-list"


def test_free_allowance_is_low():
    """Enough to demonstrate the app, not enough to matter financially."""
    assert 1 <= guard.FREE_TURNS <= 5, f"{guard.FREE_TURNS} is not a demo allowance"


# ============================================================================
# 5. visitor_id must never silently become unlimited
# ============================================================================


def test_visitor_id_rejects_non_string_ip():
    """REGRESSION. A non-string ip_address must collapse to one bucket.

    The original was `getattr(st.context, "ip_address", None) or "local"`, which
    only guards against None. Under AppTest, ip_address is a MagicMock: truthy,
    so the fallback never fired, and a NEW instance every run, so every turn got
    a fresh dict key and the counter never passed 1.

    The gate rendered a number the whole time while enforcing nothing. That is
    the failure mode this test exists to catch.
    """
    import unittest.mock as mock

    class FakeContext:
        def __init__(self, value):
            self.ip_address = value

    real_context = guard.st.context
    try:
        for bad in (mock.MagicMock(), object(), 12345, b"1.2.3.4", "", "   ", None):
            guard.st.context = FakeContext(bad)
            got = guard.visitor_id()
            assert isinstance(got, str), f"{type(bad).__name__} gave {type(got).__name__}"
            assert got == "local", f"{bad!r} should fall back to 'local', got {got!r}"

        # Two separate MagicMocks must land in the SAME bucket, or the counter
        # is defeated exactly as it was before the fix
        guard.st.context = FakeContext(mock.MagicMock())
        first = guard.visitor_id()
        guard.st.context = FakeContext(mock.MagicMock())
        second = guard.visitor_id()
        assert first == second, "distinct mocks produced distinct keys, gate defeated"

        # A genuine IP is used as-is
        guard.st.context = FakeContext("203.0.113.7")
        assert guard.visitor_id() == "203.0.113.7"
        guard.st.context = FakeContext("  203.0.113.7  ")
        assert guard.visitor_id() == "203.0.113.7", "should be stripped"
    finally:
        guard.st.context = real_context


# ============================================================================
# 6. The diagnostic must never print a value
# ============================================================================


def test_secrets_diagnostic_reports_names_only():
    """It exists to tell an operator 'the TOML did not parse' vs 'nothing set'.

    It must never print a secret VALUE, since it renders on a public page.
    """
    text = guard.secrets_diagnostic()
    assert isinstance(text, str) and text
    assert "sk-" not in text
    # It should either say nothing is configured, name keys, or explain the read
    assert any(cue in text for cue in
               ("No secrets", "Secrets this app can see", "could not be read"))


# ============================================================================
# Runner (works with or without pytest)
# ============================================================================


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and isinstance(v, types.FunctionType)]
    print("=" * 68)
    print(f"guard tests ({len(tests)}), no key, no network")
    print("=" * 68)
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
        except Exception as error:
            failures += 1
            print(f"  [FAIL] {test.__name__}: {type(error).__name__}: {error}")
    print("=" * 68)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
