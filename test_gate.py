# Tests for the usage gate, driven through the real app with Streamlit's own
# AppTest harness. No API key, no network, no GitHub.
#
# WHY THIS FILE EXISTS, SEPARATELY FROM test_guard.py
#
# test_guard.py tests the classifier functions in isolation, which is the right
# way to test pure logic. It cannot catch a WIRING bug, and both gate failures
# found so far were wiring bugs:
#
#   1. visitor_id() returned a fresh MagicMock every run, so the counter got a
#      new dictionary key each turn and never passed 1. The sidebar rendered a
#      number the whole time while enforcing nothing.
#
#   2. The counter gated CHATTING and left INDEXING open. Chatting is about
#      $0.001 a turn, indexing a large repo about $0.024, so the gate was
#      guarding the cheap operation. With the chat gate fully closed,
#      build_index ran on four attempts out of four.
#
# Neither was visible by reading guard.py. Both are obvious the moment the app
# is actually driven. So these tests drive it: the expensive calls are stubbed
# and COUNTED, and the assertions are about how many times real money would
# have been spent.
#
# Usage:
#   pytest test_gate.py -v
#   python test_gate.py

import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "sk-not-a-real-key-nothing-is-called")

import agent
import guard
import repo_index

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")

# How many times the app tried to spend real money
SPEND = {"index": 0, "chat": 0}


class FakeContents:
    owner, repo, ref, subpath = "owner", "repo", "", ""
    full_name = "owner/repo"
    files = {"main.py": "print('hi')"}
    skipped: dict = {}
    truncated = False


class FakeIndex:
    documents = ["one chunk"]

    def __init__(self, full_name="owner/repo"):
        self.contents = FakeContents()
        self.contents.full_name = full_name


def _install_stubs():
    """Replace every call that costs money or touches the network."""
    def fake_build_index(url, *args, **kwargs):
        SPEND["index"] += 1
        return FakeIndex(url.rsplit("github.com/", 1)[-1])

    def fake_ask(executor, question, chat_history):
        SPEND["chat"] += 1
        return f"stub answer to {question!r}"

    def fake_parse(url):
        parts = url.rstrip("/").split("/")
        return parts[-2], parts[-1], "", ""

    repo_index.build_index = fake_build_index
    repo_index.parse_repo_url = fake_parse
    agent.build_agent = lambda *a, **k: object()
    agent.ask = fake_ask


def fresh_app(own_key: str = ""):
    """A clean app with clean allowances. Returns the AppTest."""
    from streamlit.testing.v1 import AppTest

    _install_stubs()
    SPEND["index"] = SPEND["chat"] = 0
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    guard.free_turns_used().clear()
    guard.indexed_repos().clear()
    if own_key:
        app.session_state["user_key"] = own_key
    app.run()
    return app


def load_repo(app, name: str) -> bool:
    """Drive the landing screen. True if the app accepted the repo."""
    app.session_state["repo_url"] = ""
    app.run()
    app.text_input(key="repo_url_input").set_value(f"https://github.com/{name}")
    app.button(key="start_button").click().run()
    return app.session_state["repo_url"].endswith(name)


def send(app, text: str) -> bool:
    """Send one chat message. False if the input was disabled."""
    if not len(app.chat_input) or app.chat_input[0].disabled:
        return False
    app.chat_input[0].set_value(text).run()
    return True


# ============================================================================
# 1. THE CASE THAT MATTERS: the expensive path must be metered
# ============================================================================


def test_indexing_is_capped_per_visitor():
    """REGRESSION. Indexing used to be entirely ungated.

    Embedding a large repo is roughly twenty chat turns of cost in one click,
    so an uncapped index button drains the budget far faster than chatting can.
    """
    app = fresh_app()
    accepted = [load_repo(app, name) for name in
                ("facebook/react", "django/django", "pallets/click", "psf/requests")]

    assert accepted[:guard.FREE_INDEXES] == [True] * guard.FREE_INDEXES, \
        "a visitor must be able to use their allowance"
    assert not any(accepted[guard.FREE_INDEXES:]), \
        "indexing past the allowance must be refused"
    assert SPEND["index"] == guard.FREE_INDEXES, \
        f"build_index ran {SPEND['index']} times for an allowance of {guard.FREE_INDEXES}"


def test_spent_message_allowance_also_blocks_indexing():
    """A repo you cannot then ask about is embedding budget spent for nothing."""
    app = fresh_app()
    guard.close_gate()
    app.run()

    assert not load_repo(app, "tiangolo/fastapi")
    assert SPEND["index"] == 0, "indexed despite having no messages left"
    assert any("allowance" in e.value.lower() for e in app.error), \
        "the visitor must be told why"


def test_reopening_the_same_repo_is_free():
    """'Use a different repo' returns to the landing screen.

    Coming back to a repo you already loaded must not cost a second slot, and
    while the index is cached it costs nothing to serve.
    """
    app = fresh_app()
    assert load_repo(app, "facebook/react")
    builds_after_first = SPEND["index"]

    for _ in range(4):
        assert load_repo(app, "facebook/react"), "reopening was refused"
    assert SPEND["index"] == builds_after_first, "re-indexed a cached repo"
    assert guard.indexes_left(own_key=False) == guard.FREE_INDEXES - 1, \
        "reopening consumed extra allowance"


# ============================================================================
# 2. The chat counter must actually count
# ============================================================================


def test_chat_turns_are_counted_and_then_locked():
    """REGRESSION for the MagicMock bug: the counter must reach the limit."""
    app = fresh_app()
    assert load_repo(app, "octocat/hello")

    for turn in range(guard.FREE_TURNS):
        assert send(app, f"question {turn}"), f"turn {turn + 1} was refused early"

    assert SPEND["chat"] == guard.FREE_TURNS
    assert not send(app, "one too many"), "the input was still enabled at zero"
    assert SPEND["chat"] == guard.FREE_TURNS, "a turn ran past the limit"
    assert guard.turns_left(own_key=False) == 0


def test_the_sidebar_number_matches_what_is_enforced():
    """The first bug rendered a healthy number while enforcing nothing.

    So check the displayed value against the enforced one, not just the enforced
    one. A gate that lies on screen is worse than no gate.
    """
    app = fresh_app()
    assert load_repo(app, "octocat/hello")

    for expected in range(guard.FREE_TURNS - 1, -1, -1):
        send(app, "a question")
        shown = {m.label: m.value for m in app.get("metric")}
        assert shown.get("Messages left") == str(expected), \
            f"sidebar shows {shown.get('Messages left')!r}, enforced value is {expected}"


# ============================================================================
# 3. A visitor's own key lifts both allowances
# ============================================================================


def test_own_key_is_not_metered():
    app = fresh_app(own_key="sk-the-visitors-own-key")

    for name in ("a/one", "b/two", "c/three", "d/four"):
        assert load_repo(app, name), f"{name} refused on the visitor's own key"
    assert SPEND["index"] == 4

    for turn in range(guard.FREE_TURNS + 3):
        assert send(app, f"question {turn}"), "own key hit a message limit"

    assert guard.free_turns_used() == {}, "own-key turns were charged to the demo"
    assert guard.indexed_repos() == {}, "own-key repos were charged to the demo"


def test_app_renders_with_no_exceptions():
    app = fresh_app()
    load_repo(app, "octocat/hello")
    send(app, "what is this repo")
    assert len(app.exception) == 0, [e.value for e in app.exception]


# ============================================================================
# Runner (works with or without pytest)
# ============================================================================


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and isinstance(v, types.FunctionType)]
    print("=" * 72)
    print(f"gate tests ({len(tests)}), no key, no network, no GitHub")
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
