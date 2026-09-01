# Security brief: publishing a Streamlit app that holds your OpenAI key

Written from a deployed public demo that runs on a shared OpenAI key. Hand this to an AI
as a whole before it builds anything that exposes a paid API key to the public.

**Read this first.** Streamlit's own documentation for `st.context.ip_address` says:

> "This should not be used for security measures because it can easily be spoofed."

That is correct, and it frames everything below. **The in-app gate is cost smoothing. The
security boundary is entirely on the OpenAI side.** Build both, but do not confuse them.

---

## 1. The threat model

You are putting a key you pay for behind a public URL with no login. Rank the risks
honestly:

| Risk | Likelihood | Worst case |
|---|---|---|
| Casual overuse by genuine visitors | high | your monthly budget gone in a day |
| Someone scripting the public URL in a loop | medium | same, faster |
| The key itself leaking (git, logs, screenshots) | medium | someone else spends your money until you notice |
| Someone using your app as a free general-purpose LLM proxy | medium | budget gone, plus your account attached to their prompts |
| Your dataset publishing someone's personal data | medium if scraped data | a real privacy problem, not a cost one |

Only one of those is solved by rate limiting in the app. Build all five layers.

---

## 2. Layer 1: the OpenAI platform. This is the real boundary.

Everything here is enforced by OpenAI, not by your code, so it holds even if your app is
compromised or someone bypasses it entirely.

**Create a dedicated project.** Do not use your default project or a personal key. A
project scopes the budget, the model list and the rate limits together, so nothing you set
here can affect your other work.

**Set a hard spend cap on that project**, not just a notification threshold. Add alerts
below it. I used **50% and 80% alerts plus the hard limit**, which gives two warnings
before anything stops. The hard cap is what actually protects you; the alerts just mean
you find out on the day rather than at the end of the month.

**Use the model allow-list.** Restrict the project to exactly the models your app calls,
and nothing else. This is the control that stops your app being turned into a free proxy
for an expensive model. In my case that is one chat model and one embedding model.

**Pin the exact model snapshot in your code, and allow-list that same snapshot.** I use
the dated snapshot rather than the moving alias. If you allow-list an alias and the
provider later repoints it at a new snapshot, an unpinned app breaks in production on a
date nobody chose. Pinning both sides keeps them in step.

**Set a requests-per-minute limit on the project.** I set **20 RPM**, which is generous
for one human having a conversation and useless for a script. Tokens-per-minute can be
left at the default: RPM is the one that stops a loop, and a low TPM mostly just breaks
legitimate long turns.

**Use a project-scoped API key**, not an account-wide one, so revoking it costs you
nothing else.

Order of importance if you only do some of this: **hard spend cap, then model allow-list,
then RPM.** The cap is the only one with a guaranteed worst case.

### Worked example: exactly what I set

This is the live configuration on the demo project, as a concrete target to copy.

| Setting | Where | Value |
|---|---|---|
| Monthly spend limit | Project, Limits | **$5.00** |
| Enforce a hard limit | same dialog, toggle | **ON** |
| Alerts | Project, Limits | **50% and 80%** |
| Allowed models | Project, Limits | **`gpt-5.4-mini`** and **`text-embedding-3-small`**, nothing else |
| Rate limit, chat | Project, rate limits | **20 RPM**, TPM left at inherited 200,000 |
| Rate limit, embeddings | same | **20 RPM**, TPM left at inherited 1,000,000 |
| Rate limit, `*` default row | same | **10 RPM** |

Notes from doing it:

- **The "Enforce a hard limit" toggle is the whole step.** Off, the $5 is only a
  notification line and requests keep going. On, requests actually stop. Off is a warning,
  on is a limit.
- **The dashboard shows overrides as `your value / inherited value`.** Seeing
  `20 / 500 RPM` means your override is 20 and the organisation default was 500. If both
  numbers are identical you have not actually changed anything.
- **A model family and its dated snapshots share one rate limit.** The chat row read
  `gpt-5.4-mini` with `Shared limits: gpt-5.4-mini-2026-03-17` underneath, so setting the
  family covers the pinned snapshot. Nothing extra to do.
- **Leave TPM alone.** At 20 RPM and roughly 3,000 tokens per request you cannot exceed
  about 60,000 TPM, so a 200,000 ceiling is already unreachable. Lowering it gains nothing
  and breaks legitimate long conversations.
- **Set the `*` default row low anyway.** The allow-list already blocks every other model,
  so this is a second lock on the same door in case the allow-list is ever loosened.
- **Image-per-minute rows are irrelevant** unless the app calls an image endpoint. Ignore
  them.

### The consequence of a tight RPM, and it is not obvious

20 RPM is shared across the whole project, and an agent turn costs **two** requests, one
for the model to choose a tool and one for it to answer. So the real ceiling is about
**10 conversation turns per minute across all visitors at once.**

Fine for a portfolio demo. Tight if a post does well and thirty people arrive together.
This makes 429 responses a normal operating condition rather than a rare fault, which is
exactly why section 5 splits them from budget errors.

---

## 3. Layer 2: getting the key into the app without leaking it

**Never commit the key.** `.env` in `.gitignore`, and never a `secrets.toml` in the repo.
On Streamlit Community Cloud the key goes in the app's Settings, Secrets box, and nowhere
else.

### Two gotchas that cost me a deployment

**Streamlit copies secrets into `os.environ`, but lazily.** The variable only appears once
something has touched `st.secrets`. My app read `os.environ` directly, never touched
`st.secrets`, so on Cloud the environment stayed empty and constructing the OpenAI client
raised before the page could render a single element. Read secrets explicitly:

```python
def demo_api_key():
    """The shared key, read explicitly rather than left to the environment."""
    try:
        from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        from_secrets = ""      # no secrets.toml at all, which is fine locally
    return str(from_secrets or os.environ.get("OPENAI_API_KEY", "")).strip()
```

**The secrets box takes TOML, not `.env` syntax.** An unquoted line loads nothing at all,
not partially:

```
OPENAI_API_KEY=sk-proj-...     WRONG. Invalid TOML. Zero secrets loaded.
OPENAI_API_KEY="sk-proj-..."   Correct.
```

From outside, "invalid file" and "no secrets configured" look identical. Add a diagnostic
that reports **names only, never values**, shown only when the app is already broken:

```python
def secrets_diagnostic():
    """Why no key was found, in terms an operator can act on. Names, never values."""
    try:
        names = sorted(st.secrets.keys())
    except Exception as exc:
        return f"st.secrets could not be read ({type(exc).__name__})."
    if not names:
        return "No secrets are configured for this app."
    return "Secrets this app can see: " + ", ".join(names) + "."
```

That single line told me the key was present under the right name but the file had not
parsed. Without it I was guessing.

---

## 4. Layer 3: the in-app usage gate

Two parts: a free allowance per visitor, and a way for keen visitors to continue on their
own key.

### The counter must not live in session state

A per-session counter resets on page reload and in a new tab, which makes it no limit at
all. Use the process-wide resource cache, which returns the *same object* to every session
in the container:

```python
@st.cache_resource
def free_turns_used():
    """Turns spent per visitor, shared across every session in this app process.

    Deliberately not per-session: a session counter resets on reload and in a
    new tab, which makes it no limit at all. This survives both. It does reset
    when the app sleeps or redeploys, and everyone behind one NAT shares an
    entry, so treat it as budget-stretching rather than access control.
    """
    return {}


def visitor_id():
    return getattr(st.context, "ip_address", None) or "local"
```

The `getattr` fallback matters: on localhost the IP is `None`, so local development would
otherwise key everything under `None` or crash.

Consider bounding it, since it grows one entry per distinct IP until the app sleeps:
`@st.cache_resource(ttl="12h")`.

### The gate, computed once near the top of the script

```python
user_key   = st.session_state.user_api_key
own_key    = bool(user_key)
active_key = user_key or demo_api_key()

turns_left = max(0, FREE_TURNS - free_turns_used().get(visitor_id(), 0))
can_talk   = bool(active_key) and (own_key or turns_left > 0)
```

Three behaviours fall out of this and all three matter:

- **No key configured at all disables the input** with a plain message, instead of letting
  the client constructor throw a traceback across the whole page.
- **A visitor on their own key is never counted**, so the shared allowance is untouched.
- **The allowance is spent only on turns that actually used the shared key**, incremented
  after a successful call, not before.

### Bring-your-own-key input

```python
st.session_state.user_api_key = st.text_input(
    "OpenAI API key", type="password", placeholder="sk-...",
    value=st.session_state.user_api_key,
    help="Used only for this browser session. Never stored or logged.",
).strip()
```

`type="password"` masks it. It lives in session state and nowhere else: not written to
disk, not logged, not sent anywhere except the provider. **Say so in the help text**,
because you are asking a stranger for a credential and they deserve to know.

Rebuild your clients when the key changes, guarded by a dirty flag so it does not happen
on every rerun:

```python
if active_key and st.session_state.get("clients_for_key") != active_key:
    st.session_state.client = build_client(api_key=active_key)
    st.session_state.clients_for_key = active_key
```

**Set the allowance low.** I use three messages. Enough to demonstrate the app, not enough
to matter financially.

---

## 5. Layer 4: failing without leaking or crashing

A spent budget is an expected end state, not a bug. Handle it as one.

**The trap: OpenAI answers two completely different situations with HTTP 429.** "Your
budget is gone" and "you are going too fast" arrive looking almost identical. Treating
them the same means a visitor who happens to arrive during a busy minute is told the demo
is over, and if you also close their gate, their whole free allowance is destroyed by a
fifteen-second queue. With a deliberately low project RPM this is not hypothetical, it is
the common case.

Split them, and **check the terminal case first**, because a quota error is also delivered
as a 429 and would otherwise be misread as a speed problem:

```python
BUDGET_GONE = ("insufficient_quota", "exceeded your current quota",
               "billing_hard_limit")
TOO_FAST    = ("rate limit", "rate_limit", "429", "too many requests")

def _error_text(exc):
    return f"{type(exc).__name__} {exc}".lower()

def out_of_credit(exc):
    return any(m in _error_text(exc) for m in BUDGET_GONE)

def rate_limited(exc):
    """Transient. Must be checked AFTER out_of_credit."""
    return any(m in _error_text(exc) for m in TOO_FAST)
```

```python
try:
    run_agent(user_text)
except Exception as exc:
    if out_of_credit(exc):
        message = ("That key has no credit left. Check its billing."
                   if own_key else ALLOWANCE_SPENT)
        if not own_key:
            # Terminal. Close the gate even if the counter had turns left:
            # the budget is gone, so further attempts would just fail again.
            free_turns_used()[visitor_id()] = FREE_TURNS
    elif rate_limited(exc):
        # Transient. Deliberately does NOT touch the counter.
        message = ("That key is being rate limited. Wait a moment and retry."
                   if own_key else TOO_BUSY)
    else:
        message = "Sorry, something went wrong on my end. Please try again."
        print(f"agent error: {type(exc).__name__}: {exc}")   # server log only
```

Five deliberate choices here:

- **Terminal is checked before transient**, for the ordering reason above. Get this
  backwards and every quota error reads as "try again shortly", inviting the visitor to
  retry into a wall forever.
- **A transient error never decrements the allowance.** Increment the counter only after a
  successful call, so a failed turn costs the visitor nothing and there is nothing to
  refund.
- **Never print a raw exception to the page.** Provider errors can carry request details,
  internal identifiers, and occasionally fragments of the request itself. The real
  exception goes to the server log.
- **Distinguish "your key failed" from "the shared allowance is gone"**, since the fix
  differs depending on whose key was in play.
- **Audit your own logging.** `print(exc)` is safe for OpenAI errors, which do not echo the
  key, but if you log request payloads or use a library that does, check what actually
  reaches your logs before going public.

Test the classifier against real message strings rather than trusting the patterns by
eye. The case that matters is a quota error that also contains "429": it must come out
terminal.

---

## 6. Layer 5: what the repository publishes

Making the repo public publishes **everything in it, including history**. Two checks
before you flip that switch.

**Scan git history, not just the working tree.** A key committed and later removed is
still in the history and still valid.

```bash
git log -p | grep -i "sk-"
# or use a scanner such as gitleaks
```

If you find one, **rotating the key is the fix.** Rewriting history alone is not, because
clones and forks may already exist.

**If you ship a data file, scrub it.** My dataset is scraped classified adverts, so seller
prose contained phone numbers, emails and URLs. The pipeline's first clean caught most of
them and **twenty rows still got through**. The deployment step applies a second,
deliberately broader pass and refuses to write if anything survives:

```python
if after_scrub_count:
    raise SystemExit("contact details survived the scrub, do not publish this")
```

Make it a hard gate, not a warning. A false positive costs a few mangled words in a
description; a miss publishes a stranger's phone number under your name.

One practical note from doing this: a "looser" phone regex I wrote to replace the original
turned out to be **narrower** on `(541) 480-3265`, the commonest US format, and let a real
number through. Keep the working expression and only add to it. Verify by counting matches
before and after, not by reading the pattern.

**Keep deployment dependencies minimal.** Fewer packages is a smaller supply-chain surface
and a faster build. My offline pipeline needs torch and transformers; the app never
imports them, so they live in a separate optional group and are never installed on the
host.

---

## 7. What none of this protects against

Be honest about the gaps, because each one is a reason the platform-side cap is the thing
that actually saves you.

- **IP is spoofable.** Streamlit's own documentation says so directly. It is derived from
  the WebSocket connection and is not an authentication signal.
- **IPv6 makes per-IP counting weak even without spoofing.** A single user is typically
  handed an enormous address range, so cycling addresses is trivial.
- **Everyone behind one NAT shares an entry.** An office or a university shares three
  messages between hundreds of people. A usability cost, not a security one, but real.
- **The counter resets when the app sleeps or redeploys.** Community Cloud sleeps idle
  apps, which zeroes the dictionary.
- **Nothing stops prompt-directed abuse.** Someone can ask your domain-specific assistant
  to write their essay. The model allow-list limits how expensive that is; a system prompt
  that refuses off-topic requests helps, but treat it as politeness rather than
  enforcement.

The one-sentence summary, worth putting at the top of the README: **the in-app gate stops
one visitor draining the demo allowance in a sitting, and the spend cap is what stops you
losing money.**

---

## 8. Rotation and incident response

Rotate a key immediately if it has ever appeared in a screenshot, a commit, a pasted log,
a support ticket, or a chat with a tool. Partial exposure counts: a visible prefix plus a
screen recording is often enough.

Because the key is project-scoped, rotation is cheap. Create a new key in the same
project, update the Streamlit secret, delete the old key. Nothing else in your account is
affected.

If you suspect abuse rather than exposure: check the project's usage graph for the spike,
lower the RPM limit, and rotate. The spend cap means the loss is bounded whatever you find.

---

## 9. Implementation checklist

```
Before this app goes public with my OpenAI key, implement all of the following.

OPENAI PLATFORM (do these in the dashboard, not in code)
  [ ] dedicated project, project-scoped API key
  [ ] hard monthly spend cap, plus alerts at 50% and 80%
  [ ] model allow-list containing ONLY the models this app calls
  [ ] requests-per-minute limit around 20
  [ ] the exact model snapshot pinned in code matches the allow-list entry

IN THE APP
  [ ] read the key via st.secrets explicitly, with an os.environ fallback,
      because Streamlit only populates the environment after st.secrets is
      touched
  [ ] a diagnostic that lists configured secret NAMES only, never values,
      shown only when no key was found
  [ ] free-turn counter in @st.cache_resource keyed on st.context.ip_address,
      NOT in session state, with a fallback for a None IP on localhost
  [ ] a low allowance, three messages
  [ ] sidebar bring-your-own-key input with type="password", held in session
      state only, never logged, with help text saying so
  [ ] turns counted only when the shared key was used
  [ ] input disabled with a plain message when no key is configured, never a
      traceback
  [ ] budget errors and transient rate-limit errors handled SEPARATELY:
      terminal checked first, gate force-closed only on a spent budget, a 429
      throttle never costs the visitor a turn, real exception logged
      server-side only and never shown on the page

REPOSITORY
  [ ] .env and secrets.toml gitignored
  [ ] scan the FULL git history for committed keys, not just the working tree
  [ ] if any data file ships, scrub personal data with a hard gate that
      refuses to write on failure
  [ ] deployment requirements contain only what the app imports

DOCUMENT IT
  [ ] state in the README and in code comments that the in-app gate is cost
      smoothing, not access control; that IP addresses are spoofable and
      Streamlit documents them as unsuitable for security; and that the spend
      cap on the provider project is the real protection
```
