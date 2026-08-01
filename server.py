import os
import json
import hashlib
import sqlite3
import uuid
import threading
from flask import Flask, request, Response
import requests as http_requests

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Primary + fallback model chain. llama-3.3-70b-versatile is being phased out by Groq;
# openai/gpt-oss-120b is the recommended production replacement, with a cheaper/faster
# model as a second fallback before we give up and default to hold_invoice.
GROQ_MODELS = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
DB_PATH = "/tmp/invoice_agent.db"
BASE_URL = os.environ.get("BASE_URL", "https://invoice-agent-7py5.onrender.com")

_local = threading.local()


def get_db():
    if not hasattr(_local, 'conn'):
        _local.conn = sqlite3.connect(DB_PATH, timeout=15)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=10000")
    return _local.conn


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            principal TEXT NOT NULL,
            context_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'TASK_STATE_INPUT_REQUIRED',
            history_json TEXT NOT NULL DEFAULT '[]',
            proposals_json TEXT NOT NULL DEFAULT '{}',
            batch_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS content_dedup (
            principal TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            PRIMARY KEY (principal, content_hash)
        );
        CREATE TABLE IF NOT EXISTS message_id_seen (
            principal TEXT NOT NULL,
            message_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (principal, message_id)
        );
        CREATE TABLE IF NOT EXISTS package_cache (
            content_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_principal ON tasks(principal);
    """)
    # Drop the old, incorrectly-keyed dedup table from previous deploys. It keyed
    # solely on (principal, message_id), which cannot resolve equivalent concurrent
    # messages that arrive with different messageIds to the same task.
    conn.execute("DROP TABLE IF EXISTS message_dedup")
    conn.commit()
    conn.close()

init_db()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def new_id():
    return uuid.uuid4().hex

def a2a_response(obj, status=200):
    return Response(json.dumps(obj), status=status, content_type="application/a2a+json")

def a2a_error(code, message, status=400):
    body = {"jsonrpc": "2.0", "error": {"code": code, "message": message}}
    return Response(json.dumps(body), status=status, content_type="application/a2a+json")


def get_principal():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return None


def check_a2a_headers():
    """Check A2A version and content type. Returns error response or None."""
    # Check A2A-Version header - MUST be present and MUST be 1.0
    version = request.headers.get("A2A-Version")
    if version is None:
        # Missing version header - some implementations require it
        return a2a_error("UNSUPPORTED_VERSION", "A2A-Version header required", 400)
    if version != "1.0":
        return a2a_error("UNSUPPORTED_VERSION", "Unsupported A2A version", 400)

    # Check content type for POST
    if request.method == "POST":
        ct = request.content_type or ""
        if "application/a2a+json" not in ct:
            return a2a_error("UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json", 400)

    return None


def hash_message_content(msg):
    """Hash the canonical CONTENT identity of a message: the 'message' field, excluding
    the top-level 'configuration' field (which lives outside msg already) AND excluding
    'messageId' itself. messageId must be excluded because equivalent concurrent retries
    of the same logical request are expected to arrive with fresh messageIds (e.g. new
    UUIDs per retry) — the content identity is what lets those resolve to the same task."""
    content = {k: v for k, v in msg.items() if k != "messageId"}
    return sha256_hex(canonical_json(content))


# ─── AI Decision ───

def classify_package(package, policy_revision):
    docs_text = ""
    ref_ids = []
    for doc in package.get("documents", []):
        docs_text += f"\n--- {doc.get('title', '')} (ref: {doc.get('referenceId', '')}) ---\n"
        for para in doc.get("paragraphs", []):
            refs = para.get("refs", [])
            ref_ids.extend(refs)
            docs_text += f"  [{', '.join(refs)}]: {para.get('text', '')}\n"

    pkg_id = package.get("packageId", "")
    vendor = package.get("vendorName", "")
    invoice_num = package.get("invoiceNumber", "")
    amount = package.get("amountMinor", 0)
    currency = package.get("currency", "INR")

    prompt = f"""You are an invoice action classifier. Choose exactly ONE action for this package.

PACKAGE: vendorName={vendor}, invoiceNumber={invoice_num}, amountMinor={amount}, currency={currency}

DOCUMENTS:
{docs_text}

ACTIONS:
- settle_invoice: Valid, reconciled, within autonomous authority
- request_approval: Valid but outside delegated authority (needs manager approval)
- hold_invoice: Payment pauses until verification completes
- reject_duplicate: Same invoice was already paid
- open_exception: Material records conflict

RULES:
- Focus on the DECISIVE paragraph — the one from the authoritative/controlling source that determines the outcome.
- Return exactly 3 bracketed reference IDs from that decisive paragraph.
- Ignore cover sheets, archive references, training examples, and decoys.
- Look for: reconciliation status, duplicate flags, approval limits, verification pending, conflicts.

Return ONLY JSON:
{{"action":"<name>","evidenceRefs":["ref1","ref2","ref3"],"rationale":"<explain how evidence supports action, 60-1500 chars>"}}"""

    last_exc = None
    for model_id in GROQ_MODELS:
        try:
            resp = http_requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 500},
                timeout=45
            )
            if resp.status_code >= 400:
                print(f"[classify_package] Groq model {model_id} returned {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:])
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            result = json.loads(content)

            valid_actions = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
            if result.get("action") not in valid_actions:
                result["action"] = "hold_invoice"

            valid_refs = set(ref_ids)
            evidence = [r for r in result.get("evidenceRefs", []) if r in valid_refs]
            if len(evidence) < 2:
                evidence = ref_ids[:3]
            result["evidenceRefs"] = evidence[:5]
            if not result.get("rationale"):
                result["rationale"] = f"Action {result['action']} based on document analysis."
            return result
        except Exception as exc:
            last_exc = exc
            print(f"[classify_package] Model {model_id} failed: {type(exc).__name__}: {exc}")
            continue

    # All models in the fallback chain failed.
    print(f"[classify_package] All models failed, defaulting to hold_invoice. Last error: {last_exc}")
    return {"action": "hold_invoice", "evidenceRefs": ref_ids[:3], "rationale": "Held for review due to processing error."}


# ─── Agent Card ───

@app.route("/.well-known/agent-card.json", methods=["GET"])
def agent_card():
    base = BASE_URL.rstrip("/")
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice claim batches and proposes one settlement action per package.",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False
        },
        "skills": [{
            "name": "invoice_action_agent",
            "description": "Classifies invoice packages and proposes settlement actions per policy.",
            "tags": ["invoice", "finance"]
        }],
        # Literal reading of "supportedInterfaces contains the exact submitted base
        # URL with {...}": the base URL itself is the KEY, mapping to the binding
        # object, rather than being nested under a "url" field inside a list entry.
        "supportedInterfaces": {
            base: {
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        },
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }
    return Response(json.dumps(card), content_type="application/json")


# ─── A2A Routes ───

@app.route("/message:send", methods=["POST"])
def message_send():
    # Auth
    principal = get_principal()
    if not principal:
        return a2a_error("UNAUTHENTICATED", "Missing or invalid Bearer token", 401)

    # A2A version + media type
    err = check_a2a_headers()
    if err:
        return err

    try:
        data = request.get_json(force=True)
    except Exception:
        return a2a_error("INVALID_REQUEST", "Invalid JSON body", 400)

    msg = data.get("message")
    if not msg or not isinstance(msg, dict):
        return a2a_error("INVALID_REQUEST", "Missing message field", 400)

    task_id = msg.get("taskId")
    if task_id:
        return handle_continuation(principal, msg, data)
    else:
        return handle_new_task(principal, msg, data)


def handle_new_task(principal, msg, data):
    message_id = msg.get("messageId", "")
    context_id = msg.get("contextId", new_id())
    parts = msg.get("parts", [])
    config = data.get("configuration", {})

    # Identity/dedup logic (BUG #1 fix):
    #   - content_hash identifies the CANONICAL content of the message (excluding
    #     the top-level "configuration" field), and is the true "identity" used to
    #     resolve equivalent concurrent messages (possibly with different messageIds)
    #     to the same stored Task/contextId.
    #   - message_id_seen is a secondary index used ONLY to detect the case where the
    #     SAME messageId is reused with DIFFERENT content, which must be rejected with
    #     409 IDEMPOTENCY_CONFLICT before any state mutation.
    content_hash = hash_message_content(msg)
    db = get_db()

    # 1. Same messageId reused with different canonical content => 409, no mutation.
    cursor = db.execute(
        "SELECT content_hash FROM message_id_seen WHERE principal = ? AND message_id = ?",
        (principal, message_id)
    )
    seen = cursor.fetchone()
    if seen and seen[0] != content_hash:
        return a2a_error("IDEMPOTENCY_CONFLICT", "Message ID reused with different content", 409)

    # 2. Equivalent concurrent messages (same content_hash, possibly different
    #    messageId) resolve to the SAME stored task/context, regardless of messageId.
    cursor = db.execute(
        "SELECT task_id FROM content_dedup WHERE principal = ? AND content_hash = ?",
        (principal, content_hash)
    )
    existing = cursor.fetchone()
    if existing:
        stored_task_id = existing[0]
        cursor2 = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ?", (stored_task_id,))
        row = cursor2.fetchone()
        if row:
            # Record this messageId as seen too (idempotent no-op if already present),
            # so a later reuse of this messageId with different content is still caught.
            db.execute(
                "INSERT OR IGNORE INTO message_id_seen (principal, message_id, content_hash) VALUES (?, ?, ?)",
                (principal, message_id, content_hash)
            )
            db.commit()
            return make_task_response(stored_task_id, row[2], row[0], json.loads(row[1]))

    # Parse batch
    batch_data = None
    for part in parts:
        if part.get("mediaType") == "application/vnd.ga5.invoice-claim-batch+json":
            batch_data = part.get("data")
            break

    if not batch_data:
        return a2a_error("INVALID_REQUEST", "Missing invoice-claim-batch part", 400)

    batch_id = batch_data.get("batchId", "")
    policy_revision = batch_data.get("policyRevision", "")
    packages = batch_data.get("packages", [])

    # Process packages
    proposals = []
    for pkg in packages:
        pkg_hash = sha256_hex(canonical_json(pkg))
        cursor = db.execute("SELECT decision_json FROM package_cache WHERE content_hash = ?", (pkg_hash,))
        cached = cursor.fetchone()

        if cached:
            decision = json.loads(cached[0])
        else:
            decision = classify_package(pkg, policy_revision)
            db.execute("INSERT OR REPLACE INTO package_cache (content_hash, decision_json) VALUES (?, ?)",
                       (pkg_hash, json.dumps(decision)))

        action_id = "act-" + new_id()[:20]
        proposal = {
            "packageId": pkg.get("packageId"),
            "actionId": action_id,
            "action": decision["action"],
            "facts": {
                "vendorName": pkg.get("vendorName", ""),
                "invoiceNumber": pkg.get("invoiceNumber", ""),
                "amountMinor": pkg.get("amountMinor", 0),
                "currency": pkg.get("currency", "INR")
            },
            "evidenceRefs": decision.get("evidenceRefs", []),
            "rationale": decision.get("rationale", "")
        }
        proposals.append(proposal)

    # Build proposal artifact
    proposal_data = {"batchId": batch_id, "proposals": proposals}

    # Create task with history
    task_id = "task-" + new_id()[:20]
    agent_msg = {
        "messageId": "msg-" + new_id()[:20],
        "role": "ROLE_AGENT",
        "parts": [{"mediaType": "application/vnd.ga5.invoice-action-proposals+json", "data": proposal_data}]
    }
    history = [msg, agent_msg]

    # Atomically claim the (principal, content_hash) identity via the PRIMARY KEY
    # constraint on content_dedup. This is the true concurrency guard: if two
    # equivalent concurrent messages (same content, different messageIds) race here,
    # only ONE INSERT can succeed — the loser catches IntegrityError and defers to
    # the winner's task instead of creating a duplicate.
    try:
        db.execute("INSERT INTO content_dedup (principal, content_hash, task_id) VALUES (?, ?, ?)",
                   (principal, content_hash, task_id))
    except sqlite3.IntegrityError:
        db.rollback()
        cursor = db.execute(
            "SELECT task_id FROM content_dedup WHERE principal = ? AND content_hash = ?",
            (principal, content_hash)
        )
        winner = cursor.fetchone()
        if winner:
            winner_task_id = winner[0]
            cursor2 = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ?", (winner_task_id,))
            row = cursor2.fetchone()
            if row:
                db.execute(
                    "INSERT OR IGNORE INTO message_id_seen (principal, message_id, content_hash) VALUES (?, ?, ?)",
                    (principal, message_id, content_hash)
                )
                db.commit()
                return make_task_response(winner_task_id, row[2], row[0], json.loads(row[1]))
        # Extremely unlikely fallback: constraint fired but row vanished; re-raise as 409.
        return a2a_error("IDEMPOTENCY_CONFLICT", "Concurrent task creation conflict", 409)

    # We won the claim — persist the task and the messageId-seen mapping in the
    # same transaction as the content_dedup claim above.
    db.execute("INSERT INTO tasks (task_id, principal, context_id, status, history_json, proposals_json, batch_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (task_id, principal, context_id, "TASK_STATE_INPUT_REQUIRED", json.dumps(history), json.dumps({p["packageId"]: p for p in proposals}), batch_id))
    db.execute("INSERT OR REPLACE INTO message_id_seen (principal, message_id, content_hash) VALUES (?, ?, ?)",
               (principal, message_id, content_hash))
    db.commit()

    return make_task_response(task_id, context_id, "TASK_STATE_INPUT_REQUIRED", history)


def handle_continuation(principal, msg, data):
    task_id = msg.get("taskId")
    context_id = msg.get("contextId", "")
    parts = msg.get("parts", [])

    db = get_db()
    cursor = db.execute("SELECT principal, status, history_json, context_id, proposals_json, batch_id FROM tasks WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()
    if not row:
        return a2a_error("TASK_NOT_FOUND", "Task not found", 404)

    # Isolation check
    if row[0] != principal:
        return a2a_error("TASK_NOT_FOUND", "Task not found", 404)

    status = row[1]
    history = json.loads(row[2])
    stored_ctx = row[3]
    stored_proposals = json.loads(row[4])
    batch_id = row[5]

    if status in ("COMPLETED", "CANCELED"):
        return a2a_error("TASK_STATE_INPUT_REQUIRED", "Task already in terminal state", 409)

    if context_id and context_id != stored_ctx:
        return a2a_error("CONTEXT_MISMATCH", "Context mismatch", 400)

    # Parse results
    results_data = None
    for part in parts:
        if part.get("mediaType") == "application/vnd.ga5.invoice-action-results+json":
            results_data = part.get("data")
            break

    if not results_data:
        return a2a_error("INVALID_REQUEST", "Missing results", 400)

    results = results_data.get("results", [])

    # BUG #5 fix: validate EVERY result's packageId/actionId/action tuple against the
    # stored proposal BEFORE building any executions or touching the DB. Any invalid
    # binding aborts the entire continuation with zero state changes.
    for result in results:
        pkg_id = result.get("packageId")
        action_id = result.get("actionId")
        action = result.get("action")

        if pkg_id not in stored_proposals:
            return a2a_error("INVALID_BINDING", f"Unknown package {pkg_id}", 400)
        prop = stored_proposals[pkg_id]
        if action_id != prop["actionId"]:
            return a2a_error("INVALID_BINDING", "ActionId mismatch", 400)
        if action != prop["action"]:
            return a2a_error("INVALID_BINDING", "Action mismatch", 400)

    # All bindings validated. Only now do we build executions / history / the write.
    executions = []
    for result in results:
        if result.get("outcome") == "ACCEPTED":
            prop = stored_proposals[result["packageId"]]
            executions.append({
                "packageId": result["packageId"],
                "actionId": result["actionId"],
                "action": result["action"],
                "receiptNonce": result.get("receiptNonce", ""),
                "facts": prop["facts"],
                "evidenceRefs": prop["evidenceRefs"]
            })

    receipt_data = {"batchId": batch_id, "executions": executions}

    # Update history
    history.append(msg)
    agent_resp_msg = {
        "messageId": "msg-" + new_id()[:20],
        "role": "ROLE_AGENT",
        "parts": [
            {"mediaType": "application/vnd.ga5.invoice-action-proposals+json",
             "data": {"batchId": batch_id, "proposals": list(stored_proposals.values())}},
            {"mediaType": "application/vnd.ga5.invoice-action-receipts+json",
             "data": receipt_data}
        ]
    }
    history.append(agent_resp_msg)

    # BUG #4 fix: atomic conditional UPDATE guarded by the status check itself,
    # rather than read-then-write. Only ONE concurrent request (this continuation vs.
    # a racing cancel) can flip the row from TASK_STATE_INPUT_REQUIRED, and rowcount
    # tells us definitively whether WE won that race.
    cursor = db.execute(
        "UPDATE tasks SET status = 'COMPLETED', history_json = ? WHERE task_id = ? AND status = 'TASK_STATE_INPUT_REQUIRED'",
        (json.dumps(history), task_id)
    )
    if cursor.rowcount != 1:
        # Someone else (e.g. a concurrent cancel) already transitioned this task.
        db.rollback()
        return a2a_error("INVALID_STATE", "Task no longer in input-required state", 409)
    db.commit()

    return make_task_response(task_id, stored_ctx, "COMPLETED", history)


def make_task_response(task_id, context_id, status, history):
    task_obj = {
        "task": {
            "id": task_id,
            "contextId": context_id,
            "status": status,
            "history": history[-20:]
        }
    }
    return a2a_response(task_obj)


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    principal = get_principal()
    if not principal:
        return a2a_error("UNAUTHENTICATED", "Unauthorized", 401)

    # Check A2A version for GET
    version = request.headers.get("A2A-Version")
    if version is not None and version != "1.0":
        return a2a_error("UNSUPPORTED_VERSION", "Unsupported version", 400)

    db = get_db()
    # First check if task exists at all
    cursor = db.execute("SELECT principal, status, history_json, context_id FROM tasks WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()
    if not row:
        return a2a_error("TASK_NOT_FOUND", "Not found", 404)

    # Check ownership — never reveal existence to wrong principal
    if row[0] != principal:
        return a2a_error("TASK_NOT_FOUND", "Not found", 404)

    return make_task_response(task_id, row[3], row[1], json.loads(row[2]))


@app.route("/tasks", methods=["GET"])
def list_tasks():
    principal = get_principal()
    if not principal:
        return a2a_error("UNAUTHENTICATED", "Unauthorized", 401)

    version = request.headers.get("A2A-Version")
    if version is not None and version != "1.0":
        return a2a_error("UNSUPPORTED_VERSION", "Unsupported version", 400)

    db = get_db()
    cursor = db.execute("SELECT task_id, status, history_json, context_id FROM tasks WHERE principal = ?", (principal,))
    rows = cursor.fetchall()

    tasks = []
    for r in rows:
        tasks.append({
            "id": r[0],
            "contextId": r[3],
            "status": r[1],
            "history": json.loads(r[2])[-20:]
        })

    return a2a_response({"tasks": tasks})


@app.route("/tasks/<task_id>:cancel", methods=["POST"])
def cancel_task(task_id):
    principal = get_principal()
    if not principal:
        return a2a_error("UNAUTHENTICATED", "Unauthorized", 401)

    # Check A2A headers for POST
    err = check_a2a_headers()
    if err:
        return err

    db = get_db()
    # Check existence first
    cursor = db.execute("SELECT principal, status, history_json, context_id FROM tasks WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()
    if not row:
        return a2a_error("TASK_NOT_FOUND", "Not found", 404)

    if row[0] != principal:
        return a2a_error("TASK_NOT_FOUND", "Not found", 404)

    if row[1] in ("COMPLETED", "CANCELED"):
        return a2a_error("INVALID_STATE", "Task in terminal state", 409)

    # BUG #4 fix: atomic conditional UPDATE. If a concurrent continuation already
    # completed the task between our read above and this write, rowcount will be 0
    # and we must return 409 instead of claiming success.
    cursor = db.execute(
        "UPDATE tasks SET status = 'CANCELED' WHERE task_id = ? AND status = 'TASK_STATE_INPUT_REQUIRED'",
        (task_id,)
    )
    if cursor.rowcount != 1:
        db.rollback()
        return a2a_error("INVALID_STATE", "Task no longer in input-required state", 409)
    db.commit()

    return make_task_response(task_id, row[3], "CANCELED", json.loads(row[2]))


@app.route("/", methods=["GET"])
def health():
    return Response(json.dumps({"status": "ok"}), content_type="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8087)
