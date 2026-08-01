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
GROQ_MODEL = "llama-3.3-70b-versatile"
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
        CREATE TABLE IF NOT EXISTS message_dedup (
            principal TEXT NOT NULL,
            message_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            PRIMARY KEY (principal, message_hash)
        );
        CREATE TABLE IF NOT EXISTS package_cache (
            content_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_principal ON tasks(principal);
    """)
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
    # Check A2A-Version header
    version = request.headers.get("A2A-Version", "")
    if version and version != "1.0":
        return a2a_error("VERSION_NOT_SUPPORTED", "Only A2A-Version: 1.0 is supported", 400)
    if not version:
        return a2a_error("VERSION_REQUIRED", "A2A-Version header required", 400)

    # Check content type for POST
    if request.method == "POST":
        ct = request.content_type or ""
        if "application/a2a+json" not in ct:
            return a2a_error("INVALID_MEDIA_TYPE", "Content-Type must be application/a2a+json", 400)

    return None


def hash_message_content(msg):
    """Hash the canonical message content (only the 'message' field, ignoring configuration)."""
    return sha256_hex(canonical_json(msg))


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

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 500},
            timeout=45
        )
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
    except Exception:
        return {"action": "hold_invoice", "evidenceRefs": ref_ids[:3], "rationale": "Held for review due to processing error."}


# ─── Agent Card ───

@app.route("/.well-known/agent-card.json", methods=["GET"])
def agent_card():
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
        "supportedInterfaces": [{
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
            "url": BASE_URL
        }],
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
        return a2a_error("ROLE_USER", "Missing or invalid Bearer token", 401)

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

    # Dedup by message content
    msg_hash = hash_message_content(msg)
    db = get_db()

    cursor = db.execute("SELECT task_id FROM message_dedup WHERE principal = ? AND message_hash = ?", (principal, msg_hash))
    existing = cursor.fetchone()
    if existing:
        task_id = existing[0]
        cursor2 = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor2.fetchone()
        if row:
            return make_task_response(task_id, row[2], row[0], json.loads(row[1]))

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

    # Store
    db.execute("INSERT INTO tasks (task_id, principal, context_id, status, history_json, proposals_json, batch_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (task_id, principal, context_id, "TASK_STATE_INPUT_REQUIRED", json.dumps(history), json.dumps({p["packageId"]: p for p in proposals}), batch_id))
    db.execute("INSERT OR REPLACE INTO message_dedup (principal, message_hash, task_id) VALUES (?, ?, ?)",
               (principal, msg_hash, task_id))
    db.commit()

    return make_task_response(task_id, context_id, "TASK_STATE_INPUT_REQUIRED", history)


def handle_continuation(principal, msg, data):
    task_id = msg.get("taskId")
    context_id = msg.get("contextId", "")
    parts = msg.get("parts", [])

    db = get_db()
    cursor = db.execute("SELECT status, history_json, context_id, proposals_json, batch_id FROM tasks WHERE task_id = ? AND principal = ?",
                        (task_id, principal))
    row = cursor.fetchone()
    if not row:
        return a2a_error("TASK_NOT_FOUND", "Task not found or access denied", 404)

    status, history_json, stored_ctx, proposals_json, batch_id = row
    history = json.loads(history_json)
    stored_proposals = json.loads(proposals_json)

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

    # Validate bindings
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

    # Build executions (accepted only)
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

    # Complete task
    db.execute("UPDATE tasks SET status = 'COMPLETED', history_json = ? WHERE task_id = ?",
               (json.dumps(history), task_id))
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
        return a2a_error("ROLE_USER", "Unauthorized", 401)

    err = check_a2a_headers_get()
    if err:
        return err

    db = get_db()
    cursor = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ? AND principal = ?", (task_id, principal))
    row = cursor.fetchone()
    if not row:
        return a2a_error("TASK_NOT_FOUND", "Not found", 404)

    return make_task_response(task_id, row[2], row[0], json.loads(row[1]))


@app.route("/tasks", methods=["GET"])
def list_tasks():
    principal = get_principal()
    if not principal:
        return a2a_error("ROLE_USER", "Unauthorized", 401)

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
        return a2a_error("ROLE_USER", "Unauthorized", 401)

    db = get_db()
    cursor = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ? AND principal = ?", (task_id, principal))
    row = cursor.fetchone()
    if not row:
        return a2a_error("TASK_NOT_FOUND", "Not found", 404)

    if row[0] in ("COMPLETED", "CANCELED"):
        return a2a_error("INVALID_STATE", "Task in terminal state", 409)

    db.execute("UPDATE tasks SET status = 'CANCELED' WHERE task_id = ?", (task_id,))
    db.commit()

    return make_task_response(task_id, row[2], "CANCELED", json.loads(row[1]))


def check_a2a_headers_get():
    """For GET requests, just check version."""
    version = request.headers.get("A2A-Version", "")
    if version and version != "1.0":
        return a2a_error("VERSION_NOT_SUPPORTED", "Only A2A-Version: 1.0 supported", 400)
    return None


@app.route("/", methods=["GET"])
def health():
    return Response(json.dumps({"status": "ok"}), content_type="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8087)
