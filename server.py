import os
import json
import hashlib
import sqlite3
import uuid
import threading
from flask import Flask, request, Response, g
import requests as http_requests

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
DB_PATH = "/tmp/invoice_agent.db"
BASE_URL = os.environ.get("BASE_URL", "https://invoice-agent-7py5.onrender.com")  # Set on Render

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
            status TEXT NOT NULL DEFAULT 'INPUT_REQUIRED',
            history_json TEXT NOT NULL DEFAULT '[]',
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


# ─── Utilities ───

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def new_id():
    return uuid.uuid4().hex


def make_resp(obj, status=200, media_type="application/json"):
    body = json.dumps(obj)
    return Response(body, status=status, content_type=media_type)


def a2a_resp(obj, status=200):
    return Response(json.dumps(obj), status=status, content_type="application/a2a+json")


def error_resp(code, message, status=400):
    return Response(json.dumps({"error": {"code": code, "message": message}}), status=status, content_type="application/a2a+json")


def get_principal():
    """Extract bearer token as principal. Returns None if missing."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def hash_message(msg):
    """Hash canonical message content (excluding configuration)."""
    msg_copy = {k: v for k, v in msg.items() if k != "configuration"}
    return sha256_hex(canonical_json(msg_copy))


# ─── AI Decision ───

def classify_package(package, policy_revision):
    """Use AI to classify an invoice package into an action."""
    # Build content for LLM
    docs_text = ""
    ref_ids = []
    for i, doc in enumerate(package.get("documents", [])):
        docs_text += f"\n--- Document {i+1}: {doc.get('title', '')} (ref: {doc.get('referenceId', '')}) ---\n"
        for para in doc.get("paragraphs", []):
            refs = para.get("refs", [])
            ref_ids.extend(refs)
            docs_text += f"  [{', '.join(refs)}]: {para.get('text', '')}\n"

    pkg_id = package.get("packageId", "")
    vendor = package.get("vendorName", "")
    invoice_num = package.get("invoiceNumber", "")
    amount = package.get("amountMinor", 0)
    currency = package.get("currency", "INR")

    prompt = f"""You are an invoice action classifier. Analyze this invoice package and choose exactly ONE action.

POLICY REVISION: {policy_revision}

PACKAGE:
  packageId: {pkg_id}
  vendorName: {vendor}
  invoiceNumber: {invoice_num}
  amountMinor: {amount}
  currency: {currency}

DOCUMENTS:
{docs_text}

ACTIONS (choose exactly one):
- settle_invoice: Valid, reconciled, and within autonomous authority. Use when all documents confirm the invoice is legitimate, reconciled with PO/delivery, and within approval limits.
- request_approval: Commercially valid, but outside delegated authority. Use when invoice is valid but amount exceeds autonomous settlement threshold or needs manager sign-off.
- hold_invoice: Payment pauses until a stated verification completes. Use when documents indicate pending verification, missing confirmation, or unresolved query.
- reject_duplicate: The same commercial invoice was already paid. Use when documents show this exact invoice number was previously settled/paid.
- open_exception: Material records conflict and need an exception workflow. Use when documents show contradictions, mismatches between PO and invoice, or pricing discrepancies.

INSTRUCTIONS:
- Read ALL documents carefully. Look for: reconciliation status, approval notes, duplicate flags, amount thresholds, verification status.
- The decisive paragraph is the one that DETERMINES the action — often from an internal system or authority note.
- Ignore training decoys, old examples, and irrelevant action words. Focus on the CURRENT STATE of this package.
- Return EXACTLY 3 evidence refs from the decisive paragraph (the one that determines your action). Do NOT include cover-sheet refs, archive refs, or training decoys.

Respond with ONLY JSON:
{{"action":"<one action>","evidenceRefs":["ref1","ref2","ref3"],"rationale":"<60-1500 chars: name the action, explain why based on evidence>"}}"""

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

        # Validate refs
        valid_refs = set(ref_ids)
        evidence = [r for r in result.get("evidenceRefs", []) if r in valid_refs]
        if len(evidence) < 2 and ref_ids:
            evidence = ref_ids[:3]
        result["evidenceRefs"] = evidence[:5]

        if not result.get("rationale"):
            result["rationale"] = f"Action {result['action']} chosen based on document analysis."

        return result
    except Exception:
        return {
            "action": "hold_invoice",
            "evidenceRefs": ref_ids[:3] if ref_ids else [],
            "rationale": "Held for manual review due to classification error."
        }


# ─── Agent Card ───

@app.route("/.well-known/agent-card.json", methods=["GET"])
def agent_card():
    base = BASE_URL or request.url_root.rstrip("/")
    card = {
        "name": "Invoice Action Agent",
        "description": "Classifies invoice packages and proposes settlement actions per policy.",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False
        },
        "skills": [
            {
                "name": "invoice_action_agent",
                "description": "Reads invoice claim batches and proposes one action per package.",
                "tags": ["invoice", "finance", "classification"]
            }
        ],
        "supportedInterfaces": [
            {
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
                "url": base
            }
        ],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }
    return Response(json.dumps(card), content_type="application/json")


# ─── A2A Endpoints ───

@app.route("/message:send", methods=["POST"])
def message_send():
    # Auth check
    principal = get_principal()
    if not principal:
        return error_resp("UNAUTHORIZED", "Missing Bearer token", 401)

    # Version check
    a2a_version = request.headers.get("A2A-Version", "")
    if a2a_version != "1.0":
        return error_resp("VERSION_NOT_SUPPORTED", "Require A2A-Version: 1.0", 400)

    # Content type check
    ct = request.content_type or ""
    if "application/a2a+json" not in ct and "application/json" not in ct:
        return error_resp("INVALID_MEDIA_TYPE", "Require application/a2a+json", 400)

    try:
        data = request.get_json(force=True)
    except Exception:
        return error_resp("INVALID_REQUEST", "Invalid JSON", 400)

    msg = data.get("message")
    if not msg:
        return error_resp("INVALID_REQUEST", "Missing message", 400)

    message_id = msg.get("messageId")
    role = msg.get("role")
    parts = msg.get("parts", [])
    task_id = msg.get("taskId")
    context_id = msg.get("contextId", new_id())

    if role != "ROLE_USER":
        return error_resp("INVALID_REQUEST", "Expected ROLE_USER", 400)

    db = get_db()

    # Determine if this is initial message or continuation
    if task_id:
        # Continuation (results coming back)
        return handle_continuation(db, principal, msg, data)
    else:
        # New task (initial batch)
        return handle_new_task(db, principal, msg, data)


def handle_new_task(db, principal, msg, data):
    message_id = msg.get("messageId")
    context_id = msg.get("contextId", new_id())
    parts = msg.get("parts", [])

    # Message deduplication
    msg_hash = hash_message(msg)
    cursor = db.execute("SELECT task_id FROM message_dedup WHERE principal = ? AND message_hash = ?", (principal, msg_hash))
    existing = cursor.fetchone()
    if existing:
        # Return existing task
        task_id = existing[0]
        cursor2 = db.execute("SELECT status, history_json FROM tasks WHERE task_id = ? AND principal = ?", (task_id, principal))
        task_row = cursor2.fetchone()
        if task_row:
            return build_task_response(task_id, context_id, task_row[0], json.loads(task_row[1]))

    # Parse the batch
    batch_data = None
    for part in parts:
        if part.get("mediaType") == "application/vnd.ga5.invoice-claim-batch+json":
            batch_data = part.get("data")
            break

    if not batch_data:
        return error_resp("INVALID_REQUEST", "Missing invoice-claim-batch part", 400)

    batch_id = batch_data.get("batchId")
    policy_revision = batch_data.get("policyRevision", "")
    packages = batch_data.get("packages", [])

    if not packages:
        return error_resp("INVALID_REQUEST", "No packages in batch", 400)

    # Process packages (with caching)
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
    proposal_artifact = {
        "batchId": batch_id,
        "proposals": proposals
    }

    # Create task
    task_id = "task-" + new_id()[:20]

    # History: initial message + our response
    history = [
        msg,
        {
            "messageId": "msg-" + new_id()[:20],
            "role": "ROLE_AGENT",
            "parts": [{
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": proposal_artifact
            }]
        }
    ]

    # Store task
    db.execute("INSERT INTO tasks (task_id, principal, context_id, status, history_json) VALUES (?, ?, ?, ?, ?)",
               (task_id, principal, context_id, "TASK_STATE_INPUT_REQUIRED", json.dumps(history)))
    db.execute("INSERT OR REPLACE INTO message_dedup (principal, message_hash, task_id) VALUES (?, ?, ?)",
               (principal, msg_hash, task_id))
    db.commit()

    return build_task_response(task_id, context_id, "TASK_STATE_INPUT_REQUIRED", history)


def handle_continuation(db, principal, msg, data):
    task_id = msg.get("taskId")
    context_id = msg.get("contextId", "")
    parts = msg.get("parts", [])

    # Verify task belongs to this principal
    cursor = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ? AND principal = ?", (task_id, principal))
    task_row = cursor.fetchone()
    if not task_row:
        return error_resp("TASK_NOT_FOUND", "Task not found", 404)

    current_status, history_json, stored_context = task_row
    history = json.loads(history_json)

    # If already completed, reject
    if current_status == "COMPLETED":
        return error_resp("TASK_ALREADY_COMPLETED", "Task already completed", 409)
    if current_status == "CANCELED":
        return error_resp("TASK_CANCELED", "Task was canceled", 409)

    # Verify context matches
    if context_id and context_id != stored_context:
        return error_resp("CONTEXT_MISMATCH", "Context ID mismatch", 400)

    # Parse results
    results_data = None
    for part in parts:
        if part.get("mediaType") == "application/vnd.ga5.invoice-action-results+json":
            results_data = part.get("data")
            break

    if not results_data:
        return error_resp("INVALID_REQUEST", "Missing results part", 400)

    # Get our proposals from history
    our_proposals = {}
    for h_msg in history:
        if h_msg.get("role") == "ROLE_AGENT":
            for part in h_msg.get("parts", []):
                if part.get("mediaType") == "application/vnd.ga5.invoice-action-proposals+json":
                    for p in part.get("data", {}).get("proposals", []):
                        our_proposals[p["packageId"]] = p

    # Validate results against our proposals
    batch_id = results_data.get("batchId")
    results = results_data.get("results", [])

    executions = []
    for result in results:
        pkg_id = result.get("packageId")
        action_id = result.get("actionId")
        action = result.get("action")
        outcome = result.get("outcome")
        receipt_nonce = result.get("receiptNonce")

        # Verify matches stored proposal
        if pkg_id not in our_proposals:
            return error_resp("INVALID_BINDING", f"Unknown package {pkg_id}", 400)

        stored = our_proposals[pkg_id]
        if action_id != stored["actionId"]:
            return error_resp("INVALID_BINDING", "ActionId mismatch", 400)
        if action != stored["action"]:
            return error_resp("INVALID_BINDING", "Action mismatch", 400)

        if outcome == "ACCEPTED":
            executions.append({
                "packageId": pkg_id,
                "actionId": action_id,
                "action": action,
                "receiptNonce": receipt_nonce,
                "facts": stored["facts"],
                "evidenceRefs": stored["evidenceRefs"]
            })
        # REJECTED ones are not included in executions

    # Build receipt artifact
    receipt_artifact = {
        "batchId": batch_id,
        "executions": executions
    }

    # Add continuation message and our response to history
    history.append(msg)
    agent_msg = {
        "messageId": "msg-" + new_id()[:20],
        "role": "ROLE_AGENT",
        "parts": [
            {
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": {"batchId": batch_id, "proposals": list(our_proposals.values())}
            },
            {
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": receipt_artifact
            }
        ]
    }
    history.append(agent_msg)

    # Mark task completed
    db.execute("UPDATE tasks SET status = 'COMPLETED', history_json = ? WHERE task_id = ?",
               (json.dumps(history), task_id))
    db.commit()

    return build_task_response(task_id, stored_context, "COMPLETED", history)


def build_task_response(task_id, context_id, status, history):
    # Trim history to last 20 messages
    trimmed = history[-20:] if len(history) > 20 else history

    task = {
        "task": {
            "id": task_id,
            "contextId": context_id,
            "status": status,
            "history": trimmed
        }
    }
    return Response(json.dumps(task), status=200, content_type="application/a2a+json")


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    principal = get_principal()
    if not principal:
        return error_resp("UNAUTHORIZED", "Missing Bearer token", 401)

    db = get_db()
    cursor = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ? AND principal = ?", (task_id, principal))
    row = cursor.fetchone()
    if not row:
        return error_resp("TASK_NOT_FOUND", "Task not found", 404)

    status, history_json, context_id = row
    history = json.loads(history_json)
    return build_task_response(task_id, context_id, status, history)


@app.route("/tasks", methods=["GET"])
def list_tasks():
    principal = get_principal()
    if not principal:
        return error_resp("UNAUTHORIZED", "Missing Bearer token", 401)

    db = get_db()
    cursor = db.execute("SELECT task_id, status, history_json, context_id FROM tasks WHERE principal = ? ORDER BY created_at DESC", (principal,))
    rows = cursor.fetchall()

    tasks = []
    for row in rows:
        tid, status, hist_json, ctx_id = row
        history = json.loads(hist_json)
        tasks.append({
            "id": tid,
            "contextId": ctx_id,
            "status": status,
            "history": history[-20:]
        })

    return Response(json.dumps({"tasks": tasks}), status=200, content_type="application/a2a+json")


@app.route("/tasks/<task_id>:cancel", methods=["POST"])
def cancel_task(task_id):
    principal = get_principal()
    if not principal:
        return error_resp("UNAUTHORIZED", "Missing Bearer token", 401)

    db = get_db()
    cursor = db.execute("SELECT status, history_json, context_id FROM tasks WHERE task_id = ? AND principal = ?", (task_id, principal))
    row = cursor.fetchone()
    if not row:
        return error_resp("TASK_NOT_FOUND", "Task not found", 404)

    status, history_json, context_id = row

    if status in ("COMPLETED", "CANCELED"):
        return error_resp("TASK_FINAL", "Task already in terminal state", 409)

    # Cancel the task
    db.execute("UPDATE tasks SET status = 'CANCELED' WHERE task_id = ?", (task_id,))
    db.commit()

    history = json.loads(history_json)
    return build_task_response(task_id, context_id, "CANCELED", history)


@app.route("/", methods=["GET"])
def health():
    return Response(json.dumps({"status": "ok"}), content_type="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8087)
