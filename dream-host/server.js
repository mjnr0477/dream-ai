import http from "node:http";
import { createHash, randomUUID, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import {
  CopyObjectCommand,
  DeleteObjectCommand,
  GetObjectCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client
} from "@aws-sdk/client-s3";
import { NotebookQueue } from "./notebook-core.js";
import { getArtifact, listArtifacts, putArtifact } from "./artifact-store.js";
import { createR2ArtifactStore } from "./r2-artifact-store.js";
import { createR2StateStore } from "./r2-state-store.js";

const PORT = Number(process.env.PORT || 10000);
function normalizeToken(value) {
  const token = String(value || "").trim();
  return token.toLowerCase().startsWith("bearer ") ? token.slice(7).trim() : token;
}

const WORKER_TOKEN = normalizeToken(process.env.DREAM_HOST_WORKER_TOKEN);
console.log("DREAM_HOST_WORKER_AUTH_DIAGNOSTIC", JSON.stringify({configured: Boolean(WORKER_TOKEN), length: WORKER_TOKEN.length, fingerprint: createHash("sha256").update(WORKER_TOKEN).digest("hex").slice(0, 12)}));

const CLIENT_TOKEN = process.env.DREAM_HOST_CLIENT_TOKEN || "";
const NOTEBOOK_TOKEN = normalizeToken(process.env.DREAM_HOST_NOTEBOOK_TOKEN);
const NOTEBOOK_WORKER_TOKEN = normalizeToken(process.env.DREAM_HOST_NOTEBOOK_WORKER_TOKEN);
const ARTIFACT_TOKEN = normalizeToken(process.env.DREAM_HOST_ARTIFACT_TOKEN);
const ARTIFACTS_DIR = String(process.env.DREAM_HOST_ARTIFACTS_DIR || "").trim();
const R2_ACCOUNT_ID = String(process.env.DREAM_R2_ACCOUNT_ID || "").trim();
const R2_ACCESS_KEY_ID = String(process.env.DREAM_R2_ACCESS_KEY_ID || "").trim();
const R2_SECRET_ACCESS_KEY = String(process.env.DREAM_R2_SECRET_ACCESS_KEY || "").trim();
const R2_BUCKET = String(process.env.DREAM_R2_BUCKET || "").trim();
const R2_CONFIGURED = /^[a-f0-9]{32}$/i.test(R2_ACCOUNT_ID)
  && R2_ACCESS_KEY_ID.length > 0
  && R2_SECRET_ACCESS_KEY.length > 0
  && /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(R2_BUCKET);
const ARTIFACT_MAX_BYTES = finiteNumber(process.env.DREAM_HOST_ARTIFACT_MAX_BYTES, 512 * 1024 * 1024, 1024, 1024 * 1024 * 1024);
const NOTEBOOK_TOKEN_CONFIGURED = NOTEBOOK_TOKEN.length >= 32
  && NOTEBOOK_TOKEN !== WORKER_TOKEN
  && NOTEBOOK_TOKEN !== normalizeToken(CLIENT_TOKEN);
const NOTEBOOK_WORKER_TOKEN_CONFIGURED = NOTEBOOK_WORKER_TOKEN.length >= 32
  && NOTEBOOK_WORKER_TOKEN !== WORKER_TOKEN
  && NOTEBOOK_WORKER_TOKEN !== normalizeToken(CLIENT_TOKEN)
  && NOTEBOOK_WORKER_TOKEN !== NOTEBOOK_TOKEN;
const ARTIFACT_R2_CLIENT = R2_CONFIGURED
  ? new S3Client({
      region: "auto",
      endpoint: `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
      credentials: {
        accessKeyId: R2_ACCESS_KEY_ID,
        secretAccessKey: R2_SECRET_ACCESS_KEY
      },
      maxAttempts: 3,
      requestChecksumCalculation: "WHEN_REQUIRED",
      responseChecksumValidation: "WHEN_REQUIRED"
    })
  : null;
const ARTIFACT_STORE = R2_CONFIGURED
  ? createR2ArtifactStore({
      client: ARTIFACT_R2_CLIENT,
      bucket: R2_BUCKET,
      commands: {
        CopyObjectCommand,
        DeleteObjectCommand,
        GetObjectCommand,
        ListObjectsV2Command,
        PutObjectCommand
      }
    })
  : (ARTIFACTS_DIR && path.isAbsolute(ARTIFACTS_DIR)
      ? {
          putArtifact: ({ name, source, maxBytes }) => putArtifact({ root: ARTIFACTS_DIR, name, source, maxBytes }),
          listArtifacts: () => listArtifacts(ARTIFACTS_DIR),
          getArtifact: sha256 => getArtifact(ARTIFACTS_DIR, sha256)
        }
      : null);
const ARTIFACT_STORAGE_CONFIGURED = Boolean(ARTIFACT_STORE)
  && ARTIFACT_TOKEN.length >= 32
  && ![WORKER_TOKEN, normalizeToken(CLIENT_TOKEN), NOTEBOOK_TOKEN, NOTEBOOK_WORKER_TOKEN].includes(ARTIFACT_TOKEN);
const NOTEBOOK_STATE_STORE = ARTIFACT_STORAGE_CONFIGURED && R2_CONFIGURED
  ? createR2StateStore({
      client: ARTIFACT_R2_CLIENT,
      bucket: R2_BUCKET,
      prefix: "dream-ai-state/",
      commands: { GetObjectCommand, PutObjectCommand, ListObjectsV2Command, DeleteObjectCommand }
    })
  : null;
const NOTEBOOK_QUEUE_STATE_KEY = "notebook/queue.json";
const NOTEBOOK_QUEUE_STATE_MAX_BYTES = 4 * 1024 * 1024;
const NOTEBOOK_STORAGE_MAX_BYTES = 900 * 1024;
const NOTEBOOK_QUEUE = new NotebookQueue({
  maxQueue: process.env.DREAM_HOST_NOTEBOOK_MAX_QUEUE,
  maxCodeChars: process.env.DREAM_HOST_NOTEBOOK_MAX_CODE_CHARS,
  maxOutputChars: process.env.DREAM_HOST_NOTEBOOK_MAX_OUTPUT_CHARS,
  workerStaleMs: process.env.DREAM_HOST_NOTEBOOK_WORKER_STALE_MS,
  jobLeaseMs: process.env.DREAM_HOST_NOTEBOOK_JOB_LEASE_MS,
  maxExecutionMs: process.env.DREAM_HOST_NOTEBOOK_EXECUTION_TIMEOUT_MS
});
let notebookMutationChain = Promise.resolve();

async function persistNotebookQueue() {
  if (!NOTEBOOK_STATE_STORE) {
    const error = new Error("Permanent notebook storage is not configured.");
    error.status = 503;
    throw error;
  }
  await NOTEBOOK_STATE_STORE.writeJson(NOTEBOOK_QUEUE_STATE_KEY, NOTEBOOK_QUEUE.snapshot(), NOTEBOOK_QUEUE_STATE_MAX_BYTES);
}

function mutateNotebookQueue(operation) {
  const run = notebookMutationChain.catch(() => {}).then(async () => {
    const before = NOTEBOOK_QUEUE.snapshot();
    const value = operation();
    try {
      await persistNotebookQueue();
      return value;
    } catch (error) {
      NOTEBOOK_QUEUE.restore(before);
      error.status = Number.isInteger(error.status) ? error.status : 503;
      throw error;
    }
  });
  notebookMutationChain = run.catch(() => {});
  return run;
}

async function restoreNotebookQueue() {
  if (!NOTEBOOK_STATE_STORE) {
    console.error("DREAM_NOTEBOOK_PERSISTENCE_DISABLED", JSON.stringify({ provider: "none" }));
    return;
  }
  const saved = await NOTEBOOK_STATE_STORE.readJson(NOTEBOOK_QUEUE_STATE_KEY, NOTEBOOK_QUEUE_STATE_MAX_BYTES);
  if (saved) {
    const result = NOTEBOOK_QUEUE.restore(saved);
    await persistNotebookQueue();
    console.log("DREAM_NOTEBOOK_PERSISTENCE_RESTORED", JSON.stringify({
      provider: "cloudflare-r2",
      jobs: result.restoredJobs,
      pending: result.pending
    }));
  } else {
    await persistNotebookQueue();
    console.log("DREAM_NOTEBOOK_PERSISTENCE_INITIALIZED", JSON.stringify({ provider: "cloudflare-r2" }));
  }
}

function validateNotebookDocument(input, expectedId) {
  if (!input || typeof input !== "object" || Array.isArray(input) || input.id !== expectedId) {
    const error = new Error("Notebook document is invalid.");
    error.status = 400;
    throw error;
  }
  const title = typeof input.title === "string" ? input.title.trim().slice(0, 120) : "";
  if (!title || !Array.isArray(input.cells) || input.cells.length < 1 || input.cells.length > 100) {
    const error = new Error("Notebook needs a title and between 1 and 100 cells.");
    error.status = 400;
    throw error;
  }
  const cells = input.cells.map(cell => {
    if (!cell || typeof cell !== "object" || !["code", "markdown"].includes(cell.type)
      || typeof cell.id !== "string" || !/^[A-Za-z0-9_-]{1,100}$/.test(cell.id)
      || typeof cell.source !== "string" || cell.source.length > 20000) {
      const error = new Error("Notebook contains an invalid cell.");
      error.status = 400;
      throw error;
    }
    const output = cell.output && typeof cell.output === "object" ? cell.output : {};
    for (const key of ["stdout", "stderr", "display"]) {
      if (output[key] !== undefined && (typeof output[key] !== "string" || output[key].length > 120000)) {
        const error = new Error("Notebook cell output exceeds the safe storage limit.");
        error.status = 413;
        throw error;
      }
    }
    return {
      id: cell.id,
      type: cell.type,
      source: cell.source,
      executionCount: Number.isInteger(cell.executionCount) && cell.executionCount >= 0 ? cell.executionCount : null,
      output: {
        stdout: String(output.stdout || ""),
        stderr: String(output.stderr || ""),
        display: String(output.display || "")
      },
      status: ["idle", "completed", "failed"].includes(cell.status) ? cell.status : "idle",
      error: typeof cell.error === "string" ? cell.error.slice(0, 2000) : ""
    };
  });
  const document = {
    schema: "dream-notebook-v1",
    id: expectedId,
    title,
    cells,
    modifiedAt: new Date().toISOString()
  };
  if (Buffer.byteLength(JSON.stringify(document)) > NOTEBOOK_STORAGE_MAX_BYTES) {
    const error = new Error("Notebook is too large to save to permanent cloud storage.");
    error.status = 413;
    throw error;
  }
  return document;
}

async function listPersistentNotebooks() {
  if (!NOTEBOOK_STATE_STORE) {
    const error = new Error("Permanent notebook storage is not configured.");
    error.status = 503;
    throw error;
  }
  const keys = await NOTEBOOK_STATE_STORE.list("notebooks/", 50);
  const documents = await Promise.all(keys.map(key => NOTEBOOK_STATE_STORE.readJson(key, NOTEBOOK_STORAGE_MAX_BYTES)));
  return documents.filter(item => item?.schema === "dream-notebook-v1" && typeof item.id === "string")
    .sort((a, b) => String(b.modifiedAt || "").localeCompare(String(a.modifiedAt || "")))
    .map(item => ({ id: item.id, title: item.title, modifiedAt: item.modifiedAt }));
}

async function readPersistentNotebook(notebookId) {
  if (!NOTEBOOK_STATE_STORE) {
    const error = new Error("Permanent notebook storage is not configured.");
    error.status = 503;
    throw error;
  }
  return NOTEBOOK_STATE_STORE.readJson("notebooks/" + notebookId + ".json", NOTEBOOK_STORAGE_MAX_BYTES);
}

async function writePersistentNotebook(notebookId, body) {
  if (!NOTEBOOK_STATE_STORE) {
    const error = new Error("Permanent notebook storage is not configured.");
    error.status = 503;
    throw error;
  }
  const document = validateNotebookDocument(body, notebookId);
  await NOTEBOOK_STATE_STORE.writeJson("notebooks/" + notebookId + ".json", document, NOTEBOOK_STORAGE_MAX_BYTES);
  return document;
}
let NOTEBOOK_CHAT_MODEL_READY = false;
let NOTEBOOK_CHAT_MODEL_DEVICE = null;
let NOTEBOOK_CHAT_MODEL_ERROR = "";
let NOTEBOOK_CHAT_WARMUP_JOB_ID = null;
const NOTEBOOK_CHAT_RUNTIME_ID = "dream-ai-chat-runtime";
const NOTEBOOK_CHAT_WARMUP_ENABLED = /^(1|true|yes)$/i.test(String(process.env.DREAM_HOST_NOTEBOOK_CHAT_WARMUP || "").trim());

const MAX_BODY_BYTES = 1024 * 1024;
const rawQueueLimit = Number(process.env.DREAM_HOST_MAX_QUEUE || 30);
const MAX_QUEUE = Number.isFinite(rawQueueLimit) ? Math.max(1, Math.min(rawQueueLimit, 200)) : 30;
const rawTimeout = Number(process.env.DREAM_HOST_JOB_TIMEOUT_MS || 120000);
const JOB_TIMEOUT_MS = Number.isFinite(rawTimeout) ? Math.max(15000, Math.min(rawTimeout, 300000)) : 120000;
const rawStale = Number(process.env.DREAM_HOST_WORKER_STALE_MS || 60000);
const WORKER_STALE_MS = Number.isFinite(rawStale) ? Math.max(30000, Math.min(rawStale, 180000)) : 60000;
const MODEL_NAME = process.env.DREAM_HOST_MODEL_NAME || "Qwen/Qwen3-1.7B";
const ADAPTER_NAME = process.env.DREAM_HOST_ADAPTER_NAME || "msjnr0477/dream-ai-adapter";
const HOST_VERSION = "0.3.0";
const STARTED_AT = Date.now();

if (WORKER_TOKEN.length < 32 || CLIENT_TOKEN.length < 32) {
  throw new Error("DREAM_HOST_WORKER_TOKEN and DREAM_HOST_CLIENT_TOKEN must each contain at least 32 characters.");
}

const jobs = new Map();
const workers = new Map();
const pending = [];
const waiters = new Map();

const securityHeaders = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
  "x-frame-options": "DENY",
  "strict-transport-security": "max-age=31536000; includeSubDomains"
};

function sendJson(res, status, body, extra = {}) {
  if (res.writableEnded || res.destroyed) return;
  res.writeHead(status, { ...securityHeaders, ...extra });
  res.end(JSON.stringify(body));
}

function sendEmpty(res, status = 204) {
  if (res.writableEnded || res.destroyed) return;
  res.writeHead(status, {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff"
  });
  res.end();
}

async function sendStaticFile(res, fileName, contentType) {
  const content = await readFile(new URL(fileName, import.meta.url));
  res.writeHead(200, {
    ...securityHeaders,
    "content-type": contentType,
    "cache-control": "no-cache",
    "content-security-policy": "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:"
  });
  res.end(content);
}

function tokenMatches(req, expected) {
  const header = String(req.headers.authorization || "").trim();
  if (header.slice(0, 7).toLowerCase() !== "bearer ") return false;
  const supplied = Buffer.from(normalizeToken(header.slice(7)));
  const wanted = Buffer.from(expected);
  return supplied.length === wanted.length && timingSafeEqual(supplied, wanted);
}

function notebookCookieMatches(req) {
  const cookies = String(req.headers.cookie || "").split(";");
  const entry = cookies.map(value => value.trim()).find(value => value.startsWith("dream_notebook_session="));
  if (!entry) return false;
  let value = "";
  try { value = decodeURIComponent(entry.slice("dream_notebook_session=".length)); } catch { return false; }
  const supplied = Buffer.from(value);
  const wanted = Buffer.from(NOTEBOOK_TOKEN);
  return supplied.length === wanted.length && timingSafeEqual(supplied, wanted);
}

function requireToken(req, res, expected) {
  if (tokenMatches(req, expected)) return true;
  sendJson(res, 401, { error: { message: "Unauthorized", type: "authentication_error" } });
  return false;
}

function requireNotebookToken(req, res) {
  if (!NOTEBOOK_TOKEN_CONFIGURED || !NOTEBOOK_WORKER_TOKEN_CONFIGURED) {
    sendJson(res, 503, {
      error: {
        message: "DREAM NOTEBOOK is not configured. Set DREAM_HOST_NOTEBOOK_TOKEN and DREAM_HOST_NOTEBOOK_WORKER_TOKEN to distinct private values of at least 32 characters.",
        type: "configuration_error"
      }
    });
    return false;
  }
  if (tokenMatches(req, NOTEBOOK_TOKEN) || notebookCookieMatches(req)) return true;
  sendJson(res, 401, { error: { message: "Unauthorized", type: "authentication_error" } });
  return false;
}

function requireNotebookWorkerToken(req, res) {
  if (!NOTEBOOK_TOKEN_CONFIGURED || !NOTEBOOK_WORKER_TOKEN_CONFIGURED) {
    sendJson(res, 503, {
      error: {
        message: "DREAM NOTEBOOK worker authentication is not configured. Set both private notebook tokens on DREAM HOST.",
        type: "configuration_error"
      }
    });
    return false;
  }
  return requireToken(req, res, NOTEBOOK_WORKER_TOKEN);
}

function finiteNumber(value, fallback, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(parsed, maximum));
}

async function readJson(req) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      const error = new Error("Request body exceeds 1 MiB.");
      error.status = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("JSON body must be an object.");
    }
    return value;
  } catch (error) {
    if (error.status) throw error;
    const invalid = new Error("Request body must be a JSON object.");
    invalid.status = 400;
    throw invalid;
  }
}

function newWorker(workerId, name, metadata = {}) {
  return {
    id: workerId,
    name: typeof name === "string" && name.trim() ? name.trim().slice(0, 80) : "Kaggle GPU worker",
    metadata: metadata && typeof metadata === "object" && !Array.isArray(metadata) ? metadata : {},
    connectedAt: new Date().toISOString(),
    lastSeen: Date.now(),
    status: "idle",
    busyJobId: null
  };
}

function resolveJob(job, result) {
  const waiter = waiters.get(job.id);
  if (waiter) {
    clearTimeout(waiter.timer);
    waiters.delete(job.id);
    waiter.resolve(result);
  }
}

function removePending(jobId) {
  const index = pending.indexOf(jobId);
  if (index >= 0) pending.splice(index, 1);
}

function sweep() {
  const now = Date.now();
  for (const [workerId, worker] of workers) {
    if (now - worker.lastSeen <= WORKER_STALE_MS) continue;
    workers.delete(workerId);
    if (!worker.busyJobId) continue;
    const job = jobs.get(worker.busyJobId);
    if (!job || job.status !== "running") continue;
    job.workerId = null;
    job.updatedAt = new Date().toISOString();
    if (job.attempts < 3 && Date.now() < job.deadlineAt) {
      job.status = "queued";
      pending.unshift(job.id);
    } else {
      job.status = "failed";
      job.error = "The Kaggle worker disconnected before completing this request.";
      resolveJob(job, {
        status: 503,
        body: { error: { message: job.error, type: "server_error" } }
      });
    }
  }
  for (const [jobId, job] of jobs) {
    const terminal = ["completed", "failed", "timed_out", "cancelled"].includes(job.status);
    const age = now - Date.parse(job.updatedAt || job.createdAt);
    if (terminal && age > 15 * 60 * 1000) jobs.delete(jobId);
  }
}

function connectedWorkers() {
  sweep();
  const now = Date.now();
  return [...workers.values()].filter(worker => now - worker.lastSeen <= WORKER_STALE_MS);
}

function readyWorkers() {
  return connectedWorkers().filter(worker =>
    worker.metadata?.model === MODEL_NAME &&
    worker.metadata?.adapter === ADAPTER_NAME
  );
}

function statusSnapshot() {
  const connected = connectedWorkers();
  const ready = connected.filter(worker =>
    worker.metadata?.model === MODEL_NAME &&
    worker.metadata?.adapter === ADAPTER_NAME
  );
  const notebook = NOTEBOOK_QUEUE.status();
  const notebookComputeOnline = notebook.workers.connected > 0;
  const activeWorker = ready.find(worker => typeof worker.metadata?.device === "string");
  const notebookDevice = notebook.workers.devices.find(item => typeof item.device === "string")?.device || null;
  const modelReady = ready.length > 0 || (NOTEBOOK_CHAT_WARMUP_ENABLED && NOTEBOOK_CHAT_MODEL_READY && notebookComputeOnline);
  const warmup = NOTEBOOK_CHAT_WARMUP_JOB_ID
    ? NOTEBOOK_QUEUE.getJob(NOTEBOOK_CHAT_WARMUP_JOB_ID)
    : null;
  const modelLoading = Boolean(warmup && ["queued", "running"].includes(warmup.status));
  return {
    name: "DREAM HOST",
    version: HOST_VERSION,
    ok: true,
    modelConfigured: true,
    modelReady,
    chatAvailable: ready.length > 0 || (NOTEBOOK_CHAT_WARMUP_ENABLED && notebookComputeOnline),
    notebookChatWarmupEnabled: NOTEBOOK_CHAT_WARMUP_ENABLED,
    modelLoading,
    modelError: NOTEBOOK_CHAT_MODEL_ERROR || null,
    model: MODEL_NAME,
    adapter: ADAPTER_NAME,
    modelRuntime: ready.length > 0 ? "gpu-worker" : (NOTEBOOK_CHAT_WARMUP_ENABLED && notebookComputeOnline ? "dream-notebook" : "offline"),
    device: activeWorker?.metadata?.device || (NOTEBOOK_CHAT_WARMUP_ENABLED && NOTEBOOK_CHAT_MODEL_READY ? NOTEBOOK_CHAT_MODEL_DEVICE : null) || (ready.length > 0 ? notebookDevice : null),
    uptimeSeconds: Math.max(0, Math.floor((Date.now() - STARTED_AT) / 1000)),
    persistence: "in-memory",
    notebookPersistence: NOTEBOOK_STATE_STORE ? "cloudflare-r2" : "unavailable",
    artifactStorage: ARTIFACT_STORAGE_CONFIGURED ? (R2_CONFIGURED ? "r2" : "filesystem") : "disabled",
    queue: {
      pending: pending.filter(id => jobs.get(id)?.status === "queued").length,
      running: [...jobs.values()].filter(job => job.status === "running").length,
      limit: MAX_QUEUE
    },
    workers: {
      connected: connected.length,
      ready: ready.length,
      idle: ready.filter(worker => worker.status === "idle").length,
      busy: ready.filter(worker => worker.status === "busy").length
    }
  };
}

function openAiError(message, type = "server_error") {
  return { error: { message, type } };
}

function notebookModelBootstrapCode() {
  return [
    "import json",
    "import os",
    "if '_dream_chat_model' not in globals():",
    "    import torch",
    "    if not torch.cuda.is_available(): raise RuntimeError('DREAM AI chat requires a CUDA GPU in the connected notebook runtime.')",
    "    import importlib.metadata as _dream_metadata",
    "    import subprocess as _dream_subprocess",
    "    import sys as _dream_sys",
    "    try:",
    "        _dream_torchao_version = _dream_metadata.version('torchao')",
    "        _dream_torchao_parts = tuple(int(part) for part in _dream_torchao_version.split('+')[0].split('.')[:3])",
    "    except _dream_metadata.PackageNotFoundError:",
    "        _dream_torchao_parts = ()",
    "    if _dream_torchao_parts and _dream_torchao_parts <= (0, 16, 0):",
    "        _dream_subprocess.run([_dream_sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True, stdout=_dream_subprocess.DEVNULL, stderr=_dream_subprocess.DEVNULL)",
    "    from transformers import AutoTokenizer, AutoModelForCausalLM",
    "    from peft import PeftModel",
    "    _dream_hf_token = os.environ.get('DREAM_NOTEBOOK_HF_TOKEN') or None",
    "    _dream_chat_tokenizer = AutoTokenizer.from_pretrained(" + JSON.stringify(ADAPTER_NAME) + ", token=_dream_hf_token)",
    "    _dream_chat_base = AutoModelForCausalLM.from_pretrained(" + JSON.stringify(MODEL_NAME) + ", torch_dtype=torch.float16, low_cpu_mem_usage=True, token=_dream_hf_token)",
    "    _dream_chat_model = PeftModel.from_pretrained(_dream_chat_base, " + JSON.stringify(ADAPTER_NAME) + ", token=_dream_hf_token, torch_device='cpu')",
    "    _dream_chat_model = _dream_chat_model.to('cuda')",
    "    _dream_chat_model.eval()",
    "    _dream_chat_device = torch.cuda.get_device_name(0)",
    "else:",
    "    import torch",
    "    _dream_chat_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'",
    "print('DREAM_CHAT_MODEL_READY:' + str(_dream_chat_device))"
  ].join("\n");
}

function buildNotebookChatCode(payload) {
  const requestLiteral = JSON.stringify(JSON.stringify(payload));
  return [
    notebookModelBootstrapCode(),
    "_dream_request = json.loads(" + requestLiteral + ")",
    "_dream_prompt = _dream_chat_tokenizer.apply_chat_template(_dream_request['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)",
    "_dream_inputs = _dream_chat_tokenizer(_dream_prompt, return_tensors='pt', truncation=True, max_length=6144)",
    "_dream_inputs = {key: value.to('cuda') for key, value in _dream_inputs.items()}",
    "_dream_options = {'max_new_tokens': max(1, min(int(_dream_request['max_tokens']), 1024)), 'pad_token_id': _dream_chat_tokenizer.eos_token_id, 'repetition_penalty': 1.05}",
    "if float(_dream_request['temperature']) <= 0:",
    "    _dream_options['do_sample'] = False",
    "else:",
    "    _dream_options.update({'do_sample': True, 'temperature': float(_dream_request['temperature']), 'top_p': float(_dream_request['top_p'])})",
    "with torch.inference_mode():",
    "    _dream_output = _dream_chat_model.generate(**_dream_inputs, **_dream_options)",
    "_dream_prompt_tokens = int(_dream_inputs['input_ids'].shape[-1])",
    "_dream_generated = _dream_output[0][_dream_prompt_tokens:]",
    "_dream_content = _dream_chat_tokenizer.decode(_dream_generated, skip_special_tokens=True).strip()",
    "if not _dream_content: raise RuntimeError('The saved DREAM AI model returned an empty answer.')",
    "_dream_result = {'content': _dream_content, 'promptTokens': _dream_prompt_tokens, 'completionTokens': int(_dream_generated.shape[-1]), 'device': _dream_chat_device}",
    "print('DREAM_CHAT_RESULT:' + json.dumps(_dream_result, ensure_ascii=False))"
  ].join("\n");
}

async function scheduleNotebookChatWarmup() {
  if (!NOTEBOOK_CHAT_WARMUP_ENABLED) return;
  if (NOTEBOOK_CHAT_MODEL_READY || NOTEBOOK_CHAT_WARMUP_JOB_ID || readyWorkers().length > 0) return;
  if (NOTEBOOK_QUEUE.status().workers.connected < 1) return;
  try {
    const warmup = await mutateNotebookQueue(() => NOTEBOOK_QUEUE.submit({
      notebookId: NOTEBOOK_CHAT_RUNTIME_ID,
      cellId: "warmup_" + randomUUID().replaceAll("-", ""),
      code: notebookModelBootstrapCode()
    }));
    NOTEBOOK_CHAT_WARMUP_JOB_ID = warmup.id;
    NOTEBOOK_CHAT_MODEL_ERROR = "";
    console.log("DREAM_NOTEBOOK_CHAT_WARMUP_QUEUED", JSON.stringify({ jobId: warmup.id }));
  } catch (error) {
    NOTEBOOK_CHAT_MODEL_ERROR = "DREAM AI model warm-up could not be queued.";
    console.warn("DREAM_NOTEBOOK_CHAT_WARMUP_REJECTED", error.message);
  }
}

function sendNotebookChatCompletion(res, model, result) {
  NOTEBOOK_CHAT_MODEL_READY = true;
  NOTEBOOK_CHAT_MODEL_ERROR = "";
  NOTEBOOK_CHAT_MODEL_DEVICE = typeof result.device === "string" ? result.device : NOTEBOOK_CHAT_MODEL_DEVICE;
  return sendJson(res, 200, {
    id: "chatcmpl_" + randomUUID().replaceAll("-", ""),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message: { role: "assistant", content: result.content }, finish_reason: "stop" }],
    usage: {
      prompt_tokens: Math.max(0, Number(result.promptTokens) || 0),
      completion_tokens: Math.max(0, Number(result.completionTokens) || 0),
      total_tokens: Math.max(0, Number(result.promptTokens) || 0) + Math.max(0, Number(result.completionTokens) || 0)
    }
  });
}

async function handleNotebookChat(res, model, messages, maxTokens, temperature, topP) {
  if (NOTEBOOK_QUEUE.status().workers.connected < 1) {
    return sendJson(res, 503, openAiError("DREAM NOTEBOOK GPU compute is offline. Connect the DREAM AI + notebook GPU runtime.", "server_error"), { "retry-after": "10" });
  }
  const payload = { messages, max_tokens: maxTokens, temperature, top_p: topP };
  let submitted;
  try {
    submitted = await mutateNotebookQueue(() => NOTEBOOK_QUEUE.submit({
      notebookId: NOTEBOOK_CHAT_RUNTIME_ID,
      cellId: "chat_" + randomUUID().replaceAll("-", ""),
      code: buildNotebookChatCode(payload)
    }));
  } catch (error) {
    const status = Number.isInteger(error.status) ? error.status : 500;
    return sendJson(res, status, openAiError(status >= 500 ? "DREAM NOTEBOOK could not queue the AI request." : error.message, status === 429 ? "rate_limit_error" : "server_error"));
  }

  const deadline = Date.now() + JOB_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (res.destroyed || res.writableEnded) return;
    const job = NOTEBOOK_QUEUE.getJob(submitted.id);
    if (!job) {
      return sendJson(res, 503, openAiError("DREAM NOTEBOOK lost the queued AI request. Please retry.", "server_error"), { "retry-after": "5" });
    }
    if (job.status === "failed") {
      return sendJson(res, 502, openAiError("DREAM NOTEBOOK could not run the saved DREAM AI model. Check the GPU runtime status and retry.", "server_error"));
    }
    if (job.status === "completed") {
      const marker = String(job.result?.stdout || "").split(/\r?\n/).reverse().find(line => line.startsWith("DREAM_CHAT_RESULT:"));
      if (!marker) {
        NOTEBOOK_CHAT_MODEL_ERROR = "DREAM AI inference did not return a valid result.";
        return sendJson(res, 502, openAiError("The saved DREAM AI model did not return a valid answer. Check the notebook GPU runtime and retry.", "server_error"));
      }
      try {
        const result = JSON.parse(marker.slice("DREAM_CHAT_RESULT:".length));
        if (typeof result.content !== "string" || !result.content.trim()) throw new Error("Empty model response.");
        return sendNotebookChatCompletion(res, model, result);
      } catch {
        NOTEBOOK_CHAT_MODEL_ERROR = "DREAM AI inference returned an invalid result.";
        return sendJson(res, 502, openAiError("The saved DREAM AI model returned an invalid answer. Please retry.", "server_error"));
      }
    }
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  return sendJson(res, 504, openAiError("DREAM AI did not finish the notebook GPU request before the timeout. Please retry.", "timeout_error"), { "retry-after": "5" });
}

async function handleChat(req, res) {
  if (!requireToken(req, res, CLIENT_TOKEN)) return;
  const body = await readJson(req);
  if (body.stream === true) {
    return sendJson(res, 400, openAiError("Streaming is not supported by this first DREAM HOST release.", "invalid_request_error"));
  }
  if (!Array.isArray(body.messages) || body.messages.length < 1 || body.messages.length > 64) {
    return sendJson(res, 400, openAiError("messages must be an array containing 1 to 64 entries.", "invalid_request_error"));
  }
  if (body.model !== undefined && body.model !== MODEL_NAME) {
    return sendJson(res, 400, openAiError("Requested model does not match DREAM HOST configuration.", "invalid_request_error"));
  }
  const messages = body.messages.map(message => {
    if (!message || typeof message !== "object" || !["system", "user", "assistant"].includes(message.role) ||
        typeof message.content !== "string" || message.content.length > 12000) return null;
    return { role: message.role, content: message.content };
  });
  if (messages.some(message => message === null)) {
    return sendJson(res, 400, openAiError("Each message must have a supported role and text content up to 12,000 characters.", "invalid_request_error"));
  }
  sweep();
  const availableWorkers = readyWorkers();
  const maxTokens = finiteNumber(body.max_tokens ?? body.max_completion_tokens, 512, 1, 1024);
  const temperature = finiteNumber(body.temperature, 0.7, 0, 2);
  const topP = finiteNumber(body.top_p, 0.9, 0.01, 1);
  if (availableWorkers.length === 0) {
    if (NOTEBOOK_QUEUE.status().workers.connected > 0 && NOTEBOOK_CHAT_WARMUP_ENABLED) {
      return await handleNotebookChat(res, MODEL_NAME, messages, maxTokens, temperature, topP);
    }
    return sendJson(
      res,
      503,
      openAiError(
        NOTEBOOK_QUEUE.status().workers.connected > 0
          ? "DREAM NOTEBOOK is connected for training and notebook execution. A native DREAM inference worker is not active yet."
          : "No DREAM inference GPU worker is connected.",
        "server_error"
      ),
      { "retry-after": "10" }
    );
  }
  const queuedCount = pending.filter(id => jobs.get(id)?.status === "queued").length;
  if (queuedCount >= MAX_QUEUE) {
    return sendJson(res, 429, openAiError("DREAM HOST queue is full. Retry shortly.", "rate_limit_error"), { "retry-after": "5" });
  }

  const id = "dreamjob_" + randomUUID().replaceAll("-", "");
  const now = new Date().toISOString();
  const job = {
    id,
    model: typeof body.model === "string" && body.model.trim() ? body.model.trim().slice(0, 120) : MODEL_NAME,
    messages,
    maxTokens,
    temperature,
    topP,
    status: "queued",
    workerId: null,
    attempts: 0,
    createdAt: now,
    updatedAt: now,
    deadlineAt: Date.now() + JOB_TIMEOUT_MS
  };
  jobs.set(id, job);
  pending.push(id);

  const result = await new Promise(resolve => {
    const timer = setTimeout(() => {
      waiters.delete(job.id);
      job.status = "timed_out";
      job.updatedAt = new Date().toISOString();
      removePending(job.id);
      resolve({ status: 504, body: openAiError("No GPU worker completed this request before the timeout.", "timeout_error") });
    }, JOB_TIMEOUT_MS);
    waiters.set(job.id, { resolve, timer });
    res.on("close", () => {
      if (res.writableEnded) return;
      const waiter = waiters.get(job.id);
      if (!waiter) return;
      clearTimeout(waiter.timer);
      waiters.delete(job.id);
      removePending(job.id);
      if (job.status === "queued") {
        job.status = "cancelled";
        job.updatedAt = new Date().toISOString();
      }
    });
  });
  sendJson(res, result.status, result.body);
}

async function handleWorkerRoute(req, res, url) {
  if (!tokenMatches(req, WORKER_TOKEN)) {
    const authorization = String(req.headers.authorization || "");
    const hasBearerScheme = authorization.slice(0, 7).toLowerCase() === "bearer ";
    const supplied = hasBearerScheme ? normalizeToken(authorization.slice(7)) : "";
    console.warn("DREAM_HOST_WORKER_AUTH_REJECTED", JSON.stringify({
      path: url.pathname,
      authorizationPresent: Boolean(authorization),
      bearerScheme: hasBearerScheme,
      suppliedLength: supplied.length,
      suppliedFingerprint: createHash("sha256").update(supplied).digest("hex").slice(0, 12),
      expectedLength: WORKER_TOKEN.length,
      expectedFingerprint: createHash("sha256").update(WORKER_TOKEN).digest("hex").slice(0, 12)
    }));
    return sendJson(res, 401, { error: { message: "Unauthorized", type: "authentication_error" } });
  }
  sweep();

  if (req.method === "POST" && url.pathname === "/api/v1/workers/register") {
    const body = await readJson(req);
    const workerId = typeof body.workerId === "string" ? body.workerId.trim() : "";
    if (!/^[A-Za-z0-9_-]{6,120}$/.test(workerId)) {
      return sendJson(res, 400, { error: "workerId must contain 6 to 120 letters, numbers, underscores or hyphens." });
    }
    const existing = workers.get(workerId);
    const worker = existing || newWorker(workerId, body.name, body.metadata);
    worker.lastSeen = Date.now();
    worker.name = typeof body.name === "string" && body.name.trim() ? body.name.trim().slice(0, 80) : worker.name;
    worker.metadata = body.metadata && typeof body.metadata === "object" && !Array.isArray(body.metadata) ? body.metadata : worker.metadata;
    workers.set(workerId, worker);
    return sendJson(res, 200, { ok: true, workerId, heartbeatIntervalSeconds: 15 });
  }

  if (req.method === "POST" && url.pathname === "/api/v1/workers/heartbeat") {
    const body = await readJson(req);
    const workerId = typeof body.workerId === "string" ? body.workerId.trim() : "";
    if (!workerId) return sendJson(res, 400, { error: "workerId is required." });
    let worker = workers.get(workerId);
    if (!worker) {
      worker = newWorker(workerId, body.name, body.metadata);
      workers.set(workerId, worker);
    }
    worker.lastSeen = Date.now();
    worker.status = body.busy ? "busy" : (worker.busyJobId ? "busy" : "idle");
    if (worker.busyJobId) {
      const running = jobs.get(worker.busyJobId);
      if (running?.status === "running") running.updatedAt = new Date().toISOString();
    }
    return sendJson(res, 200, { ok: true, workerId, serverTime: new Date().toISOString() });
  }

  if (req.method === "POST" && url.pathname === "/api/v1/workers/jobs/claim") {
    const body = await readJson(req);
    const workerId = typeof body.workerId === "string" ? body.workerId.trim() : "";
    const worker = workers.get(workerId);
    if (!worker) return sendJson(res, 409, { error: "Worker is not registered. Register it again." });
    if (worker.metadata?.model !== MODEL_NAME || worker.metadata?.adapter !== ADAPTER_NAME) {
      return sendJson(res, 409, { error: "Worker model or adapter does not match DREAM HOST configuration." });
    }
    if (worker.busyJobId) return sendJson(res, 409, { error: "Worker already has an active job." });
    worker.lastSeen = Date.now();
    while (pending.length) {
      const id = pending.shift();
      const job = jobs.get(id);
      if (!job || job.status !== "queued") continue;
      job.status = "running";
      job.workerId = workerId;
      job.attempts += 1;
      job.updatedAt = new Date().toISOString();
      worker.status = "busy";
      worker.busyJobId = job.id;
      return sendJson(res, 200, {
        ok: true,
        job: {
          id: job.id,
          model: job.model,
          messages: job.messages,
          max_tokens: job.maxTokens,
          temperature: job.temperature,
          top_p: job.topP
        }
      });
    }
    worker.status = "idle";
    return sendEmpty(res, 204);
  }

  const resultMatch = url.pathname.match(/^\/api\/v1\/workers\/jobs\/([A-Za-z0-9_-]+)\/(result|fail)$/);
  if (req.method === "POST" && resultMatch) {
    const [, jobId, action] = resultMatch;
    const body = await readJson(req);
    const workerId = typeof body.workerId === "string" ? body.workerId.trim() : "";
    const job = jobs.get(jobId);
    const worker = workers.get(workerId);
    if (!job || job.status !== "running") return sendJson(res, 404, { error: "Active job not found." });
    if (!worker || job.workerId !== workerId || worker.busyJobId !== jobId) {
      return sendJson(res, 403, { error: "This worker does not own the job." });
    }
    if (action === "result" && (typeof body.content !== "string" || !body.content.trim() || body.content.length > 200000)) {
      return sendJson(res, 400, { error: "content must contain 1 to 200,000 characters." });
    }
    worker.lastSeen = Date.now();
    worker.status = "idle";
    worker.busyJobId = null;
    job.updatedAt = new Date().toISOString();
    if (action === "fail") {
      const message = typeof body.error === "string" ? body.error.slice(0, 1000) : "GPU worker failed to process the job.";
      job.status = "failed";
      resolveJob(job, { status: 502, body: openAiError(message) });
      return sendJson(res, 200, { ok: true, jobId, status: "failed" });
    }
    job.status = "completed";
    const promptTokens = Math.max(0, Number(body.promptTokens) || 0);
    const completionTokens = Math.max(0, Number(body.completionTokens) || 0);
    const completion = {
      id: "chatcmpl_" + randomUUID().replaceAll("-", ""),
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: job.model,
      choices: [{ index: 0, message: { role: "assistant", content: body.content }, finish_reason: "stop" }],
      usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens }
    };
    resolveJob(job, { status: 200, body: completion });
    return sendJson(res, 200, { ok: true, jobId, status: "completed" });
  }

  return sendJson(res, 404, { error: "Worker endpoint not found." });
}

async function handleArtifactRoute(req, res, url) {
  if (!ARTIFACT_STORAGE_CONFIGURED) {
    return sendJson(res, 503, { error: { message: "DREAM artifact vault is disabled. Configure its dedicated token and either Cloudflare R2 credentials or an absolute local storage directory.", type: "configuration_error" } });
  }
  if (!tokenMatches(req, ARTIFACT_TOKEN)) {
    return sendJson(res, 401, { error: { message: "Unauthorized", type: "authentication_error" } });
  }
  try {
    if (req.method === "GET" && url.pathname === "/api/v1/notebook/artifacts") {
      return sendJson(res, 200, { ok: true, artifacts: await ARTIFACT_STORE.listArtifacts() });
    }
    const downloadMatch = url.pathname.match(/^\/api\/v1\/notebook\/artifacts\/([a-f0-9]{64})$/);
    if (req.method === "GET" && downloadMatch) {
      const { metadata, stream } = await ARTIFACT_STORE.getArtifact(downloadMatch[1]);
      res.writeHead(200, {
        "content-type": "application/octet-stream",
        "content-length": String(metadata.size_bytes),
        "content-disposition": `attachment; filename="dream-artifact-${metadata.sha256}.bin"`,
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
        "x-artifact-sha256": metadata.sha256
      });
      stream.on("error", () => res.destroy());
      stream.pipe(res);
      return;
    }
    const uploadMatch = url.pathname.match(/^\/api\/v1\/notebook\/artifacts\/([A-Za-z0-9._-]{1,120})$/);
    if ((req.method === "PUT" || req.method === "POST") && uploadMatch) {
      const metadata = await ARTIFACT_STORE.putArtifact({
        name: uploadMatch[1],
        source: req,
        contentLength: req.headers["content-length"],
        maxBytes: ARTIFACT_MAX_BYTES
      });
      return sendJson(res, 201, { ok: true, artifact: metadata });
    }
    return sendJson(res, 404, { error: { message: "Artifact endpoint not found.", type: "not_found" } });
  } catch (error) {
    const status = Number.isInteger(error.status) ? error.status : 500;
    return sendJson(res, status, { error: { message: status >= 500 ? "DREAM artifact vault request failed." : error.message, type: "artifact_error" } });
  }
}

async function handleNotebookClientRoute(req, res, url) {
  if (url.pathname === "/api/v1/notebook/session" && req.method === "POST") {
    if (!NOTEBOOK_TOKEN_CONFIGURED || !NOTEBOOK_WORKER_TOKEN_CONFIGURED) {
      return sendJson(res, 503, { error: { message: "DREAM NOTEBOOK is not configured.", type: "configuration_error" } });
    }
    if (!requireToken(req, res, NOTEBOOK_TOKEN)) return;
    try {
      const body = await readJson(req);
      const remember = body.remember !== false;
      const persistent = remember ? "; Max-Age=2592000" : "";
      const cookie = "dream_notebook_session=" + encodeURIComponent(NOTEBOOK_TOKEN)
        + "; Path=/api/v1/notebook; HttpOnly; Secure; SameSite=Strict" + persistent;
      return sendJson(res, 200, { ok: true, remembered: remember }, { "set-cookie": cookie });
    } catch (error) {
      const status = Number.isInteger(error.status) ? error.status : 400;
      return sendJson(res, status, { error: { message: error.message, type: "session_error" } });
    }
  }
  if (url.pathname === "/api/v1/notebook/session" && req.method === "DELETE") {
    const cookie = "dream_notebook_session=; Path=/api/v1/notebook; HttpOnly; Secure; SameSite=Strict; Max-Age=0";
    return sendJson(res, 200, { ok: true, cleared: true }, { "set-cookie": cookie });
  }
  if (!requireNotebookToken(req, res)) return;
  try {
    if (req.method === "GET" && url.pathname === "/api/v1/notebook/status") {
      return sendJson(res, 200, {
        ...NOTEBOOK_QUEUE.status(),
        persistence: { provider: NOTEBOOK_STATE_STORE ? "cloudflare-r2" : "unavailable" }
      });
    }
    if (req.method === "GET" && url.pathname === "/api/v1/notebook/notebooks") {
      return sendJson(res, 200, { ok: true, notebooks: await listPersistentNotebooks() });
    }
    const notebookMatch = url.pathname.match(/^\\/api\\/v1\\/notebook\\/notebooks\\/([A-Za-z0-9_-]{1,100})$/);
    if (req.method === "GET" && notebookMatch) {
      const document = await readPersistentNotebook(notebookMatch[1]);
      return document
        ? sendJson(res, 200, { ok: true, notebook: document })
        : sendJson(res, 404, { error: { message: "Saved notebook not found.", type: "not_found" } });
    }
    if (req.method === "PUT" && notebookMatch) {
      const body = await readJson(req);
      const document = await writePersistentNotebook(notebookMatch[1], body);
      return sendJson(res, 200, { ok: true, notebook: document, persistence: "cloudflare-r2" });
    }
    if (req.method === "POST" && url.pathname === "/api/v1/notebook/jobs") {
      const body = await readJson(req);
      const job = await mutateNotebookQueue(() => NOTEBOOK_QUEUE.submit(body));
      return sendJson(res, 202, { ok: true, job });
    }
    const match = url.pathname.match(/^\/api\/v1\/notebook\/jobs\/([A-Za-z0-9_-]{1,100})$/);
    if (req.method === "GET" && match) {
      const job = NOTEBOOK_QUEUE.getJob(match[1]);
      return job
        ? sendJson(res, 200, { ok: true, job })
        : sendJson(res, 404, { error: { message: "Notebook job not found.", type: "not_found" } });
    }
    return sendJson(res, 404, { error: { message: "Notebook endpoint not found.", type: "not_found" } });
  } catch (error) {
    const status = Number.isInteger(error.status) ? error.status : 500;
    return sendJson(res, status, {
      error: {
        message: status >= 500 ? "DREAM NOTEBOOK could not process the request." : error.message,
        type: status === 429 ? "rate_limit_error" : "notebook_error"
      }
    });
  }
}

async function handleNotebookWorkerRoute(req, res, url) {
  if (!requireNotebookWorkerToken(req, res)) return;
  try {
    NOTEBOOK_QUEUE.sweep();
    if (req.method === "POST" && url.pathname === "/api/v1/notebook/workers/register") {
      const body = await readJson(req);
      const registration = NOTEBOOK_QUEUE.registerWorker(body);
      await scheduleNotebookChatWarmup();
      return sendJson(res, 200, registration);
    }
    if (req.method === "POST" && url.pathname === "/api/v1/notebook/workers/heartbeat") {
      const body = await readJson(req);
      return sendJson(res, 200, NOTEBOOK_QUEUE.heartbeat(body));
    }
    if (req.method === "POST" && url.pathname === "/api/v1/notebook/workers/jobs/claim") {
      const body = await readJson(req);
      const job = await mutateNotebookQueue(() => NOTEBOOK_QUEUE.claim(body.workerId));
      return job ? sendJson(res, 200, { ok: true, job }) : sendEmpty(res, 204);
    }
    const match = url.pathname.match(/^\/api\/v1\/notebook\/workers\/jobs\/([A-Za-z0-9_-]{1,100})\/(result|fail)$/);
    if (req.method === "POST" && match) {
      const body = await readJson(req);
      const job = await mutateNotebookQueue(() => match[2] === "result"
        ? NOTEBOOK_QUEUE.complete(body.workerId, match[1], body)
        : NOTEBOOK_QUEUE.fail(body.workerId, match[1], body.error));
      if (match[1] === NOTEBOOK_CHAT_WARMUP_JOB_ID) {
        NOTEBOOK_CHAT_WARMUP_JOB_ID = null;
        const readyLine = String(job.result?.stdout || "").split(/\r?\n/).find(line => line.startsWith("DREAM_CHAT_MODEL_READY:"));
        if (job.status === "completed" && readyLine) {
          NOTEBOOK_CHAT_MODEL_READY = true;
          NOTEBOOK_CHAT_MODEL_DEVICE = readyLine.slice("DREAM_CHAT_MODEL_READY:".length).trim() || null;
          NOTEBOOK_CHAT_MODEL_ERROR = "";
          console.log("DREAM_NOTEBOOK_CHAT_MODEL_READY", JSON.stringify({ device: NOTEBOOK_CHAT_MODEL_DEVICE }));
        } else {
          NOTEBOOK_CHAT_MODEL_READY = false;
          NOTEBOOK_CHAT_MODEL_ERROR = "DREAM AI model warm-up did not complete.";
          const diagnostic = String(job.result?.stderr || job.result?.display || "")
            .replace(/(?:hf_[A-Za-z0-9]{20,}|Bearer\\s+\\S+)/gi, "[redacted]")
            .slice(-1200);
          console.warn("DREAM_NOTEBOOK_CHAT_MODEL_WARMUP_FAILED", JSON.stringify({
            output: String(job.result?.stdout || "").slice(-300),
            diagnostic
          }));
        }
      }
      return sendJson(res, 200, { ok: true, job });
    }
    return sendJson(res, 404, { error: "Notebook worker endpoint not found." });
  } catch (error) {
    const status = Number.isInteger(error.status) ? error.status : 500;
    return sendJson(res, status, {
      error: {
        message: status >= 500 ? "DREAM NOTEBOOK worker request failed." : error.message,
        type: "notebook_worker_error"
      }
    });
  }
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", "http://localhost");
    if (req.method === "GET" && url.pathname === "/notebook") {
      return await sendStaticFile(res, "./notebook.html", "text/html; charset=utf-8");
    }
    if (req.method === "GET" && url.pathname === "/notebook/permanent-gpu-setup.md") {
      const content = await readFile(new URL("../docs/PERMANENT_GPU_WORKER.md", import.meta.url));
      res.writeHead(200, {
        ...securityHeaders,
        "content-type": "text/markdown; charset=utf-8",
        "cache-control": "no-store"
      });
      res.end(content);
      return;
    }
    if (req.method === "GET" && url.pathname === "/notebook/training-template.ipynb") {
      const content = await readFile(new URL("../training/dream_native_persistent_notebook.ipynb", import.meta.url));
      res.writeHead(200, {
        ...securityHeaders,
        "content-type": "application/x-ipynb+json; charset=utf-8",
        "cache-control": "no-store"
      });
      res.end(content);
      return;
    }
    if (req.method === "GET" && url.pathname === "/notebook.js") {
      return await sendStaticFile(res, "./notebook.js", "text/javascript; charset=utf-8");
    }
    if (req.method === "GET" && url.pathname === "/notebook.css") {
      return await sendStaticFile(res, "./notebook.css", "text/css; charset=utf-8");
    }
    if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/health" || url.pathname === "/healthz" || url.pathname === "/health/live")) {
      return sendJson(res, 200, statusSnapshot());
    }
    if (req.method === "GET" && url.pathname === "/health/ready") {
      const snapshot = statusSnapshot();
      return sendJson(res, snapshot.modelReady ? 200 : 503, snapshot, snapshot.modelReady ? {} : { "retry-after": "10" });
    }
    if (req.method === "GET" && url.pathname === "/api/v1/status") {
      if (!requireToken(req, res, CLIENT_TOKEN)) return;
      return sendJson(res, 200, { ok: true, ...statusSnapshot() });
    }
    if (req.method === "GET" && url.pathname === "/v1/models") {
      if (!requireToken(req, res, CLIENT_TOKEN)) return;
      return sendJson(res, 200, {
        object: "list",
        data: [{ id: MODEL_NAME, object: "model", created: 0, owned_by: "dream-host" }]
      });
    }
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      return await handleChat(req, res);
    }
    if (url.pathname.startsWith("/api/v1/notebook/artifacts")) {
      return await handleArtifactRoute(req, res, url);
    }
    if (url.pathname.startsWith("/api/v1/notebook/workers/")) {
      return await handleNotebookWorkerRoute(req, res, url);
    }
    if (url.pathname.startsWith("/api/v1/notebook/")) {
      return await handleNotebookClientRoute(req, res, url);
    }
    if (url.pathname.startsWith("/api/v1/workers/")) {
      return await handleWorkerRoute(req, res, url);
    }
    return sendJson(res, 404, { error: "Route not found." });
  } catch (error) {
    const status = Number.isInteger(error.status) ? error.status : 500;
    if (status >= 500) console.error("DREAM HOST request failed:", error.message);
    sendJson(res, status, { error: { message: status >= 500 ? "DREAM HOST could not process the request." : error.message, type: "server_error" } });
  }
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
restoreNotebookQueue().then(() => {
  server.listen(PORT, "0.0.0.0", () => {
    console.log("DREAM HOST v" + HOST_VERSION + " listening on 0.0.0.0:" + PORT);
  });
}).catch(error => {
  console.error("DREAM_NOTEBOOK_PERSISTENCE_STARTUP_FAILED", error.message);
  process.exit(1);
});
