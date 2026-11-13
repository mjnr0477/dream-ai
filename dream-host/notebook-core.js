import { randomUUID } from "node:crypto";

export class NotebookQueueError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.name = "NotebookQueueError";
    this.status = status;
  }
}

function boundedInteger(value, fallback, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(Math.trunc(parsed), maximum));
}

function validId(value, label) {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]{1,100}$/.test(value.trim())) {
    throw new NotebookQueueError(label + " must contain 1 to 100 letters, numbers, underscores or hyphens.");
  }
  return value.trim();
}

function cleanMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new NotebookQueueError("Worker metadata must be an object.");
  }
  if (value.role !== "notebook") {
    throw new NotebookQueueError("Worker must register with role notebook.");
  }
  const safe = {};
  for (const key of ["role", "device", "deviceType", "python", "torch", "workerVersion"]) {
    const candidate = value[key];
    if (typeof candidate === "string" && candidate.trim()) safe[key] = candidate.trim().slice(0, 120);
  }
  safe.role = "notebook";
  return safe;
}

export class NotebookQueue {
  constructor(options = {}) {
    this.now = typeof options.now === "function" ? options.now : Date.now;
    this.maxQueue = boundedInteger(options.maxQueue, 12, 1, 100);
    this.maxCodeChars = boundedInteger(options.maxCodeChars, 20000, 1, 100000);
    this.maxOutputChars = boundedInteger(options.maxOutputChars, 120000, 1, 200000);
    this.workerStaleMs = boundedInteger(options.workerStaleMs, 60000, 1, 300000);
    this.jobLeaseMs = boundedInteger(options.jobLeaseMs, 180000, 5000, 1800000);
    this.maxExecutionMs = boundedInteger(options.maxExecutionMs, 1800000, 30000, 1800000);
    this.maxAttempts = boundedInteger(options.maxAttempts, 2, 1, 5);
    this.queueTtlMs = boundedInteger(options.queueTtlMs, 30 * 60 * 1000, 1000, 24 * 60 * 60 * 1000);
    this.jobTtlMs = boundedInteger(options.jobTtlMs, 15 * 60 * 1000, 60000, 24 * 60 * 60 * 1000);
    this.workers = new Map();
    this.jobs = new Map();
    this.pending = [];
  }

  submit(input = {}) {
    this.sweep();
    const notebookId = validId(input.notebookId, "notebookId");
    const cellId = validId(input.cellId, "cellId");
    if (typeof input.code !== "string" || !input.code.trim()) {
      throw new NotebookQueueError("code must contain Python source.");
    }
    if (input.code.length > this.maxCodeChars) {
      throw new NotebookQueueError("code exceeds the " + this.maxCodeChars + " character limit.", 413);
    }

    const queued = this.pending.filter(id => this.jobs.get(id)?.status === "queued").length;
    if (queued >= this.maxQueue) {
      throw new NotebookQueueError("Notebook queue is full. Retry after a worker completes a cell.", 429);
    }

    const now = this.now();
    const job = {
      id: "nbjob_" + randomUUID().replaceAll("-", ""),
      notebookId,
      cellId,
      code: input.code,
      status: "queued",
      workerId: null,
      attempts: 0,
      createdAt: new Date(now).toISOString(),
      updatedAt: new Date(now).toISOString(),
      leaseExpiresAt: null,
      result: null,
      error: null
    };
    this.jobs.set(job.id, job);
    this.pending.push(job.id);
    return this.publicJob(job);
  }

  registerWorker(input = {}) {
    this.sweep();
    const workerId = validId(input.workerId, "workerId");
    const metadata = cleanMetadata(input.metadata);
    const now = this.now();
    const existing = this.workers.get(workerId);
    const worker = existing || {
      id: workerId,
      name: "DREAM NOTEBOOK worker",
      connectedAt: new Date(now).toISOString(),
      status: "idle",
      busyJobId: null
    };
    worker.name = typeof input.name === "string" && input.name.trim()
      ? input.name.trim().slice(0, 80)
      : worker.name;
    worker.metadata = metadata;
    worker.lastSeen = now;
    this.workers.set(workerId, worker);
    return { ok: true, workerId, heartbeatIntervalSeconds: 15 };
  }

  heartbeat(input = {}) {
    const workerId = validId(input.workerId, "workerId");
    const worker = this.workers.get(workerId);
    if (!worker) throw new NotebookQueueError("Notebook worker is not registered.", 409);
    const now = this.now();
    worker.lastSeen = now;
    worker.status = worker.busyJobId ? "busy" : (input.busy === true ? "busy" : "idle");
    if (worker.busyJobId) {
      const job = this.jobs.get(worker.busyJobId);
      if (job?.status === "running") {
        job.leaseExpiresAt = now + this.jobLeaseMs;
        job.updatedAt = new Date(now).toISOString();
      }
    }
    return { ok: true, workerId, serverTime: new Date(now).toISOString() };
  }

  claim(workerIdValue) {
    this.sweep();
    const workerId = validId(workerIdValue, "workerId");
    const worker = this.workers.get(workerId);
    if (!worker || this.now() - worker.lastSeen > this.workerStaleMs) {
      throw new NotebookQueueError("Notebook worker is not registered or its heartbeat is stale.", 409);
    }
    if (worker.metadata?.role !== "notebook") {
      throw new NotebookQueueError("Worker does not have notebook capability.", 403);
    }
    if (worker.busyJobId) {
      throw new NotebookQueueError("Notebook worker already has an active job.", 409);
    }

    while (this.pending.length) {
      const id = this.pending.shift();
      const job = this.jobs.get(id);
      if (!job || job.status !== "queued") continue;
      const now = this.now();
      job.status = "running";
      job.workerId = workerId;
      job.attempts += 1;
      job.leaseExpiresAt = now + this.jobLeaseMs;
      job.updatedAt = new Date(now).toISOString();
      worker.status = "busy";
      worker.busyJobId = job.id;
      worker.lastSeen = now;
      return {
        id: job.id,
        notebookId: job.notebookId,
        cellId: job.cellId,
        code: job.code,
        status: "running",
        attempts: job.attempts,
        timeoutMs: this.maxExecutionMs
      };
    }

    worker.status = "idle";
    worker.lastSeen = this.now();
    return null;
  }

  complete(workerIdValue, jobIdValue, result = {}) {
    const workerId = validId(workerIdValue, "workerId");
    const jobId = validId(jobIdValue, "jobId");
    const job = this.jobs.get(jobId);
    if (!job || job.status !== "running") {
      throw new NotebookQueueError("Notebook job is not running.", 404);
    }
    const worker = this.workers.get(workerId);
    if (!worker || job.workerId !== workerId || worker.busyJobId !== jobId) {
      throw new NotebookQueueError("Worker does not own this notebook job.", 403);
    }

    const output = {};
    for (const key of ["stdout", "stderr", "display"]) {
      const value = result[key] === undefined || result[key] === null ? "" : result[key];
      if (typeof value !== "string") throw new NotebookQueueError(key + " must be text.");
      if (value.length > this.maxOutputChars) {
        throw new NotebookQueueError("Notebook job output exceeds the " + this.maxOutputChars + " character limit.", 413);
      }
      output[key] = value;
    }
    const executionCount = Number(result.executionCount);
    output.executionCount = Number.isInteger(executionCount) && executionCount >= 0 ? executionCount : null;

    const now = this.now();
    job.status = "completed";
    job.result = output;
    job.workerId = null;
    job.leaseExpiresAt = null;
    job.updatedAt = new Date(now).toISOString();
    worker.status = "idle";
    worker.busyJobId = null;
    worker.lastSeen = now;
    return this.publicJob(job);
  }

  fail(workerIdValue, jobIdValue, errorValue) {
    const workerId = validId(workerIdValue, "workerId");
    const jobId = validId(jobIdValue, "jobId");
    const job = this.jobs.get(jobId);
    if (!job || job.status !== "running") {
      throw new NotebookQueueError("Notebook job is not running.", 404);
    }
    const worker = this.workers.get(workerId);
    if (!worker || job.workerId !== workerId || worker.busyJobId !== jobId) {
      throw new NotebookQueueError("Worker does not own this notebook job.", 403);
    }
    const now = this.now();
    job.status = "failed";
    job.error = typeof errorValue === "string" && errorValue.trim()
      ? errorValue.trim().slice(0, 2000)
      : "Notebook worker failed to execute this cell.";
    job.workerId = null;
    job.leaseExpiresAt = null;
    job.updatedAt = new Date(now).toISOString();
    worker.status = "idle";
    worker.busyJobId = null;
    worker.lastSeen = now;
    return this.publicJob(job);
  }

  getJob(jobIdValue) {
    this.sweep();
    const jobId = validId(jobIdValue, "jobId");
    const job = this.jobs.get(jobId);
    return job ? this.publicJob(job) : null;
  }

  snapshot() {
    return {
      schema: "dream-notebook-queue-v1",
      savedAt: new Date(this.now()).toISOString(),
      jobs: [...this.jobs.values()].map(job => ({
        id: job.id,
        notebookId: job.notebookId,
        cellId: job.cellId,
        code: job.code,
        status: job.status,
        attempts: job.attempts,
        createdAt: job.createdAt,
        updatedAt: job.updatedAt,
        result: job.result,
        error: job.error
      })).slice(-500)
    };
  }

  restore(snapshot) {
    if (!snapshot || snapshot.schema !== "dream-notebook-queue-v1" || !Array.isArray(snapshot.jobs)) {
      throw new NotebookQueueError("Persisted notebook queue has an unsupported format.", 503);
    }
    if (snapshot.jobs.length > 500) {
      throw new NotebookQueueError("Persisted notebook queue exceeds the recovery limit.", 503);
    }

    const restored = new Map();
    const pending = [];
    for (const item of snapshot.jobs) {
      if (!item || typeof item !== "object" || Array.isArray(item)) continue;
      if (typeof item.id !== "string" || !/^nbjob_[a-f0-9]{32}$/.test(item.id)) continue;
      if (typeof item.notebookId !== "string" || !/^[A-Za-z0-9_-]{1,100}$/.test(item.notebookId)) continue;
      if (typeof item.cellId !== "string" || !/^[A-Za-z0-9_-]{1,100}$/.test(item.cellId)) continue;
      if (typeof item.code !== "string" || !item.code.trim() || item.code.length > this.maxCodeChars) continue;
      if (!["queued", "running", "completed", "failed"].includes(item.status)) continue;
      const created = Date.parse(item.createdAt);
      const updated = Date.parse(item.updatedAt);
      if (!Number.isFinite(created) || !Number.isFinite(updated)) continue;
      const attempts = boundedInteger(item.attempts, 0, 0, this.maxAttempts);
      let status = item.status;
      let error = typeof item.error === "string" ? item.error.slice(0, 2000) : null;
      if (status === "running") {
        if (attempts < this.maxAttempts) {
          status = "queued";
          error = null;
        } else {
          status = "failed";
          error = "Notebook host restarted while this job was running; retry limit reached.";
        }
      }
      let result = null;
      if (status === "completed" && item.result && typeof item.result === "object" && !Array.isArray(item.result)) {
        const output = {};
        let valid = true;
        for (const key of ["stdout", "stderr", "display"]) {
          const value = item.result[key] ?? "";
          if (typeof value !== "string" || value.length > this.maxOutputChars) valid = false;
          output[key] = typeof value === "string" ? value : "";
        }
        const count = Number(item.result.executionCount);
        output.executionCount = Number.isInteger(count) && count >= 0 ? count : null;
        if (valid) result = output;
        else { status = "failed"; error = "Saved notebook output failed recovery validation."; }
      } else if (status === "completed") {
        status = "failed";
        error = "Saved notebook result was missing or invalid.";
      }
      const job = {
        id: item.id,
        notebookId: item.notebookId,
        cellId: item.cellId,
        code: item.code,
        status,
        workerId: null,
        attempts,
        createdAt: new Date(created).toISOString(),
        updatedAt: new Date(updated).toISOString(),
        leaseExpiresAt: null,
        result,
        error
      };
      restored.set(job.id, job);
      if (status === "queued") pending.push(job.id);
    }

    this.jobs = restored;
    this.pending = pending;
    for (const worker of this.workers.values()) {
      worker.status = "idle";
      worker.busyJobId = null;
    }
    this.sweep();
    return { restoredJobs: this.jobs.size, pending: this.pending.length };
  }

  status() {
    this.sweep();
    const now = this.now();
    const connected = [...this.workers.values()].filter(worker => now - worker.lastSeen <= this.workerStaleMs);
    return {
      ok: true,
      workers: {
        connected: connected.length,
        idle: connected.filter(worker => worker.status === "idle").length,
        busy: connected.filter(worker => worker.status === "busy").length,
        devices: connected.map(worker => ({
          name: worker.name,
          device: worker.metadata?.device || worker.metadata?.deviceType || "unknown",
          python: worker.metadata?.python || null,
          torch: worker.metadata?.torch || null
        }))
      },
      queue: {
        pending: this.pending.filter(id => this.jobs.get(id)?.status === "queued").length,
        running: [...this.jobs.values()].filter(job => job.status === "running").length,
        limit: this.maxQueue
      }
    };
  }

  sweep() {
    const now = this.now();
    for (const [workerId, worker] of this.workers) {
      const stale = now - worker.lastSeen > this.workerStaleMs;
      const job = worker.busyJobId ? this.jobs.get(worker.busyJobId) : null;
      const leaseExpired = job?.status === "running" && job.leaseExpiresAt !== null && now > job.leaseExpiresAt;
      if (!stale && !leaseExpired) continue;
      this.workers.delete(workerId);
      if (!job || job.status !== "running") continue;
      job.workerId = null;
      job.leaseExpiresAt = null;
      job.updatedAt = new Date(now).toISOString();
      if (job.attempts < this.maxAttempts) {
        job.status = "queued";
        this.pending.unshift(job.id);
      } else {
        job.status = "failed";
        job.error = "Notebook worker disconnected before completing this cell.";
      }
    }
    for (const [jobId, job] of this.jobs) {
      const createdAge = now - Date.parse(job.createdAt);
      if (job.status === "queued" && createdAge > this.queueTtlMs) {
        job.status = "failed";
        job.error = "Notebook job expired while waiting for a connected worker.";
        job.updatedAt = new Date(now).toISOString();
      }
      const terminal = job.status === "completed" || job.status === "failed";
      const age = now - Date.parse(job.updatedAt || job.createdAt);
      if (terminal && age > this.jobTtlMs) this.jobs.delete(jobId);
    }
    this.pending.splice(0, this.pending.length, ...this.pending.filter(id => this.jobs.get(id)?.status === "queued"));
  }

  publicJob(job) {
    return {
      id: job.id,
      notebookId: job.notebookId,
      cellId: job.cellId,
      status: job.status,
      attempts: job.attempts,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      result: job.result,
      error: job.error
    };
  }
}
