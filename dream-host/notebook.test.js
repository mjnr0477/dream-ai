import test from "node:test";
import assert from "node:assert/strict";
import { NotebookQueue } from "./notebook-core.js";

function worker(queue, id = "notebook-worker-01") {
  return queue.registerWorker({
    workerId: id,
    name: "Test notebook worker",
    metadata: { role: "notebook", deviceType: "cpu", python: "3.11" }
  });
}

test("notebook jobs are claimed and completed by a registered notebook worker", () => {
  const queue = new NotebookQueue({ now: () => 1000 });
  worker(queue);
  const submitted = queue.submit({
    notebookId: "notebook-123",
    cellId: "cell-123",
    code: "print('hello')"
  });
  assert.equal(submitted.status, "queued");

  const claimed = queue.claim("notebook-worker-01");
  assert.equal(claimed.id, submitted.id);
  assert.equal(claimed.code, "print('hello')");
  assert.equal(claimed.status, "running");

  const completed = queue.complete("notebook-worker-01", submitted.id, {
    stdout: "hello\n",
    stderr: "",
    display: "None",
    executionCount: 1
  });
  assert.equal(completed.status, "completed");
  assert.equal(completed.result.stdout, "hello\n");
  assert.equal(queue.status().queue.running, 0);
});

test("notebook submissions validate notebook, cell and code bounds", () => {
  const queue = new NotebookQueue({ maxCodeChars: 12 });
  assert.throws(() => queue.submit({ notebookId: "../bad", cellId: "cell-1", code: "print(1)" }), /notebookId/);
  assert.throws(() => queue.submit({ notebookId: "notebook-1", cellId: "cell-1", code: "x".repeat(13) }), /code/);
  assert.throws(() => queue.submit({ notebookId: "notebook-1", cellId: "cell-1", code: "" }), /code/);
});

test("only a registered notebook worker can claim jobs and a worker owns one active job", () => {
  const queue = new NotebookQueue();
  queue.submit({ notebookId: "notebook-1", cellId: "cell-1", code: "1 + 1" });
  assert.throws(() => queue.claim("unknown-worker"), /registered/);

  worker(queue, "notebook-worker-01");
  const first = queue.claim("notebook-worker-01");
  queue.submit({ notebookId: "notebook-1", cellId: "cell-2", code: "2 + 2" });
  assert.throws(() => queue.claim("notebook-worker-01"), /active job/);
  assert.throws(() => queue.complete("notebook-worker-02", first.id, { stdout: "", stderr: "", display: "2" }), /own/);
});

test("queue capacity is bounded and terminal jobs do not occupy queue slots", () => {
  const queue = new NotebookQueue({ maxQueue: 1 });
  const first = queue.submit({ notebookId: "notebook-1", cellId: "cell-1", code: "1" });
  assert.throws(() => queue.submit({ notebookId: "notebook-1", cellId: "cell-2", code: "2" }), /queue is full/);
  worker(queue);
  const claimed = queue.claim("notebook-worker-01");
  queue.complete("notebook-worker-01", first.id, { stdout: "", stderr: "", display: "1" });
  const second = queue.submit({ notebookId: "notebook-1", cellId: "cell-2", code: "2" });
  assert.equal(second.status, "queued");
  assert.equal(claimed.id, first.id);
});

test("stale worker jobs are requeued once and then fail instead of hanging forever", () => {
  let now = 1000;
  const queue = new NotebookQueue({ now: () => now, workerStaleMs: 10, maxAttempts: 2 });
  worker(queue);
  const submitted = queue.submit({ notebookId: "notebook-1", cellId: "cell-1", code: "time.sleep(1)" });
  queue.claim("notebook-worker-01");

  now = 1011;
  queue.sweep();
  assert.equal(queue.getJob(submitted.id).status, "queued");

  worker(queue);
  queue.claim("notebook-worker-01");
  now = 1022;
  queue.sweep();
  assert.equal(queue.getJob(submitted.id).status, "failed");
});

test("worker output is bounded and completion is accepted only once", () => {
  const queue = new NotebookQueue({ maxOutputChars: 8 });
  worker(queue);
  const job = queue.submit({ notebookId: "notebook-1", cellId: "cell-1", code: "print('ok')" });
  queue.claim("notebook-worker-01");
  assert.throws(() => queue.complete("notebook-worker-01", job.id, {
    stdout: "0123456789",
    stderr: "",
    display: ""
  }), /output/);
  const result = queue.complete("notebook-worker-01", job.id, {
    stdout: "ok\n",
    stderr: "",
    display: ""
  });
  assert.equal(result.status, "completed");
  assert.throws(() => queue.complete("notebook-worker-01", job.id, {
    stdout: "",
    stderr: "",
    display: ""
  }), /not running/);
});

test("execution timeout is independent from the shorter worker heartbeat lease", () => {
  const queue = new NotebookQueue({ jobLeaseMs: 15000, maxExecutionMs: 900000 });
  worker(queue);
  queue.submit({ notebookId: "notebook-1", cellId: "cell-long", code: "train_model()" });
  const job = queue.claim("notebook-worker-01");
  assert.equal(job.timeoutMs, 900000);
});

test("queued jobs expire when no notebook worker connects", () => {
  let now = 1000;
  const queue = new NotebookQueue({ now: () => now, queueTtlMs: 5000 });
  const job = queue.submit({ notebookId: "notebook-1", cellId: "cell-waiting", code: "1 + 1" });
  now = 6001;
  const expired = queue.getJob(job.id);
  assert.equal(expired.status, "failed");
  assert.match(expired.error, /waiting for a connected worker/);
});

test("a returning worker can reclaim a stale assignment after registering again", () => {
  let now = 1000;
  const queue = new NotebookQueue({ now: () => now, workerStaleMs: 10, maxAttempts: 2 });
  worker(queue);
  const submitted = queue.submit({ notebookId: "notebook-1", cellId: "cell-reconnect", code: "value = 42" });
  queue.claim("notebook-worker-01");
  now = 1011;
  worker(queue);
  const reclaimed = queue.claim("notebook-worker-01");
  assert.equal(reclaimed.id, submitted.id);
  assert.equal(reclaimed.attempts, 2);
});

test("idle worker polling keeps its heartbeat fresh when the queue is empty", () => {
  let now = 1000;
  const queue = new NotebookQueue({ now: () => now, workerStaleMs: 10 });
  worker(queue);
  now = 1008;
  assert.equal(queue.claim("notebook-worker-01"), null);
  now = 1016;
  assert.equal(queue.claim("notebook-worker-01"), null);
  assert.equal(queue.status().workers.connected, 1);
});
