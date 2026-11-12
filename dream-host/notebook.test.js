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
