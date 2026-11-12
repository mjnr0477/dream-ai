(() => {
  "use strict";

  const DB_NAME = "dream-notebook-store";
  const DB_VERSION = 1;
  const STORE_NAME = "notebooks";
  const ACTIVE_KEY = "dream-notebook-active-id";
  const TOKEN_KEY = "dream-host-notebook-token";
  const MAX_CELLS = 100;
  const MAX_CELL_CHARS = 20000;
  const MAX_POLL_MS = 30 * 60 * 1000;

  const el = id => document.getElementById(id);
  const cellsRoot = el("cells");
  const tokenInput = el("access-token");
  const connectPanel = el("connect-panel");
  const connectMessage = el("connect-message");
  const toast = el("toast");
  let token = sessionStorage.getItem(TOKEN_KEY) || "";
  let notebook = null;
  let dbPromise = null;
  let refreshHandle = null;
  let running = false;
  let saveTimer = null;

  function id(prefix) {
    const value = typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : String(Date.now()) + Math.random().toString(16).slice(2);
    return prefix + value.replaceAll("-", "");
  }

  function makeCell(type = "code", source = "") {
    return {
      id: id("cell-"),
      type,
      source,
      executionCount: null,
      output: { stdout: "", stderr: "", display: "" },
      status: "idle"
    };
  }

  function makeNotebook(title = "Untitled notebook") {
    return {
      id: id("nb-"),
      title,
      cells: [makeCell("code", 'print("DREAM NOTEBOOK is ready")\n2 + 2')]
    };
  }

  function openDatabase() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      if (!("indexedDB" in window)) {
        reject(new Error("This browser does not support IndexedDB. Export your notebook regularly."));
        return;
      }
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: "id" });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("Notebook storage could not be opened."));
    });
    return dbPromise;
  }

  async function storeNotebook() {
    const db = await openDatabase();
    await new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put(notebook);
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error || new Error("Notebook save failed."));
      transaction.onabort = () => reject(transaction.error || new Error("Notebook save was interrupted."));
    });
    localStorage.setItem(ACTIVE_KEY, notebook.id);
    el("save-state").textContent = "Saved in this browser";
  }

  function scheduleSave() {
    el("save-state").textContent = "Saving…";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      storeNotebook().catch(error => {
        el("save-state").textContent = "Save failed";
        showToast(error.message);
      });
    }, 250);
  }

  async function loadNotebook() {
    const db = await openDatabase();
    const activeId = localStorage.getItem(ACTIVE_KEY);
    if (!activeId) {
      notebook = makeNotebook();
      await storeNotebook();
      return;
    }
    const loaded = await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(activeId);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error("Notebook could not be loaded."));
    });
    notebook = loaded && Array.isArray(loaded.cells) ? loaded : makeNotebook();
    notebook.cells = notebook.cells.slice(0, MAX_CELLS).map(cell => ({
      ...makeCell(cell.type === "markdown" ? "markdown" : "code"),
      ...cell,
      id: typeof cell.id === "string" ? cell.id : id("cell-"),
      source: typeof cell.source === "string" ? cell.source.slice(0, MAX_CELL_CHARS) : "",
      output: {
        stdout: String(cell.output?.stdout || "").slice(0, 120000),
        stderr: String(cell.output?.stderr || "").slice(0, 120000),
        display: String(cell.output?.display || "").slice(0, 120000)
      }
    }));
    if (!notebook.cells.length) notebook.cells = [makeCell()];
  }

  function showToast(message) {
    toast.textContent = String(message || "");
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 3000);
  }

  function statusLabel(text, state = "") {
    el("connection-label").textContent = text;
    el("connection-dot").className = "status-dot" + (state ? " " + state : "");
  }

  function showOutput(container, output, errorText = "") {
    container.replaceChildren();
    const values = [
      ["stdout", output?.stdout, ""],
      ["stderr", output?.stderr, "error"],
      ["display", output?.display, "display"]
    ];
    for (const [label, value, className] of values) {
      if (!value) continue;
      const pre = document.createElement("pre");
      pre.className = "output-block " + className;
      pre.dataset.output = label;
      pre.textContent = String(value);
      container.append(pre);
    }
    if (errorText) {
      const pre = document.createElement("pre");
      pre.className = "output-block error";
      pre.textContent = errorText;
      container.append(pre);
    }
  }

  function render() {
    if (!notebook) return;
    el("notebook-title").value = notebook.title;
    cellsRoot.replaceChildren();

    notebook.cells.forEach((cell, index) => {
      const article = document.createElement("article");
      article.className = "cell " + (cell.type === "markdown" ? "markdown" : "code") + (cell.status === "running" ? " running" : "");
      article.dataset.cellId = cell.id;

      const head = document.createElement("div");
      head.className = "cell-head";
      const kind = document.createElement("span");
      kind.className = "cell-kind";
      kind.textContent = cell.type === "markdown" ? "Markdown" : "Python · In [" + (cell.executionCount ?? " ") + "]";
      head.append(kind);

      const actions = document.createElement("div");
      actions.className = "cell-actions";
      const state = document.createElement("span");
      state.className = "cell-status";
      state.textContent = cell.status === "running" ? "Running…" : cell.status === "failed" ? "Failed" : "";
      actions.append(state);

      if (cell.type === "code") {
        const runButton = document.createElement("button");
        runButton.type = "button";
        runButton.className = "button run-all";
        runButton.textContent = cell.status === "running" ? "Running…" : "Run";
        runButton.disabled = running || cell.status === "running" || !token;
        runButton.addEventListener("click", () => runCell(cell.id));
        actions.append(runButton);
      }

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "button quiet";
      deleteButton.textContent = "Delete";
      deleteButton.disabled = running;
      deleteButton.addEventListener("click", () => {
        if (notebook.cells.length === 1) {
          notebook.cells = [makeCell()];
        } else {
          notebook.cells.splice(index, 1);
        }
        scheduleSave();
        render();
      });
      actions.append(deleteButton);
      head.append(actions);

      const editor = document.createElement("textarea");
      editor.className = "cell-editor";
      editor.setAttribute("aria-label", cell.type === "markdown" ? "Markdown cell source" : "Python cell source");
      editor.spellcheck = false;
      editor.value = cell.source;
      editor.maxLength = MAX_CELL_CHARS;
      editor.addEventListener("input", () => {
        cell.source = editor.value;
        scheduleSave();
      });
      if (cell.type === "code") {
        editor.addEventListener("keydown", event => {
          if (event.key === "Enter" && event.shiftKey) {
            event.preventDefault();
            runCell(cell.id);
          }
        });
      }

      const output = document.createElement("div");
      output.className = "cell-output";
      showOutput(output, cell.output, cell.error || "");

      article.append(head, editor, output);
      cellsRoot.append(article);
    });
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        ...(options.headers || {}),
        authorization: "Bearer " + token,
        ...(options.body ? { "content-type": "application/json" } : {})
      },
      cache: "no-store"
    });
    let body = {};
    try { body = await response.json(); } catch {}
    if (!response.ok) {
      const message = body.error?.message || body.error || "Request failed (" + response.status + ").";
      const error = new Error(String(message));
      error.status = response.status;
      throw error;
    }
    return body;
  }

  async function refreshStatus() {
    if (!token) return;
    try {
      const data = await api("/api/v1/notebook/status");
      connectPanel.hidden = true;
      const count = Number(data.workers?.connected || 0);
      el("worker-count").textContent = String(count);
      el("queue-count").textContent = String(data.queue?.pending || 0);
      if (count > 0) {
        const device = data.workers?.devices?.map(item => item.device).filter(Boolean).join(", ") || "connected";
        el("worker-detail").textContent = device + " · " + (data.workers.busy || 0) + " busy";
        statusLabel("Connected · " + count + " worker" + (count === 1 ? "" : "s"), "online");
      } else {
        el("worker-detail").textContent = "No notebook worker is connected yet. Start an operator-owned worker.";
        statusLabel("Connected · waiting for compute", "warning");
      }
      el("add-code").disabled = false;
      el("add-markdown").disabled = false;
      el("add-code-bottom").disabled = false;
      el("run-all").disabled = running;
    } catch (error) {
      if (error.status === 401) {
        sessionStorage.removeItem(TOKEN_KEY);
        token = "";
        statusLabel("Access key rejected", "warning");
        connectPanel.hidden = false;
      } else if (error.status === 503) {
        statusLabel("Notebook not configured", "warning");
        connectPanel.hidden = false;
        connectMessage.textContent = error.message;
      } else {
        statusLabel("Connection unavailable", "warning");
        el("worker-detail").textContent = error.message;
      }
    }
  }

  async function runCell(cellId) {
    if (!token || running) return;
    const cell = notebook.cells.find(item => item.id === cellId);
    if (!cell || cell.type !== "code") return;
    if (!cell.source.trim()) {
      showToast("Add Python code before running this cell.");
      return;
    }
    running = true;
    cell.status = "running";
    cell.error = "";
    render();
    scheduleSave();
    try {
      const submitted = await api("/api/v1/notebook/jobs", {
        method: "POST",
        body: JSON.stringify({ notebookId: notebook.id, cellId: cell.id, code: cell.source })
      });
      const deadline = Date.now() + MAX_POLL_MS;
      let completed = null;
      while (Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 1100));
        const current = await api("/api/v1/notebook/jobs/" + encodeURIComponent(submitted.job.id));
        if (current.job.status === "completed" || current.job.status === "failed") {
          completed = current.job;
          break;
        }
      }
      if (!completed) throw new Error("This cell exceeded the 30-minute notebook execution window.");
      if (completed.status === "completed") {
        cell.output = completed.result || { stdout: "", stderr: "", display: "" };
        cell.executionCount = cell.output.executionCount ?? ((cell.executionCount || 0) + 1);
        cell.status = "idle";
        cell.error = "";
      } else {
        cell.status = "failed";
        cell.error = completed.error || "Notebook worker failed this cell.";
      }
    } catch (error) {
      cell.status = "failed";
      cell.error = error.message;
      showToast(error.message);
    } finally {
      running = false;
      scheduleSave();
      render();
      el("run-all").disabled = !token;
      await refreshStatus();
    }
  }

  async function runAll() {
    if (!token || running) return;
    for (const cell of notebook.cells) {
      if (cell.type === "code") {
        await runCell(cell.id);
        if (cell.status === "failed") break;
      }
    }
  }

  function addCell(type) {
    if (notebook.cells.length >= MAX_CELLS) {
      showToast("A notebook can contain up to " + MAX_CELLS + " cells.");
      return;
    }
    notebook.cells.push(makeCell(type));
    scheduleSave();
    render();
    const editors = cellsRoot.querySelectorAll("textarea");
    editors[editors.length - 1]?.focus();
  }

  function exportNotebook() {
    const notebookFile = {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
        language_info: { name: "python" }
      },
      cells: notebook.cells.map(cell => {
        const base = {
          cell_type: cell.type,
          id: cell.id.slice(0, 64),
          metadata: {},
          source: cell.source.split("\n").map((line, index, lines) => index < lines.length - 1 ? line + "\n" : line)
        };
        if (cell.type === "markdown") return base;
        const outputs = [];
        if (cell.output?.stdout) outputs.push({ output_type: "stream", name: "stdout", text: cell.output.stdout.split("\n").map((line,index,lines)=>index<lines.length-1?line+"\n":line) });
        if (cell.output?.stderr) outputs.push({ output_type: "stream", name: "stderr", text: cell.output.stderr.split("\n").map((line,index,lines)=>index<lines.length-1?line+"\n":line) });
        if (cell.output?.display) outputs.push({ output_type: "execute_result", execution_count: cell.executionCount || null, metadata: {}, data: { "text/plain": cell.output.display } });
        return { ...base, execution_count: cell.executionCount, outputs };
      })
    };
    const blob = new Blob([JSON.stringify(notebookFile, null, 2)], { type: "application/x-ipynb+json" });
    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");
    anchor.href = url;
    anchor.download = (notebook.title || "dream-notebook").replace(/[^A-Za-z0-9_-]+/g, "-") + ".ipynb";
    anchor.click();
    URL.revokeObjectURL(url);
    showToast("Notebook exported.");
  }

  async function importNotebook(file) {
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) throw new Error("Notebook files must be 5 MB or smaller.");
    const data = JSON.parse(await file.text());
    if (data.nbformat !== 4 || !Array.isArray(data.cells)) throw new Error("Choose a valid Jupyter Notebook (nbformat 4).");
    if (data.cells.length > MAX_CELLS) throw new Error("This notebook exceeds the " + MAX_CELLS + " cell limit.");
    const cells = data.cells.map(item => {
      if (!["code", "markdown"].includes(item.cell_type)) return null;
      const source = Array.isArray(item.source) ? item.source.join("") : String(item.source || "");
      return {
        ...makeCell(item.cell_type, source.slice(0, MAX_CELL_CHARS)),
        executionCount: Number.isInteger(item.execution_count) ? item.execution_count : null
      };
    }).filter(Boolean);
    notebook = makeNotebook(String(file.name).replace(/\.ipynb$/i, "").slice(0, 120) || "Imported notebook");
    notebook.cells = cells.length ? cells : [makeCell()];
    await storeNotebook();
    render();
    showToast("Notebook imported.");
  }

  async function connect(event) {
    event.preventDefault();
    const candidate = tokenInput.value.trim();
    if (candidate.length < 32) {
      connectMessage.textContent = "The notebook key must contain at least 32 characters.";
      return;
    }
    token = candidate;
    sessionStorage.setItem(TOKEN_KEY, candidate);
    connectMessage.textContent = "Checking the private notebook connection…";
    await refreshStatus();
    if (token) connectMessage.textContent = "Connected for this browser session.";
  }

  async function init() {
    try {
      await loadNotebook();
      render();
    } catch (error) {
      showToast(error.message);
      notebook = makeNotebook();
      render();
    }

    el("connect-form").addEventListener("submit", connect);
    el("new-notebook").addEventListener("click", async () => {
      notebook = makeNotebook("Untitled notebook");
      await storeNotebook();
      render();
      showToast("New notebook created.");
    });
    el("add-code").addEventListener("click", () => addCell("code"));
    el("add-code-bottom").addEventListener("click", () => addCell("code"));
    el("add-markdown").addEventListener("click", () => addCell("markdown"));
    el("run-all").addEventListener("click", runAll);
    el("export-notebook").addEventListener("click", exportNotebook);
    el("import-notebook").addEventListener("click", () => el("file-input").click());
    el("file-input").addEventListener("change", async event => {
      try { await importNotebook(event.target.files?.[0]); }
      catch (error) { showToast(error.message); }
      event.target.value = "";
    });
    el("notebook-title").addEventListener("input", event => {
      notebook.title = event.target.value.slice(0, 120);
      scheduleSave();
    });

    connectPanel.hidden = Boolean(token);
    if (token) await refreshStatus();
    refreshHandle = setInterval(refreshStatus, 10000);
  }

  window.addEventListener("beforeunload", () => {
    if (refreshHandle) clearInterval(refreshHandle);
  });
  init();
})();
