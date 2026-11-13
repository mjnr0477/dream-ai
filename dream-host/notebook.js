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
  let token = "";
  let authorized = false;
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
    notebook.modifiedAt = new Date().toISOString();
    await new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put(notebook);
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error || new Error("Notebook save failed."));
      transaction.onabort = () => reject(transaction.error || new Error("Notebook save was interrupted."));
    });
    localStorage.setItem(ACTIVE_KEY, notebook.id);
    el("save-state").textContent = "Saved in this browser";
    await updateNotebookList();
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

  async function updateNotebookList() {
    const select = el("notebook-list");
    if (!select || !notebook) return;
    const db = await openDatabase();
    const saved = await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).getAll();
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error || new Error("Saved notebooks could not be listed."));
    });
    saved.sort((a, b) => String(b.modifiedAt || "").localeCompare(String(a.modifiedAt || "")));
    select.replaceChildren();
    for (const item of saved) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = String(item.title || "Untitled notebook").slice(0, 80);
      select.append(option);
    }
    select.value = notebook.id;
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
      },
      status: "idle",
      error: ""
    }));
    if (!notebook.cells.length) notebook.cells = [makeCell()];
    await updateNotebookList();
  }

  async function switchNotebook(notebookId) {
    if (!notebookId || notebookId === notebook.id) return;
    const db = await openDatabase();
    const selected = await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(notebookId);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error("Notebook could not be opened."));
    });
    if (!selected || !Array.isArray(selected.cells)) {
      showToast("That saved notebook could not be found.");
      await updateNotebookList();
      return;
    }
    notebook = selected;
    localStorage.setItem(ACTIVE_KEY, notebook.id);
    render();
    await updateNotebookList();
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

  function updateControls() {
    el("new-notebook").disabled = running;
    el("import-notebook").disabled = running;
    el("notebook-list").disabled = running;
    const locked = running || (!authorized && !token);
    el("add-code").disabled = locked;
    el("add-markdown").disabled = locked;
    el("add-code-bottom").disabled = locked;
    el("run-all").disabled = locked;
    el("download-dataset").disabled = locked;
    el("scan-datasets").disabled = locked;
    el("delete-dataset").disabled = locked;
    el("import-training").disabled = locked;
    el("import-adapter-training").disabled = locked;
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
    updateControls();

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
      credentials: "same-origin",
      headers: {
        ...(options.headers || {}),
        ...(token ? { authorization: "Bearer " + token } : {}),
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
    try {
      const data = await api("/api/v1/notebook/status");
      authorized = true;
      connectPanel.hidden = true;
      const count = Number(data.workers?.connected || 0);
      el("worker-count").textContent = String(count);
      el("queue-count").textContent = String(data.queue?.pending || 0);
      let chatReady = false;
      try {
        const hostResponse = await fetch("/health", { credentials: "same-origin", cache: "no-store" });
        if (hostResponse.ok) {
          const hostStatus = await hostResponse.json();
          chatReady = hostStatus.modelReady === true;
        }
      } catch {}
      if (count > 0) {
        const device = data.workers?.devices?.map(item => item.device).filter(Boolean).join(", ") || "connected";
        el("worker-detail").textContent = device + " · " + (data.workers.busy || 0) + " busy · DREAM AI chat " + (chatReady ? "ready" : "offline");
        statusLabel("Notebook connected · AI chat " + (chatReady ? "ready" : "offline"), chatReady ? "online" : "warning");
      } else {
        el("worker-detail").textContent = "No notebook compute worker is connected. DREAM AI chat runs on a separate model worker.";
        statusLabel("Connected · waiting for compute", "warning");
      }
      updateControls();
      return true;
    } catch (error) {
      if (error.status === 401) {
        authorized = false;
        token = "";
        updateControls();
        statusLabel("Access key required", "warning");
        connectPanel.hidden = false;
      } else if (error.status === 503) {
        statusLabel("Notebook not configured", "warning");
        connectPanel.hidden = false;
        updateControls();
        connectMessage.textContent = error.message;
      } else {
        statusLabel("Connection unavailable", "warning");
        el("worker-detail").textContent = error.message;
      }
      updateControls();
      return false;
    }
  }

  async function runCell(cellId) {
    if ((!authorized && !token) || running) return;
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
      updateControls();
      await refreshStatus();
    }
  }

  async function runAll() {
    if ((!authorized && !token) || running) return;
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

  async function importNotebook(file, preserveWorkspace = false) {
    if (!file) return;
    const workspaceNotebookId = preserveWorkspace ? notebook?.id : null;
    if (file.size > 5 * 1024 * 1024) throw new Error("Notebook files must be 5 MB or smaller.");
    const data = JSON.parse(await file.text());
    if (data.nbformat !== 4 || !Array.isArray(data.cells)) throw new Error("Choose a valid Jupyter Notebook (nbformat 4).");
    if (data.cells.length > MAX_CELLS) throw new Error("This notebook exceeds the " + MAX_CELLS + " cell limit.");
    const cells = data.cells.map(item => {
      if (!["code", "markdown"].includes(item.cell_type)) return null;
      const source = Array.isArray(item.source) ? item.source.join("") : String(item.source || "");
      const output = { stdout: "", stderr: "", display: "" };
      let error = "";
      for (const itemOutput of Array.isArray(item.outputs) ? item.outputs : []) {
        if (itemOutput.output_type === "stream") {
          const name = itemOutput.name === "stderr" ? "stderr" : "stdout";
          const text = Array.isArray(itemOutput.text) ? itemOutput.text.join("") : String(itemOutput.text || "");
          output[name] = (output[name] + text).slice(0, 120000);
        } else if (itemOutput.output_type === "execute_result" || itemOutput.output_type === "display_data") {
          const plain = itemOutput.data?.["text/plain"];
          if (plain !== undefined) {
            output.display = (Array.isArray(plain) ? plain.join("") : String(plain)).slice(0, 120000);
          }
        } else if (itemOutput.output_type === "error") {
          error = Array.isArray(itemOutput.traceback)
            ? itemOutput.traceback.join("\n").slice(0, 120000)
            : String(itemOutput.evalue || "Imported notebook cell failed.");
        }
      }
      return {
        ...makeCell(item.cell_type, source.slice(0, MAX_CELL_CHARS)),
        executionCount: Number.isInteger(item.execution_count) ? item.execution_count : null,
        output,
        error,
        status: error ? "failed" : "idle"
      };
    }).filter(Boolean);
    notebook = makeNotebook(String(file.name).replace(/\.ipynb$/i, "").slice(0, 120) || "Imported notebook");
    if (workspaceNotebookId) notebook.id = workspaceNotebookId;
    notebook.cells = cells.length ? cells : [makeCell()];
    await storeNotebook();
    render();
    showToast("Notebook imported.");
  }

  async function createRememberedSession(candidate, remember) {
    const response = await fetch("/api/v1/notebook/session", {
      method: "POST",
      credentials: "same-origin",
      headers: {
        authorization: "Bearer " + candidate,
        "content-type": "application/json"
      },
      body: JSON.stringify({ remember: Boolean(remember) }),
      cache: "no-store"
    });
    let body = {};
    try { body = await response.json(); } catch {}
    if (!response.ok) {
      throw new Error(body.error?.message || "DREAM HOST could not save this browser session.");
    }
  }

  async function connect(event) {
    event.preventDefault();
    const candidate = tokenInput.value.trim();
    if (candidate.length < 32) {
      connectMessage.textContent = "The notebook key must contain at least 32 characters.";
      return;
    }
    connectMessage.textContent = "Verifying the private notebook key…";
    try {
      await createRememberedSession(candidate, el("remember-key").checked);
      token = "";
      tokenInput.value = "";
      const connected = await refreshStatus();
      connectMessage.textContent = connected
        ? (el("remember-key").checked ? "Connected. This device will stay signed in for 30 days." : "Connected for this browser session.")
        : "The key was accepted, but the notebook service is not responding yet.";
    } catch (error) {
      token = "";
      connectMessage.textContent = error.message;
    }
  }

  async function forgetAccess() {
    try {
      await fetch("/api/v1/notebook/session", {
        method: "DELETE",
        credentials: "same-origin",
        cache: "no-store"
      });
    } finally {
      token = "";
      authorized = false;
      sessionStorage.removeItem(TOKEN_KEY);
      connectPanel.hidden = false;
      statusLabel("Access forgotten", "warning");
      updateControls();
      tokenInput.value = "";
      showToast("This browser has been disconnected from DREAM NOTEBOOK.");
    }
  }

  async function addGeneratedCodeCell(source) {
    if (!authorized || running) return;
    if (notebook.cells.length >= MAX_CELLS) {
      showToast("This notebook has reached the " + MAX_CELLS + " cell limit.");
      return;
    }
    const cell = makeCell("code", source);
    notebook.cells.push(cell);
    await storeNotebook();
    render();
    await runCell(cell.id);
  }

  function validDatasetFilename(value) {
    const name = String(value || "").trim();
    return /^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}$/.test(name)
      && name !== "." && name !== ".." && !name.includes("..");
  }

  function downloadDataset() {
    const source = el("dataset-url").value.trim();
    const filename = el("dataset-filename").value.trim();
    let parsed;
    try { parsed = new URL(source); } catch {}
    if (!parsed || parsed.protocol !== "https:") {
      showToast("Enter a valid HTTPS dataset URL.");
      return;
    }
    if (!validDatasetFilename(filename)) {
      showToast("Use a simple filename without folders or path traversal.");
      return;
    }
    const code = [
      "from pathlib import Path",
      "from urllib.parse import urlparse",
      "from urllib.request import urlopen",
      "source = " + JSON.stringify(source),
      "filename = " + JSON.stringify(filename),
      "if urlparse(source).scheme != 'https': raise ValueError('Only HTTPS dataset downloads are allowed.')",
      "target = Path('datasets') / filename",
      "target.parent.mkdir(parents=True, exist_ok=True)",
      "downloaded = 0",
      "try:",
      "    with urlopen(source, timeout=60) as response, target.open('wb') as output:",
      "        while True:",
      "            chunk = response.read(1024 * 1024)",
      "            if not chunk: break",
      "            downloaded += len(chunk)",
      "            if downloaded > 2 * 1024 * 1024 * 1024: raise ValueError('Dataset exceeds the 2 GiB per-file limit.')",
      "            output.write(chunk)",
      "except Exception:",
      "    target.unlink(missing_ok=True)",
      "    raise",
      "print('Dataset saved:', target.resolve())",
      "print('Bytes:', downloaded)"
    ].join("\n");
    el("dataset-status").textContent = "Downloading " + filename + " to the connected worker…";
    addGeneratedCodeCell(code).catch(error => showToast(error.message));
  }

  function scanDatasets() {
    const code = [
      "from pathlib import Path",
      "import itertools",
      "roots = [('Notebook datasets', Path('datasets')), ('Attached Kaggle datasets', Path('/kaggle/input'))]",
      "for label, root in roots:",
      "    print('\\n' + label + ':')",
      "    if not root.exists():",
      "        print('  (not available)')",
      "        continue",
      "    count = 0",
      "    for path in itertools.islice(sorted(root.rglob('*')), 500):",
      "        if path.is_file():",
      "            print(f'  {path}  |  {path.stat().st_size:,} bytes')",
      "            count += 1",
      "    if count == 0: print('  (no files found)')",
      "    if count == 500: print('  Showing the first 500 files only.')"
    ].join("\n");
    el("dataset-status").textContent = "Scanning the worker's dataset folders…";
    addGeneratedCodeCell(code).catch(error => showToast(error.message));
  }

  function deleteDataset() {
    const filename = el("dataset-delete-name").value.trim();
    if (!validDatasetFilename(filename)) {
      showToast("Enter the exact downloaded filename to delete.");
      return;
    }
    const code = [
      "from pathlib import Path",
      "filename = " + JSON.stringify(filename),
      "target = Path('datasets') / filename",
      "if target.is_file():",
      "    target.unlink()",
      "    print('Deleted:', target)",
      "else:",
      "    print('No downloaded file found with that name:', filename)"
    ].join("\n");
    el("dataset-status").textContent = "Deleting " + filename + " from the notebook workspace…";
    addGeneratedCodeCell(code).catch(error => showToast(error.message));
  }

  async function importTrainingWorkflow() {
    if (!authorized || running) return;
    if (!window.confirm("Load DREAM’s native from-scratch training workflow? This replaces the current notebook, so export any unsaved work first.")) return;
    try {
      const response = await fetch("/notebook/training-template.ipynb", {
        credentials: "same-origin",
        cache: "no-store"
      });
      if (!response.ok) throw new Error("The DREAM training workflow could not be loaded.");
      const blob = await response.blob();
      const file = new File([blob], "dream-native-training.ipynb", { type: "application/json" });
      await importNotebook(file, true);
      el("dataset-status").textContent = "Native DREAM workflow loaded. Start the Kaggle GPU connector, attach the reviewed corpus, then run the cells here.";
    } catch (error) {
      showToast(error.message);
    }
  }

  async function importAdapterWorkflow() {
    if (!authorized || running) return;
    if (!window.confirm("Load the separate Qwen adapter-improvement workflow? It fine-tunes the existing adapter, not native DREAM. This replaces the open notebook, so export unsaved work first.")) return;
    try {
      const response = await fetch("/notebook/adapter-template.ipynb", {
        credentials: "same-origin",
        cache: "no-store"
      });
      if (!response.ok) throw new Error("The adapter-improvement workflow could not be loaded.");
      const blob = await response.blob();
      const file = new File([blob], "dream-adapter-improvement.ipynb", { type: "application/json" });
      await importNotebook(file, true);
      el("dataset-status").textContent = "Adapter-improvement workflow loaded. Attach the four reviewed opt-in files and set HF_TOKEN in the Kaggle GPU connector only if you intend to train this separate Qwen-backed adapter.";
    } catch (error) {
      showToast(error.message);
    }
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
    el("forget-key").addEventListener("click", forgetAccess);
    el("download-dataset").addEventListener("click", downloadDataset);
    el("scan-datasets").addEventListener("click", scanDatasets);
    el("delete-dataset").addEventListener("click", deleteDataset);
    el("import-training").addEventListener("click", importTrainingWorkflow);
    el("import-adapter-training").addEventListener("click", importAdapterWorkflow);
    el("notebook-list").addEventListener("change", event => {
      switchNotebook(event.target.value).catch(error => showToast(error.message));
    });
    el("new-notebook").addEventListener("click", async () => {
      await storeNotebook();
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

    connectPanel.hidden = false;
    updateControls();
    const legacyToken = sessionStorage.getItem(TOKEN_KEY) || "";
    if (legacyToken) {
      try { await createRememberedSession(legacyToken, true); } catch {}
      sessionStorage.removeItem(TOKEN_KEY);
    }
    await refreshStatus();
    refreshHandle = setInterval(refreshStatus, 10000);
  }

  window.addEventListener("beforeunload", () => {
    if (refreshHandle) clearInterval(refreshHandle);
  });
  init();
})();
