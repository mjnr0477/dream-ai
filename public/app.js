const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const statusText = document.querySelector("#service-status");
const statusDot = document.querySelector(".status-dot");
const historyList = document.querySelector("#history-list");
const STORAGE_KEY = "dream-ai-conversations-v1";
const MAX_SAVED_CHATS = 30;
let chats = loadChats();
let activeChatId = null;
let messages = [];
let activeMode = "general";

function loadChats() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value.filter(chat => chat && typeof chat.id === "string" && Array.isArray(chat.messages)) : [];
  } catch { return []; }
}

function saveChats() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(chats.slice(0, MAX_SAVED_CHATS))); }
  catch { statusText.textContent = "Local storage full · export chats to keep them"; }
}

function titleFor(items) {
  const first = items.find(item => item.role === "user");
  return (first?.content || "New conversation").replace(/\s+/g, " ").slice(0, 54);
}

function persistActiveChat() {
  if (!messages.length) return;
  const chat = {
    id: activeChatId || (globalThis.crypto?.randomUUID?.() || String(Date.now())),
    title: titleFor(messages),
    mode: activeMode,
    updatedAt: Date.now(),
    messages: messages.slice(-60)
  };
  activeChatId = chat.id;
  chats = [chat, ...chats.filter(item => item.id !== chat.id)].slice(0, MAX_SAVED_CHATS);
  saveChats();
  renderHistory();
}

function renderHistory() {
  historyList.replaceChildren();
  if (!chats.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "Your saved conversations will appear here.";
    historyList.append(empty);
    return;
  }
  for (const chat of chats) {
    const button = document.createElement("button");
    button.className = "history-item" + (chat.id === activeChatId ? " selected" : "");
    button.textContent = chat.title || "Conversation";
    button.title = chat.title || "Conversation";
    button.addEventListener("click", () => openChat(chat.id));
    historyList.append(button);
  }
}


function renderAssistantContent(container, content) {
  container.replaceChildren();
  const segments = content.split(/```([A-Za-z0-9_+-]*)\n([\s\S]*?)```/g);

  function appendInline(parent, text) {
    const pattern = /(`[^`]+?`|\*\*[^*]+?\*\*|\*[^*]+?\*)/g;
    let cursor = 0;
    for (const match of text.matchAll(pattern)) {
      const token = match[0];
      const start = match.index;
      if (start > cursor) parent.append(document.createTextNode(text.slice(cursor, start)));
      const node = document.createElement(token.startsWith("`") ? "code" : token.startsWith("**") ? "strong" : "em");
      node.textContent = token.startsWith("`") ? token.slice(1, -1) : token.startsWith("**") ? token.slice(2, -2) : token.slice(1, -1);
      if (node.tagName === "CODE") node.className = "inline-code";
      parent.append(node);
      cursor = start + token.length;
    }
    if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
  }

  for (let i = 0; i < segments.length; i += 3) {
    const prose = segments[i];
    if (prose) appendInline(container, prose);
    const language = segments[i + 1];
    const codeText = segments[i + 2];
    if (codeText !== undefined) {
      const block = document.createElement("div");
      block.className = "code-block";
      const header = document.createElement("div");
      header.className = "code-block-header";
      const label = document.createElement("span");
      label.textContent = language || "CODE";
      const copy = document.createElement("button");
      copy.type = "button";
      copy.textContent = "Copy code";
      copy.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(codeText); copy.textContent = "Copied"; }
        catch { copy.textContent = "Copy unavailable"; }
      });
      header.append(label, copy);
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = codeText;
      pre.append(code);
      block.append(header, pre);
      container.append(block);
    }
  }
}

function addMessage(role, content) {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "✳";
  const body = document.createElement("div");
  body.className = "message-content";
  if (role === "assistant") renderAssistantContent(body, content);
  else body.textContent = content;
  row.append(avatar, body);
  conversation.append(row);
  if (role === "assistant" && content && !/thinking/i.test(content)) {
    const tools = document.createElement("div");
    tools.className = "message-tools";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy response";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(content); copy.textContent = "Copied"; }
      catch { copy.textContent = "Copy unavailable"; }
    });
    tools.append(copy);
    row.append(tools);
  }
  row.scrollIntoView({ behavior: "smooth", block: "end" });
  return body;
}

function renderMessages() {
  conversation.replaceChildren();
  for (const item of messages) addMessage(item.role, item.content);
  welcome.hidden = messages.length > 0;
}

function setMode(mode) {
  activeMode = ["general", "coding", "learning"].includes(mode) ? mode : "general";
  document.querySelectorAll("[data-mode-select]").forEach(button => {
    button.classList.toggle("active", button.dataset.modeSelect === activeMode);
  });
}

function openChat(id) {
  const chat = chats.find(item => item.id === id);
  if (!chat) return;
  activeChatId = chat.id;
  messages = chat.messages.map(item => ({ role: item.role, content: item.content }));
  setMode(chat.mode);
  renderMessages();
  renderHistory();
}

const capabilityMenu = document.querySelector("#capability-menu");
const capabilityModel = document.querySelector("#capability-model");
const capabilityTools = document.querySelector("#capability-tools");

function readableStatus(status) {
  return String(status || "unknown").replaceAll("-", " ");
}

async function loadCapabilities() {
  if (!capabilityMenu || capabilityMenu.dataset.loaded === "true") return;
  capabilityModel.textContent = "Checking the native model and connected tools…";
  capabilityTools.replaceChildren();
  try {
    const response = await fetch("/api/capabilities", { headers: { accept: "application/json" } });
    if (!response.ok) throw new Error("Capability status is unavailable.");
    const data = await response.json();
    const model = data.model || {};
    capabilityModel.textContent = `Native DREAM model: ${readableStatus(model.status)}${Number.isSafeInteger(model.completedTrainingSteps) ? ` · ${model.completedTrainingSteps.toLocaleString()} training steps` : ""}`;
    for (const tool of Array.isArray(data.tools) ? data.tools : []) {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = tool.label || tool.id || "Unnamed capability";
      const state = document.createElement("span");
      state.textContent = readableStatus(tool.status);
      const description = document.createElement("p");
      description.textContent = tool.description || "No further details are available.";
      item.append(title, state, description);
      capabilityTools.append(item);
    }
    capabilityMenu.dataset.loaded = "true";
  } catch (error) {
    capabilityModel.textContent = error.message || "Unable to load verified capability status.";
  }
}

capabilityMenu?.addEventListener("toggle", () => {
  if (capabilityMenu.open) loadCapabilities();
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    if (!response.ok) throw new Error("Health check failed");
    statusText.textContent = health.modelReady
      ? "DREAM native model ready"
      : health.modelConfigured
        ? "Application online · native model unavailable"
        : "Application online · native model setup needed";
    statusDot.classList.add("online");
  } catch {
    statusText.textContent = "Service connection unavailable";
    statusDot.classList.remove("online");
  }
}

async function sendMessage(text, isRetry = false) {
  const content = text.trim();
  if (!content || send.disabled) return;
  welcome.hidden = true;
  messages.push({ role: "user", content });
  if (!isRetry) addMessage("user", content);
  input.value = "";
  input.style.height = "auto";
  send.disabled = true;
  send.textContent = "…";
  const replyNode = addMessage("assistant", "Dream AI is thinking…");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: messages.slice(-12), mode: activeMode })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The request failed.");
    renderAssistantContent(replyNode, data.reply);
    statusText.textContent = data.source === "dream-model"
      ? "DREAM native model connected"
      : "Application online · native model not connected";
    const tools = document.createElement("div");
    tools.className = "message-tools";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy response";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(data.reply); copy.textContent = "Copied"; }
      catch { copy.textContent = "Copy unavailable"; }
    });
    tools.append(copy);
    replyNode.parentElement.append(tools);
    messages.push({ role: "assistant", content: data.reply });
    persistActiveChat();
    if (data.mode === "starter") {
      statusText.textContent = "Service online · connect a model for real AI replies";
    }
  } catch (error) {
    replyNode.textContent = error.message || "Something went wrong. Please try again.";
    statusText.textContent = /retry in a few seconds/i.test(error.message || "")
      ? "DREAM AI is busy · please retry shortly"
      : "Model runtime unavailable · check connection";
    messages.pop();

    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "retry-message";
    retry.textContent = "Retry message";
    retry.addEventListener("click", () => {
      if (send.disabled) return;
      retry.closest(".message")?.remove();
      sendMessage(content, true);
    });
    replyNode.append(document.createTextNode(" "));
    replyNode.append(retry);
  } finally {
    send.disabled = false;
    send.textContent = "↑";
    input.focus();
  }
}

function resetChat() {
  activeChatId = null;
  messages = [];
  conversation.replaceChildren();
  welcome.hidden = false;
  input.value = "";
  renderHistory();
  input.focus();
}

form.addEventListener("submit", event => {
  event.preventDefault();
  sendMessage(input.value);
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
document.querySelector("#new-chat").addEventListener("click", resetChat);
document.querySelector("#mobile-new").addEventListener("click", resetChat);
document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => {
    setMode(button.dataset.mode || "general");
    sendMessage(button.dataset.prompt);
  });
});
document.querySelectorAll("[data-mode-select]").forEach(button => {
  button.addEventListener("click", () => setMode(button.dataset.modeSelect));
});
document.querySelector("#clear-history").addEventListener("click", () => {
  if (!confirm("Clear all saved conversations from this device?")) return;
  chats = [];
  saveChats();
  resetChat();
});
document.querySelector("#export-chat").addEventListener("click", () => {
  if (!messages.length) { statusText.textContent = "Nothing to export yet"; return; }
  const text = messages.map(item => `${item.role.toUpperCase()}\n${item.content}`).join("\n\n");
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `dream-ai-chat-${new Date().toISOString().slice(0,10)}.txt`;
  link.click();
  URL.revokeObjectURL(url);
});

renderHistory();
checkHealth();