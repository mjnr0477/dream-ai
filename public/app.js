const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const statusText = document.querySelector("#service-status");
const statusDot = document.querySelector(".status-dot");
const messages = [];

function addMessage(role, content) {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "✦";
  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;
  row.append(avatar, body);
  conversation.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
  return body;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    statusText.textContent = health.modelConnected ? "AI model connected" : "Core service online · model not connected";
    statusDot.classList.add("online");
  } catch {
    statusText.textContent = "Service connection unavailable";
  }
}

async function sendMessage(text) {
  const content = text.trim();
  if (!content || send.disabled) return;
  welcome.hidden = true;
  messages.push({ role: "user", content });
  addMessage("user", content);
  input.value = "";
  input.style.height = "auto";
  send.disabled = true;
  send.textContent = "…";
  const replyNode = addMessage("assistant", "Dream AI is thinking…");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The request failed.");
    replyNode.textContent = data.reply;
    messages.push({ role: "assistant", content: data.reply });
    if (messages.length > 12) messages.splice(0, messages.length - 12);
    if (data.mode === "starter") statusText.textContent = "Core service online · connect a model for full AI";
  } catch (error) {
    replyNode.textContent = error.message || "Something went wrong. Please try again.";
    messages.pop();
  } finally {
    send.disabled = false;
    send.textContent = "↑";
    input.focus();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  sendMessage(input.value);
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});
document.querySelector("#new-chat").addEventListener("click", resetChat);
document.querySelector("#mobile-new").addEventListener("click", resetChat);
document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

function resetChat() {
  messages.length = 0;
  conversation.replaceChildren();
  welcome.hidden = false;
  input.value = "";
  input.focus();
}

checkHealth();