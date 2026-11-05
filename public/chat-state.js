export function appendUserMessage(messages, content, isRetry = false) {
  const lastMessage = messages.at(-1);
  if (isRetry && lastMessage?.role === "user" && lastMessage.content === content) return false;
  messages.push({ role: "user", content });
  return true;
}
