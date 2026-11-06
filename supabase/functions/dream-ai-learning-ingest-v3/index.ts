import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2";

const EXPECTED = "1eee6011918e5568124a2b23dcc97a34316f3cedb258df31096610fb9efd6816";
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

async function digest(value: string) {
  const result = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(result)]
    .map(byte => byte.toString(16).padStart(2, "0"))
    .join("");
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  if (await digest(req.headers.get("x-dream-learning-token") || "") !== EXPECTED) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const raw = await req.text();
  if (new TextEncoder().encode(raw).length > 24000) {
    return Response.json({ error: "Payload too large" }, { status: 413 });
  }

  let body: any;
  try {
    body = JSON.parse(raw);
  } catch {
    return Response.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const mode = body?.mode;
  const prompt = typeof body?.prompt === "string" ? body.prompt.trim() : "";
  const response = typeof body?.response === "string" ? body.response.trim() : "";
  const consentVersion = body?.consentVersion;
  const sourceGroupId = body?.sourceGroupId;

  if (!["general", "coding", "learning"].includes(mode) ||
      !prompt || prompt.length > 8000 ||
      !response || response.length > 16000) {
    return Response.json({ error: "Invalid training example" }, { status: 400 });
  }

  if (consentVersion !== "dream-learning-opt-in-v1" &&
      consentVersion !== "dream-learning-opt-in-v2" &&
      consentVersion !== "dream-learning-opt-in-v3") {
    return Response.json({ error: "Invalid learning consent version" }, { status: 400 });
  }

  let metadata: Record<string, unknown> = {};
  if (consentVersion === "dream-learning-opt-in-v2" || consentVersion === "dream-learning-opt-in-v3") {
    if (typeof sourceGroupId !== "string" || !UUID_V4.test(sourceGroupId)) {
      return Response.json({ error: "A valid temporary session group is required" }, { status: 400 });
    }
    metadata = {
      source_group_sha256: await digest("dream-ai-session-group-v1\0" + sourceGroupId)
    };
  }

  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    { auth: { persistSession: false, autoRefreshToken: false } }
  );

  const contentHash = await digest(consentVersion !== "dream-learning-opt-in-v1"
    ? mode + "\n" + prompt + "\n" + response + "\n" + consentVersion + "\n" + String(metadata.source_group_sha256)
    : mode + "\n" + prompt + "\n" + response);
  const { error } = await supabase.from("dream_ai_training_examples").insert({
    mode,
    input_text: prompt,
    target_text: response,
    prompt,
    response,
    consented: true,
    consent_version: consentVersion,
    consented_at: new Date().toISOString(),
    review_status: "pending",
    status: "pending",
    content_sha256: contentHash,
    metadata
  });

  if (error?.code === "23505") {
    return Response.json({ accepted: true, duplicate: true }, { status: 200 });
  }
  if (error) {
    console.error("Dream AI learning insert failed:", error.code);
    return Response.json({ error: "Could not save this example" }, { status: 500 });
  }

  return Response.json({ accepted: true, status: "pending" }, { status: 201 });
});
