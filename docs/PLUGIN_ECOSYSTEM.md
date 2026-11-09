# DREAM AI plugin ecosystem and account access

## What the Plugins directory means

The Plugins directory is a capability catalogue. Each entry is labelled as an implemented public connector, a user-authorized account integration, a provider-gated integration, catalogued work or a planned integration. Listing a service does not mean it is connected. The frontend only shows actions that the server currently supports.

DREAM remains guest-first: no DREAM registration or password is required to use public research. Public retrieval is read-only by default. Public data, social posts, papers, repository metadata and datasets are not automatically training data. Source terms, attribution, license compatibility, privacy and consent must be checked before any retention or training.

## First private-account connector: Google/Gmail read-only

Gmail is the first account connector. It deliberately does not send, delete, label or modify mail.

- Each browser gets an opaque random guest session cookie. The cookie is HttpOnly and SameSite=Lax; it is not a DREAM user account and it is not a cross-device identity.
- Google authorization uses the authorization-code flow with PKCE (S256), state validation, the OpenID Connect email identity and only the Gmail `gmail.readonly` scope.
- Access and refresh tokens are encrypted using AES-256-GCM before they are written to the account store. Tokens are never returned to the browser or model context.
- A connected user's Gmail is queried only when that user explicitly asks a Gmail-related question. The app retrieves up to five matching messages and bounded excerpts.
- Mail content is private, untrusted input. It is injected into a separate private-account context, not sent to public research providers, memory, training or other users.
- The Plugins dialog offers account status, Connect Gmail, Ask my Gmail and Disconnect. Disconnect deletes the stored token record and asks Google to revoke the token; the UI reports only the local connection status.
- No provider password, OAuth secret, access token or API key is requested in chat.

### Required operator setup

Gmail stays unavailable until all settings below are supplied server-side and the account store is confirmed to be on durable persistent storage.

1. Create a Google Cloud project and configure the OAuth consent screen. Add the users who should test the app while the consent screen is in testing; Google's verification and publishing rules apply before broad public use.
2. Create an OAuth Client ID of type **Web application**.
3. Add the exact HTTPS redirect URI `https://YOUR-DREAM-HOST/api/account/google/callback`.
4. Store these variables only in the private deployment environment:
   - `DREAM_GOOGLE_OAUTH_CLIENT_ID`
   - `DREAM_GOOGLE_OAUTH_CLIENT_SECRET`
   - `DREAM_GOOGLE_OAUTH_REDIRECT_URI`
   - `DREAM_ACCOUNT_ENCRYPTION_KEY` — exactly 32 random bytes, encoded as 64 hex characters or base64.
   - `DREAM_ACCOUNT_STORE_PATH` — absolute file path on a durable persistent volume.
   - `DREAM_ACCOUNT_STORE_PERSISTENT=true` — set only after checking the actual volume mount and its persistence across restarts/deploys.
5. Keep the OAuth secret, encryption key and store file out of the frontend, source control, chat, logs and build output.

The integration intentionally stays disabled on Render instances without an explicitly confirmed persistent disk. An ephemeral filesystem is not safe storage for user refresh tokens. Do not set the persistence flag on a temporary disk just to make the Connect button appear.

Google account access may require Google OAuth app review, test-user allowlisting or other provider approval. Operator credentials enable the connection flow; each user must still explicitly approve Google’s permission screen.

## Remaining integrations

Google Drive, Calendar, Outlook, Dropbox, social networks, e-commerce, developer accounts, payment providers and automation services remain separately gated. Each requires its own official OAuth/API terms, minimal scopes, secure storage, revocation, quotas, and per-action confirmation for any write, purchase, message or financial action. They must not be represented as connected merely because they appear in the directory.

The public connectors currently implemented in DREAM include its existing encyclopedia, news, scholarly, biomedical and social/technical retrieval sources, plus arXiv, Semantic Scholar and conditional public GitHub repository search for coding queries. The latter returns repository metadata, not private repositories or source-code files.
