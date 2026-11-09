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

## Meta account connections: Facebook, Instagram and Threads

The Meta connector uses separate, official sign-in flows for Facebook/Instagram and Threads. DREAM does not ask users for Meta passwords or tokens.

- Configure `META_APP_ID` and `META_APP_SECRET` privately in Render. Register these exact HTTPS callback URLs in the Meta developer app:
  - `https://YOUR-DREAM-HOST/api/account/meta/facebook/callback`
  - `https://YOUR-DREAM-HOST/api/account/meta/threads/callback`
- Set `META_FACEBOOK_REDIRECT_URI` and `META_THREADS_REDIRECT_URI` to the matching URLs. Facebook Login uses `META_APP_ID` and `META_APP_SECRET` and requests `public_profile`, `pages_show_list`, `pages_read_engagement` and `instagram_basic`. Threads uses its own Threads Login credentials (`THREADS_APP_ID` and `THREADS_APP_SECRET`); when those are not supplied it falls back to the Meta app credentials. Threads requests `threads_basic`. Configure the Threads use case and its own callback in Meta's app dashboard.
- Each user's Facebook/Instagram and Threads authorizations are separate. The returned tokens are encrypted with AES-256-GCM and kept in Secure, HttpOnly, SameSite=Lax cookies for that browser; tokens are never returned to frontend JavaScript.
- A connected browser can ask DREAM to read the authorized Facebook profile and accessible Pages, discover Instagram professional accounts linked to those Pages, or read the connected Threads profile. Instagram consumer accounts are not supported by the Facebook Login route.
- This first Meta release is read-only. It does not publish posts, send messages, reply to comments, edit Pages, read a personal Facebook feed or perform financial actions. Do not claim those actions are available.
- Private account details are retrieved only when the same browser explicitly asks a Meta-account question. They are kept out of public research, shared memory, training submissions and other users' sessions. Because DREAM is guest-first, these connections belong to the browser rather than a DREAM user account; disconnect before another person uses a shared device. Disconnect clears DREAM's local cookie; revoke the app separately in Meta's account settings to remove the provider grant.
- `META_ACCESS_TOKEN` is optional and is checked only as an operator-side diagnostic. It is never treated as an end-user account and is never shared between visitors. It does not replace `META_APP_ID` or `META_APP_SECRET`.

### Meta approval and production gate

Meta's permission screen is user consent, not proof of general public approval. The app must be configured with the Facebook Login, Instagram Platform and Threads products, exact redirect URIs, privacy/data-deletion information and applicable review submissions. Advanced access, business verification or App Review may be required before accounts outside app roles can grant the requested permissions. Keep the directory honest about any permission Meta has not approved. Read the [Meta developer documentation](https://developers.meta.com/) and [Meta Permissions Reference](https://developers.facebook.com/docs/permissions/).


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

**Important public-launch gate:** Google's current Gmail scope reference classifies `https://www.googleapis.com/auth/gmail.readonly` as a **restricted scope**. A public app requesting it must complete Google's restricted-scope verification unless an exception applies. Because DREAM stores refresh tokens and receives Gmail message data on its server, Google may also require an independent security assessment by a Google-approved assessor. This is a provider approval requirement, not something a code change or an operator API key can bypass. Keep the connection disabled for general public users until Google confirms the required verification and assessment are complete. See [Google's Gmail scope classifications](https://developers.google.com/workspace/gmail/api/auth/scopes) and [restricted-scope verification](https://developers.google.com/identity/protocols/oauth2/production-readiness/restricted-scope-verification).

## Remaining integrations

Google Drive, Calendar, Outlook, Dropbox, social networks, e-commerce, developer accounts, payment providers and automation services remain separately gated. Each requires its own official OAuth/API terms, minimal scopes, secure storage, revocation, quotas, and per-action confirmation for any write, purchase, message or financial action. They must not be represented as connected merely because they appear in the directory.

The public connectors currently implemented in DREAM include its existing encyclopedia, news, scholarly, biomedical and social/technical retrieval sources, plus arXiv, Semantic Scholar and conditional public GitHub repository search for coding queries. The latter returns repository metadata, not private repositories or source-code files.
