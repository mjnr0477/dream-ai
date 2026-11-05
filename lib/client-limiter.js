/** Limit concurrent inference/research work per trusted client key. */
export class ClientConcurrencyLimiter {
  #active = new Map();

  constructor(maxActivePerClient = 1) {
    if (!Number.isInteger(maxActivePerClient) || maxActivePerClient < 1) {
      throw new TypeError("maxActivePerClient must be a positive integer");
    }
    this.maxActivePerClient = maxActivePerClient;
  }

  acquire(clientKey) {
    const key = typeof clientKey === "string" ? clientKey.trim() : "";
    // If the trusted proxy did not provide an address, rely on the global
    // concurrency cap and shared unknown-client rate bucket instead of
    // accidentally locking all anonymous requests behind one fake identity.
    if (!key || key === "unknown") return { allowed: true, release: () => {} };

    const active = this.#active.get(key) || 0;
    if (active >= this.maxActivePerClient) return { allowed: false, release: () => {} };
    this.#active.set(key, active + 1);
    let released = false;
    return {
      allowed: true,
      release: () => {
        if (released) return;
        released = true;
        const current = this.#active.get(key) || 0;
        if (current <= 1) this.#active.delete(key);
        else this.#active.set(key, current - 1);
      }
    };
  }
}
