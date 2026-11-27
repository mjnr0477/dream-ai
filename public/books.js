(() => {
  "use strict";

  const $ = selector => document.querySelector(selector);
  const grid = $("#book-grid");
  const count = $("#book-count");
  const empty = $("#empty-state");
  const search = $("#book-search");
  const preferredStore = $("#preferred-store");
  const formatFilter = $("#format-filter");
  const preferenceStatus = $("#preference-status");
  const additionalGrid = $("#additional-grid");
  const additionalSection = $("#additional-section");
  const STORAGE_KEY = "mathias-books-preferred-store-v1";

  const escapeHTML = value => String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

  function safeHttpsUrl(value) {
    try {
      const url = new URL(String(value || ""));
      return url.protocol === "https:" && !url.username && !url.password ? url.href : "";
    } catch {
      return "";
    }
  }

  function readPreference() {
    try { return localStorage.getItem(STORAGE_KEY) || ""; }
    catch { return ""; }
  }

  function savePreference(value) {
    try {
      if (value) localStorage.setItem(STORAGE_KEY, value);
      else localStorage.removeItem(STORAGE_KEY);
      preferenceStatus.textContent = value
        ? "Saved on this device. Matching store links will appear first."
        : "No store preference saved. All available stores remain visible.";
    } catch {
      preferenceStatus.textContent = "This browser blocked saving. Your store choice will still work for this visit.";
    }
  }

  function retailerButton(edition, preferred) {
    const url = safeHttpsUrl(edition.url);
    if (!url) return "";
    const retailer = escapeHTML(edition.retailer || "Bookstore");
    const format = escapeHTML(edition.format || "Edition");
    return '<a class="store-link' + (preferred ? " is-preferred" : "") +
      '" href="' + escapeHTML(url) + '" target="_blank" rel="noopener noreferrer">' +
      '<span class="store-name">' + retailer + '</span>' +
      '<span class="store-format">' + format + '</span>' +
      (preferred ? '<span class="preferred-tag">Your preferred store</span>' : "") +
      '<span class="store-arrow" aria-hidden="true">↗</span></a>';
  }

  function universalButton(url, index, total) {
    const safeUrl = safeHttpsUrl(url);
    if (!safeUrl || !safeUrl.startsWith("https://books2read.com/u/")) return "";
    const label = total > 1 ? "Compare stores · link " + (index + 1) : "Compare stores on Books2Read";
    return '<a class="universal-link" href="' + escapeHTML(safeUrl) +
      '" target="_blank" rel="noopener noreferrer">' + label + ' <span aria-hidden="true">↗</span></a>';
  }

  function renderBook(book, preferred) {
    const editions = Array.isArray(book.editions) ? book.editions.filter(e => safeHttpsUrl(e.url)) : [];
    const universalUrls = Array.isArray(book.universalUrls) ? book.universalUrls : (book.universalUrl ? [book.universalUrl] : []);
    const preferredEditions = preferred ? editions.filter(e => e.retailer === preferred) : [];
    const formats = [...new Set(editions.map(e => e.format).filter(Boolean))];
    const imageUrl = safeHttpsUrl(book.coverUrl);
    const image = imageUrl
      ? '<img class="book-cover-image" src="' + escapeHTML(imageUrl) + '" alt="Cover of ' + escapeHTML(book.title) + '" loading="lazy" referrerpolicy="no-referrer" onerror="this.closest(\'.book-cover\').classList.add(\'cover-failed\');this.remove()">'
      : "";
    const cover = '<div class="book-cover' + (imageUrl ? "" : " no-image") + '">' +
      (image || '<span>' + escapeHTML(book.category || "BOOK") + '</span>') + '</div>';
    const preferredMarkup = preferredEditions.length
      ? '<div class="preferred-options"><p class="mini-heading">Available from your preferred store</p><div class="store-links">' +
        preferredEditions.map(e => retailerButton(e, true)).join("") + '</div></div>'
      : (preferred && universalUrls.length
        ? '<p class="preference-hint">No direct ' + escapeHTML(preferred) + ' listing is recorded here. Open Books2Read to check its retailer choices.</p>'
        : (preferred
          ? '<p class="preference-hint">This title has no direct ' + escapeHTML(preferred) + ' listing in the catalogue. Other stores are available below.</p>'
          : ""));
    const universalMarkup = universalUrls.map((url, i) => universalButton(url, i, universalUrls.length)).filter(Boolean).join("");
    const storesMarkup = editions.length
      ? '<details class="store-details"><summary>All ' + editions.length + ' store links and editions</summary>' +
        '<div class="store-links">' + editions.map(e => retailerButton(e, Boolean(preferred && e.retailer === preferred))).join("") + '</div></details>'
      : "";
    const note = book.retailerNote ? '<p class="retailer-note">' + escapeHTML(book.retailerNote) + '</p>' : "";
    const formatMarkup = formats.map(format => '<span class="format-chip">' + escapeHTML(format) + '</span>').join("");
    return '<article class="book-card">' + cover + '<div class="book-main">' +
      '<div class="book-meta"><span>' + escapeHTML(book.category || "Book") + '</span><span>' +
      formats.length + ' format' + (formats.length === 1 ? "" : "s") + '</span></div>' +
      '<h3>' + escapeHTML(book.title) + '</h3>' +
      '<p class="book-description">' + escapeHTML(book.description || "Open a store listing to see more about this title.") + '</p>' +
      '<div class="format-list">' + formatMarkup + '</div>' + preferredMarkup +
      (universalMarkup ? '<div class="universal-links">' + universalMarkup + '</div>' : "") +
      storesMarkup + note + '</div></article>';
  }

  function renderAdditionalLink(link) {
    const url = safeHttpsUrl(link.url);
    if (!url || !url.startsWith("https://books2read.com/u/")) return "";
    return '<article class="additional-card"><div class="additional-icon" aria-hidden="true">↗</div>' +
      '<div><h3>' + escapeHTML(link.title || "Additional book link") + '</h3><p>' +
      escapeHTML(link.description || "Open this universal book page to see its retailer options.") +
      '</p></div><a href="' + escapeHTML(url) + '" target="_blank" rel="noopener noreferrer">Open book page <span aria-hidden="true">↗</span></a></article>';
  }

  let books = [];
  let preferred = "";

  function applyFilters() {
    const query = search.value.trim().toLocaleLowerCase();
    const format = formatFilter.value;
    const filtered = books.filter(book => {
      const editions = Array.isArray(book.editions) ? book.editions : [];
      const searchable = [
        book.title, book.category, book.description,
        ...editions.flatMap(e => [e.retailer, e.format])
      ].join(" ").toLocaleLowerCase();
      const matchesQuery = !query || searchable.includes(query);
      const matchesFormat = format === "all" || editions.some(e => e.format === format);
      return matchesQuery && matchesFormat;
    });
    if (preferred) {
      filtered.sort((a, b) => {
        const aMatch = (a.editions || []).some(e => e.retailer === preferred) ? 1 : 0;
        const bMatch = (b.editions || []).some(e => e.retailer === preferred) ? 1 : 0;
        return bMatch - aMatch;
      });
    }
    grid.innerHTML = filtered.map(book => renderBook(book, preferred)).join("");
    count.textContent = filtered.length + " of " + books.length + " books";
    empty.hidden = filtered.length !== 0;
  }

  function initializeFilters() {
    const stores = [...new Set(books.flatMap(book => (book.editions || []).map(e => e.retailer).filter(Boolean)))].sort((a, b) => a.localeCompare(b));
    stores.forEach(store => {
      const option = document.createElement("option");
      option.value = store;
      option.textContent = store;
      preferredStore.append(option);
    });
    const formats = [...new Set(books.flatMap(book => (book.editions || []).map(e => e.format).filter(Boolean)))].sort((a, b) => a.localeCompare(b));
    formats.forEach(format => {
      const option = document.createElement("option");
      option.value = format;
      option.textContent = format;
      formatFilter.append(option);
    });
    const stored = readPreference();
    preferred = stores.includes(stored) ? stored : "";
    preferredStore.value = preferred;
    if (stored && !preferred) savePreference("");
    else savePreference(preferred);
  }

  async function start() {
    try {
      const response = await fetch("/books-catalogue.json", {
        cache: "no-cache",
        headers: { "accept": "application/json" }
      });
      if (!response.ok) throw new Error("Book catalogue returned HTTP " + response.status);
      const data = await response.json();
      if (!data || !Array.isArray(data.books) || !data.books.length) throw new Error("Book catalogue data is incomplete.");
      books = data.books.filter(book => book && typeof book.title === "string" && Array.isArray(book.editions));
      if (!books.length) throw new Error("No valid book entries were found.");
      initializeFilters();
      applyFilters();

      const additional = Array.isArray(data.additionalLinks) ? data.additionalLinks : [];
      additionalGrid.innerHTML = additional.map(renderAdditionalLink).filter(Boolean).join("");
      additionalSection.hidden = !additionalGrid.children.length;

      search.addEventListener("input", applyFilters);
      formatFilter.addEventListener("change", applyFilters);
      preferredStore.addEventListener("change", () => {
        preferred = preferredStore.value;
        savePreference(preferred);
        applyFilters();
      });
    } catch (error) {
      console.error("Books catalogue could not be loaded:", error);
      grid.innerHTML = '<p class="load-error">The book collection could not be loaded. Check your connection and refresh the page.</p>';
      count.textContent = "Catalogue unavailable";
      empty.hidden = true;
    }
  }

  const backLink = $("#back-to-app");
  backLink && backLink.addEventListener("click", event => {
    if (document.referrer && history.length > 1) {
      event.preventDefault();
      history.back();
    }
  });

  start();
})();
