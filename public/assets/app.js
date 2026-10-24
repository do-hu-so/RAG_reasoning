/* =========================================================================
   RAG Reasoning — logic giao diện chat
   - Lịch sử cuộc trò chuyện lưu trong localStorage của trình duyệt
   - Tài liệu PDF lưu trên server ("dataset web"), theo từng cuộc trò chuyện
   - 4 phương pháp trả lời (01 LLM thuần, 02 RAG thuần, 03 RAG + Reranking,
     04 RAG + Reasoning) và chế độ so sánh cả 4 để đánh giá
   ========================================================================= */
(function () {
  "use strict";

  var STORAGE_KEY = "rag.conversations.v1";
  var CURRENT_KEY = "rag.current";
  var THEME_KEY = "rag.theme";
  var MODE_KEY = "rag.mode";
  var HISTORY_TURNS = 4;
  var MAX_CONVERSATIONS = 40;
  var DEFAULT_TITLE = "Cuộc trò chuyện mới";
  var DEFAULT_MODE = "compare";
  var WIDE_SCREEN = 1200; // từ độ rộng này hộp trích dẫn tự mở cạnh câu trả lời

  var MODES = {
    llm: { num: "01", title: "LLM thuần, không RAG", short: "LLM thuần", desc: "Chỉ hỏi LLM, không dùng tài liệu." },
    rag: { num: "02", title: "LLM + RAG thuần", short: "RAG thuần", desc: "Tìm các đoạn liên quan rồi đưa thẳng cho LLM." },
    rerank: { num: "03", title: "LLM + RAG + Reranking", short: "RAG + Rerank", desc: "Tìm nhiều đoạn, xếp hạng lại, giữ các đoạn tốt nhất." },
    reasoning: { num: "04", title: "LLM + RAG + Reasoning", short: "RAG + Reasoning", desc: "Phân tích câu hỏi, suy luận có trích dẫn và kiểm chứng." },
    compare: { num: "⇄", title: "So sánh cả 4 phương pháp", short: "So sánh 4", desc: "Mỗi câu hỏi chạy lần lượt cả 4 cách, hiện cạnh nhau để đánh giá." }
  };
  var MODE_ORDER = ["llm", "rag", "rerank", "reasoning"];

  var STAGE_LABELS = {
    analyzing: "Phân tích câu hỏi",
    retrieving: "Tìm đoạn liên quan",
    reranking: "Xếp hạng lại (rerank)",
    generating: "Sinh câu trả lời",
    reasoning: "Suy luận & kiểm chứng"
  };
  var STAGE_ORDER = ["analyzing", "retrieving", "reranking", "generating", "reasoning"];
  var WAIT_LABEL = "Chờ hạn mức Groq (không tính vào tổng)";
  var WAIT_HINT = "Thời gian chờ Groq cấp lại hạn mức token/phút để giữ cùng một model; không tính vào thời gian phản hồi.";

  var RATINGS = [
    { key: "correct", label: "Đúng", icon: "✓" },
    { key: "partial", label: "Một phần", icon: "≈" },
    { key: "wrong", label: "Sai", icon: "✗" }
  ];

  var ICONS = {
    logo: '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M12 2 3 7v10l9 5 9-5V7l-9-5Zm0 2.3 6.8 3.8L12 11.9 5.2 8.1 12 4.3ZM5 9.8l6 3.4v6.6l-6-3.4V9.8Zm8 10V13.2l6-3.4v6.6l-6 3.4Z"/></svg>',
    trash: '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M9 3h6l1 2h4v2H4V5h4l1-2ZM6 9h12l-1 12H7L6 9Zm4 2v8h1.5v-8H10Zm3.5 0v8H15v-8h-1.5Z"/></svg>',
    open: '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M14 3h7v7h-2V6.4l-9.3 9.3-1.4-1.4L17.6 5H14V3ZM5 5h6v2H5v12h12v-6h2v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z"/></svg>',
    copy: '<svg viewBox="0 0 24 24" width="14" height="14"><path fill="currentColor" d="M8 3h11a2 2 0 0 1 2 2v11h-2V5H8V3ZM5 7h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Zm0 2v10h10V9H5Z"/></svg>',
    upload: '<svg viewBox="0 0 24 24" width="36" height="36"><path fill="currentColor" d="M6 2h8l6 6v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm7 1.5V9h5.5L13 3.5ZM11 19h2v-4h3l-4-4-4 4h3v4Z"/></svg>',
    clock: '<svg viewBox="0 0 24 24" width="13" height="13"><path fill="currentColor" d="M12 2a10 10 0 1 1 0 20 10 10 0 0 1 0-20Zm0 2a8 8 0 1 0 0 16 8 8 0 0 0 0-16Zm1 3v4.6l3.4 2-1 1.7L11 12.7V7h2Z"/></svg>',
    check: '<svg viewBox="0 0 24 24" width="13" height="13"><path fill="currentColor" d="m9.5 16.2-4.2-4.2-1.4 1.4 5.6 5.6L20.1 8.4 18.7 7 9.5 16.2Z"/></svg>',
    quote: '<svg viewBox="0 0 24 24" width="14" height="14"><path fill="currentColor" d="M5 5h6v6H8.2c0 2 1 3.4 2.8 4l-.8 1.9C7 15.9 5 13.6 5 10V5Zm8 0h6v6h-2.8c0 2 1 3.4 2.8 4l-.8 1.9C15 15.9 13 13.6 13 10V5Z"/></svg>',
    moon: '<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg>',
    sun: '<svg viewBox="0 0 24 24" width="18" height="18"><circle cx="12" cy="12" r="4.5" fill="currentColor"/><path stroke="currentColor" stroke-width="2" stroke-linecap="round" d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
  };

  // ------------------------------------------------------------------ DOM
  function $(id) { return document.getElementById(id); }

  var els = {
    app: $("app"),
    convList: $("conv-list"),
    newChat: $("new-chat"),
    openSidebar: $("open-sidebar"),
    closeSidebar: $("close-sidebar"),
    backdrop: $("sidebar-backdrop"),
    convTitle: $("conv-title"),
    docsBtn: $("docs-btn"),
    docsCount: $("docs-count"),
    exportBtn: $("export-btn"),
    banner: $("banner"),
    main: $("main"),
    messages: $("messages"),
    inner: $("messages-inner"),
    composer: $("composer"),
    input: $("input"),
    sendBtn: $("send-btn"),
    attachBtn: $("attach-btn"),
    fileInput: $("file-input"),
    modePicker: $("mode-picker"),
    modeBtn: $("mode-btn"),
    modeNum: $("mode-num"),
    modeLabel: $("mode-label"),
    modeMenu: $("mode-menu"),
    dropOverlay: $("drop-overlay"),
    uploadHint: $("upload-hint"),
    docsPanel: $("docs-panel"),
    docsList: $("docs-list"),
    closeDocs: $("close-docs"),
    docsUpload: $("docs-upload"),
    serverStatus: $("server-status"),
    themeToggle: $("theme-toggle"),
    citePanel: $("cite-panel"),
    citeQuestion: $("cite-panel-question"),
    citeMeta: $("cite-panel-meta"),
    citeBody: $("cite-panel-body"),
    closeCite: $("close-cite")
  };

  /** Tạo phần tử DOM; chuỗi con luôn được chèn dạng text (an toàn với XSS). */
  function h(tag, props) {
    var node = document.createElement(tag);
    var key;
    props = props || {};
    for (key in props) {
      if (!Object.prototype.hasOwnProperty.call(props, key)) continue;
      var value = props[key];
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "html") node.innerHTML = value; // chỉ dùng cho chuỗi tin cậy (icon, markdown đã lọc)
      else if (key.indexOf("on") === 0) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (var i = 2; i < arguments.length; i++) appendChild(node, arguments[i]);
    return node;
  }

  function appendChild(node, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { appendChild(node, c); }); return; }
    node.append(child instanceof Node ? child : String(child));
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function truncate(text, length) {
    text = String(text).replace(/\s+/g, " ").trim();
    return text.length > length ? text.slice(0, length - 1) + "…" : text;
  }

  /** 3.456 -> "3,5 s"; 75 -> "1 phút 15 s" */
  function formatSeconds(seconds) {
    seconds = Math.max(0, Number(seconds) || 0);
    if (seconds >= 60) return Math.floor(seconds / 60) + " phút " + Math.round(seconds % 60) + " s";
    return seconds.toFixed(1).replace(".", ",") + " s";
  }

  function formatSize(bytes) {
    if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
    return Math.max(1, Math.round(bytes / 1024)) + " KB";
  }

  function formatNumber(value) {
    return Number(value || 0).toLocaleString("vi-VN");
  }

  /** "groq/openai/gpt-oss-120b" -> "gpt-oss-120b"; "groq/openai/gpt-oss-20b#2" -> "gpt-oss-20b (key 2)" */
  function shortModel(name) {
    var parts = String(name || "").split("#");
    var model = parts[0].replace(/^groq\//, "").replace(/^openai\//, "");
    return parts[1] ? model + " (key " + parts[1] + ")" : model;
  }

  function newId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "");
    var bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.prototype.map.call(bytes, function (b) { return ("0" + b.toString(16)).slice(-2); }).join("");
  }

  // ------------------------------------------------------------------ state
  var state = {
    conversations: loadConversations(),
    currentId: null,
    docs: [],
    busy: false,
    uploads: {},
    config: { max_upload_mb: 4 },
    mode: loadMode(),
    cite: { msgId: null, mode: null, focus: null }
  };

  function loadConversations() {
    try {
      var list = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(list) ? list : [];
    } catch (e) {
      return [];
    }
  }

  function loadMode() {
    var mode = null;
    try { mode = localStorage.getItem(MODE_KEY); } catch (e) { mode = null; }
    return MODES[mode] ? mode : DEFAULT_MODE;
  }

  function saveConversations() {
    state.conversations.sort(function (a, b) { return b.updatedAt - a.updatedAt; });
    state.conversations = state.conversations.slice(0, MAX_CONVERSATIONS);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations));
    } catch (e) {
      // Hết dung lượng localStorage: vẫn chạy tiếp, chỉ không lưu được lịch sử
    }
  }

  function current() {
    for (var i = 0; i < state.conversations.length; i++) {
      if (state.conversations[i].id === state.currentId) return state.conversations[i];
    }
    return null;
  }

  /** Cuộc trò chuyện mới chỉ được lưu khi có tin nhắn / tài liệu đầu tiên. */
  function ensureConversation() {
    var conv = current();
    if (!conv) {
      conv = { id: state.currentId || newId(), title: DEFAULT_TITLE, createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
      state.currentId = conv.id;
      state.conversations.unshift(conv);
      rememberCurrent();
    }
    return conv;
  }

  function touch(conv) {
    conv.updatedAt = Date.now();
    saveConversations();
    renderConversationList();
    if (conv.id === state.currentId) els.convTitle.textContent = conv.title;
  }

  function rememberCurrent() {
    try { localStorage.setItem(CURRENT_KEY, state.currentId); } catch (e) { /* bỏ qua */ }
  }

  function totalPendingUploads() { return Object.keys(state.uploads).length; }

  /**
   * Tin nhắn trả lời gồm kết quả theo từng phương pháp: msg.modes (thứ tự) và
   * msg.runs[mode] = { status, stages, result, error, totalSeconds, rating, ... }.
   * Tin nhắn cũ (trước khi có nhiều phương pháp) được chuyển thành phương pháp 04.
   */
  function normalizeAssistant(msg) {
    if (msg.role !== "assistant" || msg.runs) return;
    msg.mode = "reasoning";
    msg.modes = ["reasoning"];
    msg.runs = {
      reasoning: {
        status: msg.status === "pending" ? "error" : msg.status,
        result: msg.result,
        error: msg.error || (msg.status === "pending" ? "Câu trả lời bị gián đoạn." : undefined),
        totalSeconds: msg.totalSeconds || msg.duration || null
      }
    };
    msg.status = "done";
    delete msg.result;
    delete msg.error;
    delete msg.duration;
  }

  // ------------------------------------------------------------------ API
  function api(path) { return "/api/conversations/" + encodeURIComponent(state.currentId) + path; }

  function readError(response) {
    return response.json().then(function (data) {
      if (typeof data.detail === "string") return data.detail;
      if (Array.isArray(data.detail)) return data.detail.map(function (d) { return d.msg; }).join("; ");
      return "Lỗi " + response.status;
    }, function () {
      return "Lỗi " + response.status;
    });
  }

  async function streamAsk(conversationId, question, history, mode, onEvent) {
    var response = await fetch("/api/conversations/" + encodeURIComponent(conversationId) + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, history: history, mode: mode })
    });
    if (!response.ok) throw new Error(await readError(response));

    var handleLine = function (line) {
      line = line.trim();
      if (line) onEvent(JSON.parse(line));
    };

    if (!response.body || !response.body.getReader) {
      (await response.text()).split("\n").forEach(handleLine);
      return;
    }

    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      var newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        handleLine(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
      }
    }
    buffer += decoder.decode();
    handleLine(buffer);
  }

  // ------------------------------------------------------------------ sidebar
  function renderConversationList() {
    els.convList.textContent = "";
    if (!state.conversations.length) {
      els.convList.append(h("div", { class: "conv-empty" }, "Chưa có cuộc trò chuyện nào"));
      return;
    }
    state.conversations.forEach(function (conv) {
      els.convList.append(h("div", { class: "conv-item" + (conv.id === state.currentId ? " active" : "") },
        h("button", { class: "conv-link", type: "button", title: conv.title, onclick: function () { openConversation(conv.id); } }, conv.title),
        h("button", {
          class: "icon-btn conv-delete", type: "button", "aria-label": "Xoá cuộc trò chuyện", title: "Xoá",
          html: ICONS.trash, onclick: function () { deleteConversation(conv.id); }
        })
      ));
    });
  }

  function openConversation(id) {
    closeCitePanel();
    state.currentId = id;
    state.docs = [];
    rememberCurrent();
    renderAll();
    refreshDocs();
    closeSidebarMobile();
  }

  function startNewConversation() {
    closeCitePanel();
    state.currentId = newId();
    state.docs = [];
    renderAll();
    closeSidebarMobile();
    els.input.focus();
  }

  async function deleteConversation(id) {
    var conv = state.conversations.filter(function (c) { return c.id === id; })[0];
    if (!conv || !confirm('Xoá cuộc trò chuyện "' + conv.title + '" cùng các tài liệu đã tải lên?')) return;
    try {
      await fetch("/api/conversations/" + encodeURIComponent(id), { method: "DELETE" });
    } catch (e) {
      // Không xoá được trên server thì vẫn xoá khỏi trình duyệt
    }
    state.conversations = state.conversations.filter(function (c) { return c.id !== id; });
    saveConversations();
    if (state.currentId === id) startNewConversation();
    else renderConversationList();
  }

  function closeSidebarMobile() { els.app.classList.remove("sidebar-open"); }

  // ------------------------------------------------------------------ chọn phương pháp
  function setMode(mode) {
    if (!MODES[mode]) return;
    state.mode = mode;
    try { localStorage.setItem(MODE_KEY, mode); } catch (e) { /* bỏ qua */ }
    renderModePicker();
    var conv = current();
    if (!conv || !conv.messages.length) renderMessages();
  }

  function modeMenuItem(mode) {
    var info = MODES[mode];
    var active = state.mode === mode;
    return h("button", {
      type: "button", role: "menuitemradio", "aria-checked": active ? "true" : "false",
      class: "mode-item" + (active ? " active" : "") + (mode === "compare" ? " compare" : ""),
      onclick: function () {
        setMode(mode);
        closeModeMenu();
        els.input.focus();
      }
    },
      h("span", { class: "mode-num" }, info.num),
      h("span", { class: "mode-item-text" }, h("strong", {}, info.title), h("small", {}, info.desc)),
      active ? h("span", { class: "mode-check", html: ICONS.check }) : null
    );
  }

  function renderModePicker() {
    var info = MODES[state.mode];
    els.modeNum.textContent = info.num;
    els.modeLabel.textContent = info.short;
    els.modeBtn.classList.toggle("is-compare", state.mode === "compare");
    els.modeBtn.title = "Phương pháp: " + info.title;
    els.input.placeholder = state.mode === "compare" ? "Hỏi để so sánh 4 cách…" : "Hỏi về tài liệu của bạn…";
    els.modeMenu.textContent = "";
    els.modeMenu.append(h("div", { class: "mode-menu-title" }, "Phương pháp trả lời"));
    MODE_ORDER.forEach(function (mode) { els.modeMenu.append(modeMenuItem(mode)); });
    els.modeMenu.append(h("div", { class: "mode-menu-sep" }), modeMenuItem("compare"));
  }

  function openModeMenu() {
    els.modeMenu.hidden = false;
    els.modeBtn.setAttribute("aria-expanded", "true");
  }

  function closeModeMenu() {
    els.modeMenu.hidden = true;
    els.modeBtn.setAttribute("aria-expanded", "false");
  }

  function modeTag(mode, short) {
    var info = MODES[mode] || MODES.reasoning;
    return h("span", { class: "mode-tag" + (short ? " short" : "") },
      h("span", { class: "mode-num" }, info.num),
      h("span", { class: "mode-title" }, short ? info.short : info.title)
    );
  }

  // ------------------------------------------------------------------ messages
  function renderAll() {
    var conv = current();
    els.convTitle.textContent = conv ? conv.title : DEFAULT_TITLE;
    renderConversationList();
    renderMessages();
    renderDocs();
    updateComposer();
  }

  function renderMessages() {
    var conv = current();
    var messages = conv ? conv.messages : [];
    els.inner.textContent = "";
    if (!messages.length) els.inner.append(renderEmptyState());
    messages.forEach(function (msg) { els.inner.append(renderMessage(msg)); });
    scrollToBottom(false);
  }

  function renderMessage(msg) {
    var node;
    if (msg.role === "user") node = renderUserMessage(msg);
    else if (msg.role === "doc") node = renderDocMessage(msg);
    else node = renderAssistantMessage(msg);
    node.setAttribute("data-id", msg.id);
    return node;
  }

  function updateMessage(msg) {
    var old = els.inner.querySelector('[data-id="' + msg.id + '"]');
    if (!old) return;
    var fresh = renderMessage(msg);
    // Giữ trạng thái mở / đóng của các khối details khi cập nhật
    var openDetails = Array.prototype.map.call(old.querySelectorAll("details[open]"), function (d) { return d.getAttribute("data-key"); });
    openDetails.forEach(function (key) {
      var target = key && fresh.querySelector('details[data-key="' + key + '"]');
      if (target) target.open = true;
    });
    old.replaceWith(fresh);
  }

  function scrollToBottom(smooth) {
    var box = els.messages;
    if (smooth === false) box.style.scrollBehavior = "auto";
    box.scrollTop = box.scrollHeight;
    box.style.scrollBehavior = "";
  }

  function renderEmptyState() {
    var maxMb = state.config.max_upload_mb || 4;
    var modeRow = function (mode) {
      var info = MODES[mode];
      return h("button", {
        type: "button", class: "mode-row" + (state.mode === mode ? " active" : "") + (mode === "compare" ? " compare" : ""),
        onclick: function () { setMode(mode); }
      },
        h("span", { class: "mode-num" }, info.num),
        h("span", { class: "mode-row-text" }, h("strong", {}, info.title), h("small", {}, info.desc))
      );
    };
    return h("div", { class: "empty" },
      h("h2", {}, "Hỏi đáp tài liệu PDF với RAG reasoning"),
      h("p", {}, "Tải PDF lên rồi đặt câu hỏi. Có thể so sánh 4 phương pháp trả lời ngay trên màn hình để đánh giá."),
      h("button", { class: "dropzone", type: "button", onclick: function () { els.fileInput.click(); } },
        h("span", { html: ICONS.upload }),
        h("strong", {}, "Kéo thả hoặc bấm để chọn file PDF"),
        h("span", {}, "Tối đa " + maxMb + " MB mỗi file · PDF có lớp chữ (không phải bản scan)")
      ),
      h("div", { class: "mode-intro" },
        h("p", { class: "mode-intro-title" }, "Chọn phương pháp trả lời"),
        h("div", { class: "mode-list" }, MODE_ORDER.map(modeRow)),
        modeRow("compare")
      )
    );
  }

  function renderUserMessage(msg) {
    return h("div", { class: "msg user" }, h("div", { class: "bubble" }, msg.text));
  }

  function renderDocMessage(msg) {
    var sub, subClass = "doc-sub", progress = null;
    if (msg.status === "uploading") {
      sub = "Đang tải lên… " + (msg.progress || 0) + "%";
      progress = h("div", { class: "progress" }, h("span", { style: "width:" + (msg.progress || 0) + "%" }));
    } else if (msg.status === "processing") {
      sub = "Đang đọc PDF và chuẩn bị tìm kiếm…";
      progress = h("div", { class: "progress indeterminate" }, h("span", {}));
    } else if (msg.status === "ready") {
      var meta = msg.meta || {};
      sub = "Sẵn sàng · " + meta.pages + " trang · " + meta.chunks + " đoạn · " + formatSize(meta.size || msg.size) +
        (meta.search === "keyword" ? " · tìm theo từ khoá" : "");
      subClass += " ok";
    } else {
      sub = msg.error || "Tải lên thất bại.";
      subClass += " bad";
    }

    var body = [];
    if (msg.status === "ready" && msg.meta && msg.meta.warning) {
      body.push(h("p", { class: "doc-warning" }, "⚠ " + msg.meta.warning));
    }
    if (msg.status === "ready") {
      if (msg.overviewStatus === "loading") {
        body.push(h("p", { class: "doc-summary loading shimmer" }, "Đang đọc lướt và tóm tắt tài liệu…"));
      } else if (msg.overviewStatus === "ready" && msg.overview) {
        if (msg.overview.summary) body.push(h("p", { class: "doc-summary" }, msg.overview.summary));
        if (msg.overview.questions && msg.overview.questions.length) {
          body.push(h("div", { class: "suggestions" }, msg.overview.questions.map(function (q) {
            return h("button", { class: "suggestion", type: "button", onclick: function () { submitQuestion(q); } }, q);
          })));
        }
      } else if (msg.overviewStatus === "error") {
        body.push(h("p", { class: "doc-summary" }, "Tài liệu đã sẵn sàng, bạn có thể đặt câu hỏi ngay."));
      }
    }

    return h("div", { class: "msg doc" },
      h("div", { class: "doc-card" },
        h("div", { class: "doc-head" },
          h("div", { class: "doc-icon", "aria-hidden": "true" }, "PDF"),
          h("div", { class: "doc-info" },
            h("div", { class: "doc-name", title: msg.filename }, msg.filename),
            h("div", { class: subClass }, sub)
          )
        ),
        progress,
        body
      )
    );
  }

  var CONFIDENCE_CLASS = { "cao": "cao", "trung bình": "trung-binh", "thấp": "thap" };

  function renderMarkdown(text, sourceCount) {
    var html;
    if (window.marked && window.DOMPurify) {
      html = DOMPurify.sanitize(marked.parse(String(text), { breaks: true, gfm: true }));
    } else {
      html = escapeHtml(text).replace(/\n/g, "<br>");
    }
    // Đổi [n] thành chip bấm được để xem nguồn
    return html.replace(/\[(\d{1,3})\]/g, function (match, n) {
      var index = Number(n);
      if (index < 1 || index > sourceCount) return match;
      return '<button type="button" class="cite" data-cite="' + index + '" title="Xem nguồn ' + index + '">' + index + "</button>";
    });
  }

  // ------------------------------------------------------------------ câu trả lời
  function renderAssistantMessage(msg) {
    if (msg.modes.length > 1) return renderCompareMessage(msg);

    var mode = msg.modes[0];
    var run = msg.runs[mode] || {};
    return h("div", { class: "msg assistant" + (state.cite.msgId === msg.id ? " citing" : "") },
      h("div", { class: "avatar", html: ICONS.logo }),
      h("div", { class: "assistant-body" },
        h("div", { class: "mode-header" }, modeTag(mode)),
        renderRunBody(msg, mode, run, false)
      )
    );
  }

  function renderRunBody(msg, mode, run, compact) {
    if (run.status === "waiting") return h("p", { class: "mode-waiting" }, "Đang chờ tới lượt…");
    if (run.status === "pending") return renderLiveStatus(run);
    if (run.status === "error") return h("div", { class: "error-box" }, run.error || "Đã xảy ra lỗi.");
    return renderResult(msg, mode, run, compact);
  }

  /** Thời gian đã trôi qua của một bước (giây), tính theo đồng hồ trình duyệt. */
  function stageSeconds(run, key) {
    var times = run.stageTimes && run.stageTimes[key];
    if (!times) return null;
    return ((times.end || Date.now()) - times.start) / 1000;
  }

  function renderLiveStatus(run) {
    var list = null;
    if (run.stages && run.stages.length && run.stageTimes) {
      list = h("ol", { class: "stage-list" }, run.stages.map(function (stage) {
        var times = run.stageTimes[stage.key];
        var status = !times ? "waiting" : times.end ? "done" : "active";
        var icon = status === "done" ? h("span", { class: "stage-icon", html: ICONS.check })
          : status === "active" ? h("span", { class: "stage-icon" }, h("span", { class: "spinner" }))
          : h("span", { class: "stage-icon" }, "○");
        return h("li", { class: status, "data-stage": stage.key },
          icon,
          h("span", {}, stage.label),
          h("span", { class: "stage-time" }, times ? formatSeconds(stageSeconds(run, stage.key)) : "")
        );
      }));
    }
    return h("div", { class: "live-box" },
      h("div", { class: "live-head" },
        h("span", { class: "pulse" }),
        h("span", { class: "shimmer" }, (run.stage || "Đang xử lý") + "…"),
        h("span", { class: "live-total", title: "Thời gian phản hồi" }, formatSeconds((Date.now() - run.startedAt) / 1000))
      ),
      list
    );
  }

  /** Cập nhật đồng hồ tại chỗ (không vẽ lại cả tin nhắn để hiệu ứng không bị giật). */
  function tick(msg) {
    var node = els.inner.querySelector('[data-id="' + msg.id + '"]');
    if (!node) return;
    var total = node.querySelector(".compare-live-total");
    if (total) total.textContent = formatSeconds((Date.now() - msg.startedAt) / 1000);
    msg.modes.forEach(function (mode) {
      var run = msg.runs[mode];
      if (!run || run.status !== "pending") return;
      var scope = msg.modes.length > 1 ? node.querySelector('.mode-card[data-mode="' + mode + '"]') : node;
      if (!scope) return;
      var live = scope.querySelector(".live-total");
      if (live) live.textContent = formatSeconds((Date.now() - run.startedAt) / 1000);
      if (run.activeStage) {
        var cell = scope.querySelector('.stage-list li[data-stage="' + run.activeStage + '"] .stage-time');
        if (cell) cell.textContent = formatSeconds(stageSeconds(run, run.activeStage));
      }
    });
  }

  /** Số giây chờ Groq cấp lại hạn mức token/phút (chế độ so sánh dùng cố định một model). */
  function waitSeconds(run) {
    var timings = run && run.result && run.result.timings;
    return (timings && timings.rate_limit_wait) || 0;
  }

  /** Thời gian phản hồi người dùng thấy (giây), không tính thời gian chờ hạn mức Groq. */
  function responseSeconds(run) {
    var total = run && (run.totalSeconds || run.duration);
    return total ? Math.max(total - waitSeconds(run), 0) : null;
  }

  function metricChip(icon, text, title) {
    return h("span", { class: "metric-chip", title: title || null }, icon ? h("span", { html: icon }) : null, text);
  }

  /** Các chỉ số dùng để đánh giá: thời gian, số lần gọi LLM, token, số đoạn, model, độ tin cậy, trích dẫn khớp. */
  function renderMetrics(mode, run, result) {
    var row = h("div", { class: "metrics" });
    var total = responseSeconds(run);
    var usage = result.usage || {};
    var sources = result.sources || [];

    if (total) row.append(metricChip(ICONS.clock, formatSeconds(total), timingTooltip(run, result)));
    if (waitSeconds(run)) row.append(metricChip(null, "⏳ chờ quota " + formatSeconds(waitSeconds(run)), WAIT_HINT));
    if (usage.llm_calls) row.append(metricChip(null, usage.llm_calls + " lần gọi LLM"));
    if (usage.prompt_tokens || usage.completion_tokens) {
      row.append(metricChip(null, formatNumber(usage.prompt_tokens + usage.completion_tokens) + " token",
        "Token vào: " + formatNumber(usage.prompt_tokens) + " · Token ra: " + formatNumber(usage.completion_tokens)));
    }
    if (mode !== "llm") row.append(metricChip(null, sources.length ? sources.length + " đoạn tài liệu" : "Không có đoạn tài liệu"));
    if (result.models && result.models.length) {
      row.append(metricChip(null, result.models.map(shortModel).join(", "), "Model đã trả lời"));
    }
    if (mode === "reasoning" && result.confidence) {
      row.append(h("span", { class: "badge " + (CONFIDENCE_CLASS[result.confidence] || "thap") }, "Độ tin cậy: " + result.confidence));
    }
    if (mode === "reasoning" && result.evidence && result.evidence.length) {
      var verified = result.evidence.filter(function (item) { return item.verified; }).length;
      row.append(metricChip(null, "Trích dẫn khớp " + verified + "/" + result.evidence.length,
        "Số câu trích dẫn khớp nguyên văn tài liệu / tổng số câu trích dẫn"));
    }
    return row;
  }

  function renderRating(msg, mode, run) {
    var row = h("div", { class: "rating" }, h("span", { class: "rating-label" }, "Đánh giá:"));
    RATINGS.forEach(function (rating) {
      var active = run.rating === rating.key;
      row.append(h("button", {
        type: "button", class: "rate-btn " + rating.key + (active ? " active" : ""), "aria-pressed": active ? "true" : "false",
        onclick: function () {
          run.rating = active ? null : rating.key;
          saveConversations();
          updateMessage(msg);
        }
      }, rating.icon + " " + rating.label));
    });
    return row;
  }

  function ratingLabel(key) {
    var found = RATINGS.filter(function (rating) { return rating.key === key; })[0];
    return found ? found.label : "";
  }

  function renderResult(msg, mode, run, compact) {
    var result = run.result || {};
    var sources = result.sources || [];
    var parts = [];

    if (mode === "reasoning" && result.category === "document" && sources.length) parts.push(renderThinking(run, result));

    var answer = result.answer || "";
    parts.push(h("div", {
      class: "markdown" + (compact && !run.expanded && answer.length > 600 ? " clamped" : ""),
      html: renderMarkdown(answer, sources.length)
    }));
    if (compact && answer.length > 600) {
      parts.push(h("button", {
        type: "button", class: "expand-btn",
        onclick: function () {
          run.expanded = !run.expanded;
          saveConversations();
          updateMessage(msg);
        }
      }, run.expanded ? "Thu gọn ▴" : "Xem đầy đủ ▾"));
    }

    parts.push(renderMetrics(mode, run, result));
    if (result.pipeline && result.pipeline.length) {
      parts.push(h("p", { class: "pipeline" }, "Quy trình: " + result.pipeline.join(" → ")));
    }
    if (result.warnings && result.warnings.length) {
      parts.push(h("div", { class: "warnings" }, result.warnings.map(function (w) { return h("p", {}, "⚠ " + w); })));
    }

    var actions = h("div", { class: "answer-meta" });
    if (sources.length) {
      actions.append(h("button", {
        class: "ghost-btn view-sources", type: "button", html: ICONS.quote + "<span>Văn bản gốc (" + sources.length + ")</span>",
        onclick: function () { openCitePanel(msg.id, null, mode); }
      }));
    }
    var copyBtn = h("button", { class: "ghost-btn", type: "button", html: ICONS.copy + "<span>Sao chép</span>" });
    copyBtn.addEventListener("click", function () {
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText(answer).then(function () {
        copyBtn.lastChild.textContent = "Đã chép";
        setTimeout(function () { copyBtn.lastChild.textContent = "Sao chép"; }, 1500);
      });
    });
    actions.append(copyBtn);
    parts.push(actions);

    if (!compact && sources.length) {
      parts.push(h("div", { class: "source-row" },
        h("span", { class: "source-row-label" }, "Nguồn:"),
        sources.map(function (src) {
          return h("button", { class: "src-chip", type: "button", "data-cite": src.index, title: "Xem văn bản gốc: " + src.label },
            h("span", { class: "source-index" }, src.index),
            h("span", { class: "src-chip-label" }, shortSourceLabel(src))
          );
        })
      ));
    }

    parts.push(renderRating(msg, mode, run));
    return parts;
  }

  function shortSourceLabel(src) {
    var name = String(src.filename || "").replace(/\.pdf$/i, "");
    if (src.page_number) {
      var pages = src.page_end && src.page_end !== src.page_number ? src.page_number + "-" + src.page_end : src.page_number;
      return truncate(name, 22) + " · tr. " + pages;
    }
    return truncate(name, 28);
  }

  function timingRows(run, result) {
    var timings = result.timings || {};
    var rows = STAGE_ORDER.filter(function (key) { return timings[key] !== undefined; }).map(function (key) {
      return [STAGE_LABELS[key], timings[key]];
    });
    var total = responseSeconds(run);
    if (timings.total !== undefined && total && total > timings.total) {
      rows.push(["Mạng & hiển thị", total - timings.total]);
    }
    return rows;
  }

  function timingTooltip(run, result) {
    var lines = timingRows(run, result).map(function (row) { return row[0] + ": " + formatSeconds(row[1]); });
    if (waitSeconds(run)) lines.push(WAIT_LABEL + ": " + formatSeconds(waitSeconds(run)));
    return lines.join("\n");
  }

  function renderThinking(run, result) {
    var total = responseSeconds(run);
    var label = total ? "Đã suy luận trong " + formatSeconds(total).replace(" s", " giây") : "Quá trình suy luận";
    var body = h("div", { class: "thinking-body" });

    if (result.standalone_question && result.standalone_question !== result.question) {
      body.append(h("h4", {}, "Câu hỏi được hiểu là"), h("p", {}, result.standalone_question));
    }
    if (result.sub_queries && result.sub_queries.length) {
      body.append(h("h4", {}, "Truy vấn tìm kiếm"),
        h("div", { class: "query-tags" }, result.sub_queries.map(function (q) { return h("span", { class: "query-tag" }, q); })));
    }
    if (result.reasoning_steps && result.reasoning_steps.length) {
      body.append(h("h4", {}, "Các bước suy luận"),
        h("ol", {}, result.reasoning_steps.map(function (step) { return h("li", {}, step); })));
    }
    if (result.evidence && result.evidence.length) {
      body.append(h("h4", {}, "Bằng chứng trích từ tài liệu"),
        h("ul", { class: "evidence" }, result.evidence.map(function (item) {
          return h("li", {},
            h("span", { class: "mark " + (item.verified ? "ok" : "bad"), title: item.verified ? "Khớp tài liệu" : "Không khớp tài liệu" }, item.verified ? "✓" : "✗"),
            h("span", {}, "[" + item.source_id + "] ", h("q", {}, item.quote), " ",
              h("small", {}, item.verified ? (item.source || "") : "không khớp nguyên văn tài liệu"))
          );
        })));
    }

    var rows = timingRows(run, result);
    if (rows.length) {
      var table = h("dl", { class: "timing-table" });
      rows.forEach(function (row) { table.append(h("dt", {}, row[0]), h("dd", {}, formatSeconds(row[1]))); });
      if (total) table.append(h("dt", { class: "total" }, "Tổng thời gian phản hồi"), h("dd", { class: "total" }, formatSeconds(total)));
      if (waitSeconds(run)) table.append(h("dt", { class: "wait" }, WAIT_LABEL), h("dd", { class: "wait" }, formatSeconds(waitSeconds(run))));
      body.append(h("h4", {}, "Thời gian xử lý"), table);
    }
    return h("details", { class: "thinking", "data-key": "thinking" }, h("summary", {}, label), body);
  }

  // ------------------------------------------------------------------ so sánh 4 phương pháp
  function renderCompareMessage(msg) {
    var done = msg.status === "done";
    var waited = msg.modes.reduce(function (sum, mode) { return sum + waitSeconds(msg.runs[mode]); }, 0);
    var head = h("div", { class: "compare-head" },
      h("span", { class: "compare-title" }, "So sánh 4 phương pháp"),
      msg.pinnedModel
        ? metricChip(null, "Cùng model: " + shortModel(msg.pinnedModel),
          "Cả 4 phương pháp dùng cùng một model. Khi chạm giới hạn token/phút, app chờ Groq cấp lại hạn mức thay vì đổi model.")
        : null,
      done && msg.totalSeconds
        ? metricChip(ICONS.clock, "Tổng " + formatSeconds(msg.totalSeconds), waited ? "Gồm " + formatSeconds(waited) + " chờ hạn mức Groq" : null)
        : h("span", { class: "live-total compare-live-total" }, formatSeconds((Date.now() - msg.startedAt) / 1000)),
      done && waited ? metricChip(null, "⏳ chờ quota " + formatSeconds(waited), WAIT_HINT) : null
    );

    var grid = h("div", { class: "compare-grid" }, msg.modes.map(function (mode) {
      var run = msg.runs[mode] || { status: "waiting" };
      var citing = state.cite.msgId === msg.id && state.cite.mode === mode;
      return h("section", { class: "mode-card " + (run.status || "waiting") + (citing ? " citing" : ""), "data-mode": mode },
        h("header", { class: "mode-card-head" }, modeTag(mode)),
        h("div", { class: "mode-card-body" }, renderRunBody(msg, mode, run, true))
      );
    }));

    return h("div", { class: "msg assistant compare" },
      h("div", { class: "avatar", html: ICONS.logo }),
      h("div", { class: "assistant-body" }, head, grid, done ? renderCompareSummary(msg) : null)
    );
  }

  function renderCompareSummary(msg) {
    var headers = ["Phương pháp", "Thời gian", "Gọi LLM", "Token", "Đoạn tài liệu", "Model", "Độ tin cậy", "Đánh giá"];
    var models = {};

    var rows = msg.modes.map(function (mode) {
      var run = msg.runs[mode] || {};
      var result = run.result || {};
      var usage = result.usage || {};
      (result.models || []).forEach(function (name) { models[String(name).split("#")[0]] = true; });
      var tokens = (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
      return h("tr", { class: run.rating ? "rated-" + run.rating : null },
        h("td", {}, modeTag(mode, true)),
        h("td", {}, run.status === "done" ? formatSeconds(responseSeconds(run)) : run.status === "error" ? "Lỗi" : "—",
          waitSeconds(run) ? h("small", { class: "wait-note", title: WAIT_HINT }, " (+" + formatSeconds(waitSeconds(run)) + " chờ)") : null),
        h("td", {}, usage.llm_calls || "—"),
        h("td", {}, tokens ? formatNumber(tokens) : "—"),
        h("td", {}, mode === "llm" ? "Không dùng" : result.sources ? String(result.sources.length) : "—"),
        h("td", {}, (result.models || []).map(shortModel).join(", ") || "—"),
        h("td", {}, mode === "reasoning" && result.confidence ? result.confidence : "—"),
        h("td", { class: "rating-cell" }, ratingLabel(run.rating) || "chưa chấm")
      );
    });

    var warning = Object.keys(models).length > 1
      ? h("p", { class: "compare-warning" },
        "⚠ Các phương pháp dùng model khác nhau (Groq tự chuyển model khi một model hết quota), nên so sánh chất lượng có thể chưa hoàn toàn công bằng.")
      : null;

    return h("div", { class: "compare-summary" },
      h("div", { class: "table-wrap" },
        h("table", { class: "compare-table" },
          h("thead", {}, h("tr", {}, headers.map(function (text) { return h("th", {}, text); }))),
          h("tbody", {}, rows)
        )
      ),
      warning,
      h("p", { class: "compare-hint" }, "Chấm Đúng / Một phần / Sai ở từng phương pháp, rồi bấm \"Xuất đánh giá\" ở góc trên để tải bảng CSV.")
    );
  }

  // ------------------------------------------------------------------ hỏi
  function buildHistory(conv) {
    var turns = [];
    var question = null;
    conv.messages.forEach(function (msg) {
      if (msg.role === "user") {
        question = msg.text;
        return;
      }
      if (msg.role !== "assistant" || !question || !msg.runs) return;
      // Ưu tiên câu trả lời của phương pháp 04 làm ngữ cảnh cho câu hỏi tiếp theo
      var run = msg.runs.reasoning && msg.runs.reasoning.status === "done" ? msg.runs.reasoning : null;
      if (!run) {
        run = msg.modes.map(function (mode) { return msg.runs[mode]; })
          .filter(function (item) { return item && item.status === "done" && item.result; })[0];
      }
      if (run && run.result) {
        turns.push({ question: question.slice(0, 4000), answer: String(run.result.answer || "").slice(0, 8000) });
        question = null;
      }
    });
    return turns.slice(-HISTORY_TURNS);
  }

  function closeStage(run) {
    if (run.activeStage && run.stageTimes && run.stageTimes[run.activeStage]) {
      run.stageTimes[run.activeStage].end = Date.now();
    }
    run.activeStage = null;
  }

  function failUnfinished(msg, message) {
    msg.modes.forEach(function (mode) {
      var run = msg.runs[mode];
      if (run && run.status !== "done" && run.status !== "error") {
        closeStage(run);
        run.status = "error";
        run.error = message;
      }
    });
  }

  function handleEvent(msg, event) {
    var key = event.mode || (msg.modes.length === 1 ? msg.modes[0] : null);
    var run = key ? msg.runs[key] : null;
    var now = Date.now();

    if (event.type === "compare") {
      msg.pinnedModel = event.model;
    } else if (event.type === "start" && run) {
      run.status = "pending";
      run.stages = event.stages || [];
      run.stageTimes = {};
      run.activeStage = null;
      run.stage = "Bắt đầu";
      if (msg.modes.length > 1) run.startedAt = now;
    } else if (event.type === "status" && run) {
      closeStage(run);
      run.stageTimes = run.stageTimes || {};
      run.activeStage = event.stage;
      run.stageTimes[event.stage] = { start: now, end: null };
      run.stage = event.message;
    } else if (event.type === "result" && run) {
      closeStage(run);
      run.status = "done";
      run.result = event;
      run.totalSeconds = Math.round(now - run.startedAt) / 1000;
    } else if (event.type === "error") {
      if (run) {
        closeStage(run);
        run.status = "error";
        run.error = event.message;
        run.totalSeconds = Math.round(now - run.startedAt) / 1000;
      } else {
        failUnfinished(msg, event.message);
      }
    }
    updateMessage(msg);
  }

  async function submitQuestion(text) {
    text = String(text || "").trim();
    if (!text || state.busy) return;

    var conv = ensureConversation();
    var mode = state.mode;
    state.busy = true;
    els.input.value = "";
    autosize();
    updateComposer();

    var pending = Object.keys(state.uploads).map(function (key) { return state.uploads[key]; });
    var history = buildHistory(conv);
    var startedAt = Date.now();
    var waitingText = pending.length ? "Đang chờ xử lý xong tài liệu" : "Đang gửi câu hỏi";
    var botMsg = {
      id: newId(), role: "assistant", mode: mode, status: "pending", startedAt: startedAt,
      modes: mode === "compare" ? MODE_ORDER.slice() : [mode], runs: {}
    };
    botMsg.modes.forEach(function (item) {
      botMsg.runs[item] = { status: mode === "compare" ? "waiting" : "pending", stage: waitingText, startedAt: startedAt };
    });
    conv.messages.push({ id: newId(), role: "user", text: text }, botMsg);
    if (conv.title === DEFAULT_TITLE) conv.title = truncate(text, 48);
    touch(conv);
    if (conv.id === state.currentId) renderMessages();

    // Đồng hồ thời gian phản hồi: cập nhật mỗi 0,1 giây
    var ticker = setInterval(function () { tick(botMsg); }, 100);
    try {
      if (pending.length) {
        await Promise.all(pending);
        botMsg.startedAt = Date.now();
        botMsg.modes.forEach(function (item) {
          botMsg.runs[item].startedAt = botMsg.startedAt;
          botMsg.runs[item].stage = "Đang gửi câu hỏi";
        });
        updateMessage(botMsg);
      }
      await streamAsk(conv.id, text, history, mode, function (event) { handleEvent(botMsg, event); });
      failUnfinished(botMsg, "Server không trả về kết quả (có thể đã quá thời gian xử lý).");
    } catch (err) {
      failUnfinished(botMsg, err && err.message ? err.message : String(err));
    } finally {
      clearInterval(ticker);
      botMsg.status = "done";
      botMsg.totalSeconds = Math.round(Date.now() - botMsg.startedAt) / 1000;
      botMsg.modes.forEach(function (item) {
        delete botMsg.runs[item].stageTimes;
        delete botMsg.runs[item].activeStage;
      });
      state.busy = false;
      touch(conv);
      updateMessage(botMsg);
      updateComposer();
      scrollToBottom();
    }

    // Màn hình rộng: tự mở hộp trích dẫn văn bản gốc cạnh câu trả lời (khi chỉ chạy một phương pháp)
    var single = botMsg.modes.length === 1 ? botMsg.runs[botMsg.modes[0]] : null;
    if (single && single.status === "done" && single.result && single.result.sources && single.result.sources.length &&
        conv.id === state.currentId && window.innerWidth >= WIDE_SCREEN) {
      openCitePanel(botMsg.id, null, botMsg.modes[0]);
    }
  }

  // ------------------------------------------------------------------ xuất bảng đánh giá
  function csvCell(value) {
    var text = value === null || value === undefined ? "" : String(value);
    return /[",\r\n;]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function slugify(text) {
    var slug = String(text || "").normalize("NFD").replace(/[̀-ͯ]/g, "")
      .replace(/đ/g, "d").replace(/Đ/g, "D")
      .replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase().slice(0, 40);
    return slug || "cuoc-tro-chuyen";
  }

  function exportEvaluation() {
    var conv = current();
    var rows = [[
      "STT", "Câu hỏi", "Phương pháp", "Thời gian (giây)", "Chờ hạn mức Groq (giây)", "Số lần gọi LLM", "Token vào", "Token ra",
      "Số đoạn tài liệu", "Model", "Độ tin cậy", "Trích dẫn khớp", "Đánh giá", "Câu trả lời"
    ]];
    var question = "";
    var number = 0;

    (conv ? conv.messages : []).forEach(function (msg) {
      if (msg.role === "user") {
        question = msg.text;
        return;
      }
      if (msg.role !== "assistant" || !msg.runs) return;
      number += 1;
      msg.modes.forEach(function (mode) {
        var run = msg.runs[mode];
        if (!run) return;
        var result = run.result || {};
        var usage = result.usage || {};
        var evidence = result.evidence || [];
        var verified = evidence.filter(function (item) { return item.verified; }).length;
        rows.push([
          number,
          question,
          MODES[mode].num + " " + MODES[mode].title,
          run.status === "done" && responseSeconds(run) ? responseSeconds(run).toFixed(2) : "",
          waitSeconds(run) ? waitSeconds(run).toFixed(2) : "",
          usage.llm_calls || "",
          usage.prompt_tokens || "",
          usage.completion_tokens || "",
          mode === "llm" ? 0 : (result.sources ? result.sources.length : ""),
          (result.models || []).map(shortModel).join(", "),
          mode === "reasoning" ? (result.confidence || "") : "",
          evidence.length ? verified + "/" + evidence.length : "",
          ratingLabel(run.rating),
          run.status === "error" ? "LỖI: " + (run.error || "") : (result.answer || "")
        ]);
      });
    });

    if (rows.length === 1) {
      alert("Cuộc trò chuyện này chưa có câu trả lời nào để xuất.");
      return;
    }
    // BOM để Excel đọc đúng tiếng Việt
    var csv = "﻿" + rows.map(function (row) { return row.map(csvCell).join(","); }).join("\r\n");
    var url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    var link = h("a", { href: url, download: "danh-gia-" + slugify(conv.title) + ".csv" });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // ------------------------------------------------------------------ upload
  function handleFiles(fileList) {
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) return;

    var conv = ensureConversation();
    var maxBytes = (state.config.max_upload_mb || 4) * 1024 * 1024;

    files.forEach(function (file) {
      var msg = { id: newId(), role: "doc", status: "uploading", filename: file.name, size: file.size, progress: 0 };
      conv.messages.push(msg);
      if (conv.title === DEFAULT_TITLE) conv.title = truncate(file.name.replace(/\.pdf$/i, ""), 48);

      if (!/\.pdf$/i.test(file.name)) {
        msg.status = "error";
        msg.error = "Chỉ hỗ trợ file PDF.";
        return;
      }
      if (file.size > maxBytes) {
        msg.status = "error";
        msg.error = "File " + formatSize(file.size) + " vượt giới hạn " + state.config.max_upload_mb + " MB. Hãy tách nhỏ hoặc nén PDF.";
        return;
      }
      state.uploads[msg.id] = uploadFile(conv, msg, file).then(function () {
        delete state.uploads[msg.id];
        updateComposer();
      });
    });

    touch(conv);
    if (conv.id === state.currentId) renderMessages();
    updateComposer();
    scrollToBottom();
  }

  function uploadFile(conv, msg, file) {
    return new Promise(function (resolve) {
      var xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/conversations/" + encodeURIComponent(conv.id) + "/documents");
      xhr.upload.onprogress = function (event) {
        if (!event.lengthComputable) return;
        msg.progress = Math.round((event.loaded / event.total) * 100);
        updateMessage(msg);
      };
      xhr.upload.onload = function () {
        msg.status = "processing";
        updateMessage(msg);
      };
      xhr.onload = function () {
        var data = null;
        try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
        if (xhr.status >= 200 && xhr.status < 300 && data) {
          msg.status = "ready";
          msg.meta = data;
          msg.overviewStatus = "loading";
          if (conv.id === state.currentId) refreshDocs();
          loadOverview(conv, msg);
        } else {
          msg.status = "error";
          msg.error = data && typeof data.detail === "string" ? data.detail : "Tải lên thất bại (lỗi " + xhr.status + ").";
        }
        touch(conv);
        updateMessage(msg);
        resolve();
      };
      xhr.onerror = function () {
        msg.status = "error";
        msg.error = "Mất kết nối khi tải lên.";
        touch(conv);
        updateMessage(msg);
        resolve();
      };
      var form = new FormData();
      form.append("file", file, file.name);
      xhr.send(form);
    });
  }

  async function loadOverview(conv, msg) {
    try {
      var response = await fetch("/api/conversations/" + encodeURIComponent(conv.id) + "/documents/" + msg.meta.id + "/overview");
      if (!response.ok) throw new Error(await readError(response));
      msg.overview = await response.json();
      msg.overviewStatus = "ready";
    } catch (err) {
      msg.overviewStatus = "error";
    }
    saveConversations();
    updateMessage(msg);
  }

  // ------------------------------------------------------------------ hộp trích dẫn văn bản gốc
  var WORD_PATTERN = "[\\p{L}\\p{N}]+";
  var GAP_PATTERN = "[^\\p{L}\\p{N}]+";

  function findMessage(id) {
    var conv = current();
    if (!conv || !id) return null;
    for (var i = 0; i < conv.messages.length; i++) {
      if (conv.messages[i].id === id) return conv.messages[i];
    }
    return null;
  }

  function runOf(msg, mode) {
    if (!msg || !msg.runs) return null;
    return msg.runs[mode || msg.modes[0]] || null;
  }

  /** Các đoạn [n] được câu trả lời trích dẫn hoặc dùng làm bằng chứng. */
  function citedIndexes(result) {
    var cited = {};
    String(result.answer || "").replace(/\[(\d{1,3})\]/g, function (match, n) {
      cited[Number(n)] = true;
      return match;
    });
    (result.evidence || []).forEach(function (item) { cited[item.source_id] = true; });
    return cited;
  }

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  /** Vị trí câu trích trong văn bản gốc (bỏ qua hoa / thường, khoảng trắng, dấu câu; cho phép lược bằng "..."). */
  function findQuoteRanges(text, quote) {
    var ranges = [];
    String(quote).normalize("NFC").split(/\.\.\.|…/).forEach(function (segment) {
      var words = segment.match(new RegExp(WORD_PATTERN, "gu"));
      if (!words || words.join("").length < 4) return;
      var match = new RegExp(words.map(escapeRegExp).join(GAP_PATTERN), "iu").exec(text);
      if (match) ranges.push([match.index, match.index + match[0].length]);
    });
    return ranges;
  }

  /** Dựng văn bản có <mark> bằng DOM (không dùng innerHTML với nội dung tài liệu). */
  function highlightText(text, ranges) {
    var fragment = document.createDocumentFragment();
    var cursor = 0;
    ranges.sort(function (a, b) { return a[0] - b[0]; }).forEach(function (range) {
      var start = Math.max(range[0], cursor);
      if (start >= range[1]) return;
      if (start > cursor) fragment.append(text.slice(cursor, start));
      fragment.append(h("mark", { class: "ev" }, text.slice(start, range[1])));
      cursor = range[1];
    });
    if (cursor < text.length) fragment.append(text.slice(cursor));
    return fragment;
  }

  function renderCitePanel() {
    var msg = findMessage(state.cite.msgId);
    var mode = state.cite.mode || (msg && msg.modes ? msg.modes[0] : null);
    var run = runOf(msg, mode);
    if (!run || !run.result) {
      closeCitePanel();
      return;
    }
    var result = run.result;
    var sources = result.sources || [];
    var reasoning = mode === "reasoning";
    var cited = citedIndexes(result);
    var total = responseSeconds(run);

    els.citeQuestion.textContent = "Câu hỏi: " + (result.question || "");
    els.citeMeta.textContent = "";
    els.citeMeta.append(modeTag(mode, true));
    if (total) {
      els.citeMeta.append(h("span", { class: "time-chip", title: timingTooltip(run, result) },
        h("span", { html: ICONS.clock }), formatSeconds(total)));
    }
    if (reasoning && result.confidence) {
      els.citeMeta.append(h("span", { class: "badge " + (CONFIDENCE_CLASS[result.confidence] || "thap") }, "Độ tin cậy: " + result.confidence));
    }
    els.citeMeta.append(h("span", { class: "time-chip" }, sources.length + " đoạn trích"));

    els.citeBody.textContent = "";
    sources.forEach(function (src) {
      var text = String(src.content || "").normalize("NFC");
      var ranges = [];
      var notes = [];
      (result.evidence || []).forEach(function (item) {
        if (item.source_id !== src.index) return;
        var found = item.verified ? findQuoteRanges(text, item.quote) : [];
        if (found.length) {
          ranges = ranges.concat(found);
        } else {
          notes.push(h("div", { class: "cite-quote-line" + (item.verified ? "" : " bad") },
            (item.verified ? "✓ Trích dẫn (khớp gần đúng): " : "✗ Không khớp tài liệu: ") + "“" + item.quote + "”"));
        }
      });

      // 04: phân biệt đoạn được dùng / không dùng; 02, 03: mọi đoạn đều được đưa cho LLM
      var used = reasoning ? !!cited[src.index] : true;
      var tag = reasoning
        ? (used ? "Được dùng trong câu trả lời" : "Tìm thấy nhưng không dùng trong câu trả lời")
        : "Đưa cho LLM làm ngữ cảnh";
      var hasRerank = src.rerank_score !== null && src.rerank_score !== undefined;
      var scoreText = hasRerank
        ? "rerank " + Number(src.rerank_score).toFixed(1) + "/10 · trước #" + src.initial_rank
        : Number(src.score).toFixed(2);
      var pdfUrl = src.doc_id ? api("/documents/" + src.doc_id + "/file") + (src.page_number ? "#page=" + src.page_number : "") : null;

      els.citeBody.append(h("article", { class: "cite-card " + (used ? "cited" : "unused"), "data-index": src.index },
        h("div", { class: "cite-card-head" },
          h("span", { class: "source-index" }, src.index),
          h("span", { class: "cite-card-label", title: src.label }, src.label),
          h("span", {
            class: "cite-card-score",
            title: hasRerank ? "Điểm LLM chấm khi xếp hạng lại / thứ hạng trước khi xếp hạng lại" : "Độ liên quan khi tìm kiếm"
          }, scoreText),
          pdfUrl ? h("a", {
            class: "cite-card-open", href: pdfUrl, target: "_blank", rel: "noopener",
            title: "Mở PDF tại trang này", "aria-label": "Mở PDF tại trang này", html: ICONS.open
          }) : null
        ),
        h("p", { class: "cite-card-text" }, highlightText(text, ranges)),
        h("div", { class: "cite-card-foot" },
          h("span", { class: "cite-tag" + (used ? " used" : "") }, tag),
          notes
        )
      ));
    });
  }

  function focusCitation(index) {
    Array.prototype.forEach.call(els.citeBody.querySelectorAll(".cite-card"), function (card) {
      card.classList.toggle("active", index !== null && String(index) === card.getAttribute("data-index"));
    });
    if (index === null || index === undefined) {
      els.citeBody.scrollTop = 0;
      return;
    }
    var card = els.citeBody.querySelector('.cite-card[data-index="' + index + '"]');
    if (!card) return;
    card.scrollIntoView({ block: "start", behavior: "smooth" });
    // Cuộn phần văn bản tới câu được tô sáng đầu tiên
    var mark = card.querySelector("mark.ev");
    var box = card.querySelector(".cite-card-text");
    if (mark && box) box.scrollTop = Math.max(0, mark.offsetTop - 40);
  }

  function markCitingMessage() {
    Array.prototype.forEach.call(els.inner.querySelectorAll(".msg.assistant"), function (node) {
      var id = node.getAttribute("data-id");
      var selected = !!state.cite.msgId && id === state.cite.msgId;
      node.classList.toggle("citing", selected && !node.classList.contains("compare"));
      Array.prototype.forEach.call(node.querySelectorAll(".mode-card"), function (card) {
        card.classList.toggle("citing", selected && card.getAttribute("data-mode") === state.cite.mode);
      });
    });
  }

  function openCitePanel(msgId, focus, mode) {
    var msg = findMessage(msgId);
    state.cite.msgId = msgId;
    state.cite.mode = mode || (msg && msg.modes ? msg.modes[0] : null);
    renderCitePanel();
    if (!state.cite.msgId) return;
    els.citePanel.hidden = false;
    els.app.classList.add("cite-open");
    markCitingMessage();
    focusCitation(focus === undefined ? null : focus);
  }

  function closeCitePanel() {
    state.cite.msgId = null;
    state.cite.mode = null;
    els.citePanel.hidden = true;
    els.app.classList.remove("cite-open");
    markCitingMessage();
  }

  // ------------------------------------------------------------------ documents panel
  async function refreshDocs() {
    var conversationId = state.currentId;
    if (!current()) {
      state.docs = [];
      renderDocs();
      return;
    }
    try {
      var response = await fetch(api("/documents"));
      if (!response.ok) throw new Error(await readError(response));
      var docs = await response.json();
      if (conversationId !== state.currentId) return;
      state.docs = docs;
    } catch (err) {
      state.docs = [];
    }
    renderDocs();
  }

  function renderDocs() {
    var count = state.docs.length;
    els.docsCount.textContent = count;
    els.docsBtn.classList.toggle("has-docs", count > 0);
    els.docsList.textContent = "";
    if (!count) {
      els.docsList.append(h("div", { class: "docs-empty" }, "Chưa có tài liệu nào trong cuộc trò chuyện này."));
      return;
    }
    state.docs.forEach(function (doc) {
      els.docsList.append(h("div", { class: "docs-item" },
        h("div", { class: "doc-icon", "aria-hidden": "true" }, "PDF"),
        h("div", { class: "doc-info" },
          h("div", { class: "doc-name", title: doc.filename }, doc.filename),
          h("div", { class: "doc-sub" }, doc.pages + " trang · " + doc.chunks + " đoạn · " + formatSize(doc.size))
        ),
        h("div", { class: "docs-actions" },
          h("a", { href: api("/documents/" + doc.id + "/file"), target: "_blank", rel: "noopener", title: "Mở PDF", "aria-label": "Mở PDF", html: ICONS.open }),
          h("button", { type: "button", title: "Xoá tài liệu", "aria-label": "Xoá tài liệu", html: ICONS.trash, onclick: function () { deleteDocument(doc); } })
        )
      ));
    });
  }

  async function deleteDocument(doc) {
    if (!confirm('Xoá tài liệu "' + doc.filename + '"? Các câu hỏi sau sẽ không dùng tài liệu này nữa.')) return;
    try {
      var response = await fetch(api("/documents/" + doc.id), { method: "DELETE" });
      if (!response.ok) throw new Error(await readError(response));
    } catch (err) {
      alert("Không xoá được tài liệu: " + err.message);
    }
    refreshDocs();
  }

  // ------------------------------------------------------------------ composer
  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 220) + "px";
  }

  function updateComposer() {
    els.sendBtn.disabled = state.busy || !els.input.value.trim();
    var pending = totalPendingUploads();
    els.uploadHint.hidden = pending === 0;
    els.uploadHint.textContent = pending ? "Đang xử lý " + pending + " tài liệu… Bạn vẫn có thể gõ câu hỏi, câu hỏi sẽ được gửi khi xử lý xong." : "";
  }

  // ------------------------------------------------------------------ theme
  function currentTheme() {
    var theme = document.documentElement.getAttribute("data-theme");
    if (theme) return theme;
    return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function renderThemeToggle() {
    var dark = currentTheme() === "dark";
    els.themeToggle.querySelector(".theme-icon").innerHTML = dark ? ICONS.sun : ICONS.moon;
    els.themeToggle.querySelector(".theme-label").textContent = dark ? "Giao diện sáng" : "Giao diện tối";
  }

  // ------------------------------------------------------------------ server status
  function showBanner(text) {
    els.banner.textContent = text;
    els.banner.hidden = false;
  }

  async function loadHealth() {
    try {
      var response = await fetch("/api/health");
      var data = await response.json();
      state.config = data;
      var storage = data.storage === "vercel-blob" ? "Vercel Blob" : data.storage === "local" ? "thư mục dataset_web/" : "chưa sẵn sàng";
      var providers = (data.llm_providers || []).map(shortModel).join(" → ") || "chưa cấu hình";
      if (data.llm_keys > 1) providers += " (" + data.llm_keys + " key Groq)";
      var compare = data.compare_model ? shortModel(data.compare_model) : null;
      var search = data.embedding_model ? "từ khoá + ngữ nghĩa (" + data.embedding_model + ")" : "từ khoá (BM25)";
      els.serverStatus.innerHTML =
        "<strong>Model:</strong> " + escapeHtml(providers) +
        (compare ? "<br><strong>So sánh:</strong> " + escapeHtml(compare) : "") +
        "<br><strong>Tìm kiếm:</strong> " + escapeHtml(search) +
        "<br><strong>Lưu trữ:</strong> " + escapeHtml(storage);
      if (data.storage_error) showBanner(data.storage_error);
      else if (!data.llm_configured) showBanner("Server chưa có GROQ_API_KEY nên chưa thể trả lời câu hỏi. Thêm biến này trong Environment Variables rồi Redeploy.");
      else if (data.embedding_error) showBanner("Cấu hình Cloudflare chưa đúng nên đang chỉ tìm theo từ khoá: " + data.embedding_error);
      if (!current()) renderMessages();
    } catch (err) {
      els.serverStatus.textContent = "Không kết nối được server.";
      showBanner("Không kết nối được API của server. Kiểm tra server đang chạy.");
    }
  }

  // ------------------------------------------------------------------ events
  function bindEvents() {
    els.newChat.addEventListener("click", startNewConversation);
    els.openSidebar.addEventListener("click", function () { els.app.classList.add("sidebar-open"); });
    els.closeSidebar.addEventListener("click", closeSidebarMobile);
    els.backdrop.addEventListener("click", closeSidebarMobile);

    els.composer.addEventListener("submit", function (event) {
      event.preventDefault();
      submitQuestion(els.input.value);
    });
    els.input.addEventListener("input", function () { autosize(); updateComposer(); });
    els.input.addEventListener("keydown", function (event) {
      // isComposing: tránh gửi khi đang gõ dấu tiếng Việt bằng bộ gõ
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        submitQuestion(els.input.value);
      }
    });

    els.modeBtn.addEventListener("click", function () {
      if (els.modeMenu.hidden) openModeMenu();
      else closeModeMenu();
    });
    document.addEventListener("click", function (event) {
      if (!els.modeMenu.hidden && !els.modePicker.contains(event.target)) closeModeMenu();
    });

    els.attachBtn.addEventListener("click", function () { els.fileInput.click(); });
    els.docsUpload.addEventListener("click", function () { els.fileInput.click(); });
    els.fileInput.addEventListener("change", function () {
      handleFiles(els.fileInput.files);
      els.fileInput.value = "";
    });

    els.exportBtn.addEventListener("click", exportEvaluation);
    els.docsBtn.addEventListener("click", function () {
      els.docsPanel.hidden = !els.docsPanel.hidden;
      if (!els.docsPanel.hidden) refreshDocs();
    });
    els.closeDocs.addEventListener("click", function () { els.docsPanel.hidden = true; });

    // Bấm chip [n] hoặc chip nguồn -> mở hộp văn bản gốc tại đoạn tương ứng (đúng phương pháp)
    els.inner.addEventListener("click", function (event) {
      var trigger = event.target.closest(".cite, .src-chip");
      if (!trigger) return;
      var card = trigger.closest(".mode-card");
      openCitePanel(
        trigger.closest(".msg").getAttribute("data-id"),
        trigger.getAttribute("data-cite"),
        card ? card.getAttribute("data-mode") : null
      );
    });
    els.closeCite.addEventListener("click", closeCitePanel);

    // Kéo thả file PDF vào khu vực chat
    var dragDepth = 0;
    function hasFiles(event) {
      return event.dataTransfer && Array.prototype.indexOf.call(event.dataTransfer.types, "Files") >= 0;
    }
    els.main.addEventListener("dragenter", function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth++;
      els.dropOverlay.hidden = false;
    });
    els.main.addEventListener("dragover", function (event) {
      if (hasFiles(event)) event.preventDefault();
    });
    els.main.addEventListener("dragleave", function () {
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) els.dropOverlay.hidden = true;
    });
    els.main.addEventListener("drop", function (event) {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth = 0;
      els.dropOverlay.hidden = true;
      handleFiles(event.dataTransfer.files);
    });

    els.themeToggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* bỏ qua */ }
      renderThemeToggle();
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        if (!els.modeMenu.hidden) closeModeMenu();
        else if (!els.docsPanel.hidden) els.docsPanel.hidden = true;
        else if (state.cite.msgId) closeCitePanel();
        closeSidebarMobile();
      }
    });
  }

  // ------------------------------------------------------------------ init
  function init() {
    state.conversations.forEach(function (conv) {
      conv.messages.forEach(function (msg) {
        normalizeAssistant(msg);
        // Upload / câu hỏi bị gián đoạn do tải lại trang
        if (msg.role === "doc" && (msg.status === "uploading" || msg.status === "processing")) {
          msg.status = "error";
          msg.error = "Tải lên bị gián đoạn, hãy tải lại file.";
        }
        if (msg.role === "doc" && msg.overviewStatus === "loading") msg.overviewStatus = "error";
        if (msg.role === "assistant") {
          msg.modes.forEach(function (mode) {
            var run = msg.runs[mode];
            if (run && (run.status === "pending" || run.status === "waiting")) {
              run.status = "error";
              run.error = "Câu trả lời bị gián đoạn do trang được tải lại.";
            }
          });
          msg.status = "done";
        }
      });
    });

    var last = null;
    try { last = localStorage.getItem(CURRENT_KEY); } catch (e) { last = null; }
    var exists = state.conversations.some(function (c) { return c.id === last; });
    state.currentId = exists ? last : newId();

    bindEvents();
    renderThemeToggle();
    renderModePicker();
    renderAll();
    refreshDocs();
    loadHealth();
  }

  init();
})();
