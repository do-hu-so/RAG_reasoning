/* =========================================================================
   RAG Reasoning — logic giao diện chat
   - Lịch sử cuộc trò chuyện lưu trong localStorage của trình duyệt
   - Tài liệu PDF lưu trên server ("dataset web"), theo từng cuộc trò chuyện
   ========================================================================= */
(function () {
  "use strict";

  var STORAGE_KEY = "rag.conversations.v1";
  var CURRENT_KEY = "rag.current";
  var THEME_KEY = "rag.theme";
  var HISTORY_TURNS = 4;
  var MAX_CONVERSATIONS = 40;
  var DEFAULT_TITLE = "Cuộc trò chuyện mới";
  var WIDE_SCREEN = 1200; // từ độ rộng này hộp trích dẫn tự mở cạnh câu trả lời

  // Các bước xử lý một câu hỏi (khớp với "stage" server gửi về)
  var STAGES = [
    { key: "analyzing", label: "Phân tích câu hỏi" },
    { key: "retrieving", label: "Tìm đoạn liên quan" },
    { key: "reasoning", label: "Suy luận & kiểm chứng" }
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
    banner: $("banner"),
    main: $("main"),
    messages: $("messages"),
    inner: $("messages-inner"),
    composer: $("composer"),
    input: $("input"),
    sendBtn: $("send-btn"),
    attachBtn: $("attach-btn"),
    fileInput: $("file-input"),
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
    cite: { msgId: null, focus: null }
  };

  function loadConversations() {
    try {
      var list = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(list) ? list : [];
    } catch (e) {
      return [];
    }
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

  async function streamAsk(conversationId, question, history, onEvent) {
    var response = await fetch("/api/conversations/" + encodeURIComponent(conversationId) + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, history: history })
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
    return h("div", { class: "empty" },
      h("h2", {}, "Hỏi đáp tài liệu PDF với RAG reasoning"),
      h("p", {}, "Tải PDF lên, hệ thống sẽ đọc, tạo embedding rồi trả lời câu hỏi của bạn bằng suy luận từng bước kèm trích dẫn nguồn."),
      h("button", { class: "dropzone", type: "button", onclick: function () { els.fileInput.click(); } },
        h("span", { html: ICONS.upload }),
        h("strong", {}, "Kéo thả hoặc bấm để chọn file PDF"),
        h("span", {}, "Tối đa " + maxMb + " MB mỗi file · PDF có lớp chữ (không phải bản scan)")
      ),
      h("div", { class: "features" },
        h("div", { class: "feature" }, h("b", {}, "Trích dẫn nguồn"), "Mỗi ý trả lời đều dẫn về đoạn và trang trong tài liệu."),
        h("div", { class: "feature" }, h("b", {}, "Suy luận từng bước"), "Xem bằng chứng và lập luận phía sau câu trả lời."),
        h("div", { class: "feature" }, h("b", {}, "Kiểm chứng tự động"), "Trích dẫn không khớp tài liệu bị đánh dấu và hạ độ tin cậy.")
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
      sub = "Đang đọc PDF và tạo embedding…";
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

  function renderAssistantMessage(msg) {
    var body;
    if (msg.status === "pending") {
      body = renderLiveStatus(msg);
    } else if (msg.status === "error") {
      body = h("div", { class: "error-box" }, msg.error || "Đã xảy ra lỗi.");
    } else {
      body = renderResult(msg);
    }
    return h("div", { class: "msg assistant" + (state.cite.msgId === msg.id ? " citing" : "") },
      h("div", { class: "avatar", html: ICONS.logo }),
      h("div", { class: "assistant-body" }, body)
    );
  }

  /** Thời gian đã trôi qua của một bước (giây), tính theo đồng hồ trình duyệt. */
  function stageSeconds(msg, key) {
    var times = msg.stageTimes && msg.stageTimes[key];
    if (!times) return null;
    return ((times.end || Date.now()) - times.start) / 1000;
  }

  function renderLiveStatus(msg) {
    var started = msg.stageTimes && Object.keys(msg.stageTimes).length > 0;
    var list = null;
    if (started) {
      list = h("ol", { class: "stage-list" }, STAGES.map(function (stage) {
        var times = msg.stageTimes[stage.key];
        var state = !times ? "waiting" : times.end ? "done" : "active";
        var icon = state === "done" ? h("span", { class: "stage-icon", html: ICONS.check })
          : state === "active" ? h("span", { class: "stage-icon" }, h("span", { class: "spinner" }))
          : h("span", { class: "stage-icon" }, "○");
        return h("li", { class: state, "data-stage": stage.key },
          icon,
          h("span", {}, stage.label),
          h("span", { class: "stage-time" }, times ? formatSeconds(stageSeconds(msg, stage.key)) : "")
        );
      }));
    }
    return h("div", { class: "live-box" },
      h("div", { class: "live-head" },
        h("span", { class: "pulse" }),
        h("span", { class: "shimmer" }, (msg.stage || "Đang xử lý") + "…"),
        h("span", { class: "live-total", title: "Thời gian phản hồi" }, formatSeconds((Date.now() - msg.startedAt) / 1000))
      ),
      list
    );
  }

  /** Cập nhật đồng hồ tại chỗ (không vẽ lại cả tin nhắn để hiệu ứng không bị giật). */
  function tickPending(msg) {
    var node = els.inner.querySelector('[data-id="' + msg.id + '"]');
    if (!node) return;
    var total = node.querySelector(".live-total");
    if (total) total.textContent = formatSeconds((Date.now() - msg.startedAt) / 1000);
    if (msg.activeStage) {
      var cell = node.querySelector('.stage-list li[data-stage="' + msg.activeStage + '"] .stage-time');
      if (cell) cell.textContent = formatSeconds(stageSeconds(msg, msg.activeStage));
    }
  }

  /** Tổng thời gian phản hồi người dùng thấy (giây). */
  function responseSeconds(msg) {
    if (msg.totalSeconds) return msg.totalSeconds;
    return msg.duration || null; // tin nhắn cũ chỉ lưu số giây làm tròn
  }

  function renderResult(msg) {
    var result = msg.result || {};
    var sources = result.sources || [];
    var isDocument = result.category === "document" && sources.length > 0;
    var parts = [];
    var total = responseSeconds(msg);

    if (isDocument) parts.push(renderThinking(msg, result));

    parts.push(h("div", { class: "markdown", html: renderMarkdown(result.answer || "", sources.length) }));

    var meta = h("div", { class: "answer-meta" });
    if (total) {
      meta.append(h("span", { class: "time-chip", title: timingTooltip(msg, result) },
        h("span", { html: ICONS.clock }), formatSeconds(total)));
    }
    if (isDocument && result.confidence) {
      meta.append(h("span", { class: "badge " + (CONFIDENCE_CLASS[result.confidence] || "thap") }, "Độ tin cậy: " + result.confidence));
    }
    if (isDocument) {
      meta.append(h("button", {
        class: "ghost-btn view-sources", type: "button", html: ICONS.quote + "<span>Văn bản gốc (" + sources.length + ")</span>",
        onclick: function () { openCitePanel(msg.id, null); }
      }));
    }
    var copyBtn = h("button", { class: "ghost-btn", type: "button", html: ICONS.copy + "<span>Sao chép</span>" });
    copyBtn.addEventListener("click", function () {
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText(result.answer || "").then(function () {
        copyBtn.lastChild.textContent = "Đã chép";
        setTimeout(function () { copyBtn.lastChild.textContent = "Sao chép"; }, 1500);
      });
    });
    meta.append(copyBtn);
    parts.push(meta);

    if (result.warnings && result.warnings.length) {
      parts.push(h("div", { class: "warnings" }, result.warnings.map(function (w) { return h("p", {}, "⚠ " + w); })));
    }

    if (isDocument) {
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

  function timingRows(msg, result) {
    var timings = result.timings || {};
    var rows = STAGES.filter(function (stage) { return timings[stage.key] !== undefined; }).map(function (stage) {
      return [stage.label, timings[stage.key]];
    });
    var total = responseSeconds(msg);
    if (timings.total !== undefined && total && total > timings.total) {
      rows.push(["Mạng & hiển thị", total - timings.total]);
    }
    return rows;
  }

  function timingTooltip(msg, result) {
    return timingRows(msg, result).map(function (row) { return row[0] + ": " + formatSeconds(row[1]); }).join("\n");
  }

  function renderThinking(msg, result) {
    var total = responseSeconds(msg);
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

    var rows = timingRows(msg, result);
    if (rows.length) {
      var table = h("dl", { class: "timing-table" });
      rows.forEach(function (row) { table.append(h("dt", {}, row[0]), h("dd", {}, formatSeconds(row[1]))); });
      if (total) table.append(h("dt", { class: "total" }, "Tổng thời gian phản hồi"), h("dd", { class: "total" }, formatSeconds(total)));
      body.append(h("h4", {}, "Thời gian xử lý"), table);
    }
    return h("details", { class: "thinking", "data-key": "thinking" }, h("summary", {}, label), body);
  }

  // ------------------------------------------------------------------ ask
  function buildHistory(conv) {
    var turns = [];
    var question = null;
    conv.messages.forEach(function (msg) {
      if (msg.role === "user") question = msg.text;
      else if (msg.role === "assistant" && msg.status === "done" && question && msg.result) {
        turns.push({ question: question.slice(0, 4000), answer: String(msg.result.answer || "").slice(0, 8000) });
        question = null;
      }
    });
    return turns.slice(-HISTORY_TURNS);
  }

  async function submitQuestion(text) {
    text = String(text || "").trim();
    if (!text || state.busy) return;

    var conv = ensureConversation();
    state.busy = true;
    els.input.value = "";
    autosize();
    updateComposer();

    var pending = Object.keys(state.uploads).map(function (key) { return state.uploads[key]; });
    var history = buildHistory(conv);
    var userMsg = { id: newId(), role: "user", text: text };
    var botMsg = {
      id: newId(), role: "assistant", status: "pending", startedAt: Date.now(),
      stage: pending.length ? "Đang chờ xử lý xong tài liệu" : "Đang gửi câu hỏi"
    };
    conv.messages.push(userMsg, botMsg);
    if (conv.title === DEFAULT_TITLE) conv.title = truncate(text, 48);
    touch(conv);
    if (conv.id === state.currentId) renderMessages();

    // Đồng hồ thời gian phản hồi: cập nhật mỗi 0,1 giây
    var ticker = setInterval(function () { tickPending(botMsg); }, 100);
    var closeStage = function () {
      if (botMsg.activeStage) botMsg.stageTimes[botMsg.activeStage].end = Date.now();
      botMsg.activeStage = null;
    };
    try {
      if (pending.length) {
        await Promise.all(pending);
        botMsg.stage = "Đang gửi câu hỏi";
        botMsg.startedAt = Date.now();
        updateMessage(botMsg);
      }
      botMsg.stageTimes = {};
      await streamAsk(conv.id, text, history, function (event) {
        if (event.type === "status") {
          closeStage();
          botMsg.activeStage = event.stage;
          botMsg.stageTimes[event.stage] = { start: Date.now(), end: null };
          botMsg.stage = event.message;
          updateMessage(botMsg);
        } else if (event.type === "result") {
          closeStage();
          botMsg.status = "done";
          botMsg.result = event;
        } else if (event.type === "error") {
          closeStage();
          botMsg.status = "error";
          botMsg.error = event.message;
        }
      });
      if (botMsg.status === "pending") {
        botMsg.status = "error";
        botMsg.error = "Server không trả về kết quả (có thể đã quá thời gian xử lý).";
      }
    } catch (err) {
      botMsg.status = "error";
      botMsg.error = err && err.message ? err.message : String(err);
    } finally {
      clearInterval(ticker);
      botMsg.totalSeconds = Math.round(Date.now() - botMsg.startedAt) / 1000;
      delete botMsg.stageTimes;
      delete botMsg.activeStage;
      state.busy = false;
      touch(conv);
      updateMessage(botMsg);
      updateComposer();
      scrollToBottom();
    }

    // Màn hình rộng: tự mở hộp trích dẫn văn bản gốc cạnh câu trả lời
    var result = botMsg.result;
    if (botMsg.status === "done" && result && result.sources && result.sources.length &&
        conv.id === state.currentId && window.innerWidth >= WIDE_SCREEN) {
      openCitePanel(botMsg.id, null);
    }
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
    if (!msg || !msg.result) {
      closeCitePanel();
      return;
    }
    var result = msg.result;
    var sources = result.sources || [];
    var cited = citedIndexes(result);
    var total = responseSeconds(msg);

    els.citeQuestion.textContent = "Câu hỏi: " + (result.question || "");
    els.citeMeta.textContent = "";
    if (total) {
      els.citeMeta.append(h("span", { class: "time-chip", title: timingTooltip(msg, result) },
        h("span", { html: ICONS.clock }), "Phản hồi trong " + formatSeconds(total)));
    }
    if (result.confidence) {
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

      var used = !!cited[src.index];
      var pdfUrl = src.doc_id ? api("/documents/" + src.doc_id + "/file") + (src.page_number ? "#page=" + src.page_number : "") : null;

      els.citeBody.append(h("article", { class: "cite-card " + (used ? "cited" : "unused"), "data-index": src.index },
        h("div", { class: "cite-card-head" },
          h("span", { class: "source-index" }, src.index),
          h("span", { class: "cite-card-label", title: src.label }, src.label),
          h("span", { class: "cite-card-score", title: "Độ liên quan (cosine)" }, Number(src.score).toFixed(2)),
          pdfUrl ? h("a", {
            class: "cite-card-open", href: pdfUrl, target: "_blank", rel: "noopener",
            title: "Mở PDF tại trang này", "aria-label": "Mở PDF tại trang này", html: ICONS.open
          }) : null
        ),
        h("p", { class: "cite-card-text" }, highlightText(text, ranges)),
        h("div", { class: "cite-card-foot" },
          h("span", { class: "cite-tag" + (used ? " used" : "") },
            used ? "Được dùng trong câu trả lời" : "Tìm thấy nhưng không dùng trong câu trả lời"),
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
      node.classList.toggle("citing", !!state.cite.msgId && node.getAttribute("data-id") === state.cite.msgId);
    });
  }

  function openCitePanel(msgId, focus) {
    state.cite.msgId = msgId;
    renderCitePanel();
    if (!state.cite.msgId) return;
    els.citePanel.hidden = false;
    els.app.classList.add("cite-open");
    markCitingMessage();
    focusCitation(focus === undefined ? null : focus);
  }

  function closeCitePanel() {
    state.cite.msgId = null;
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
      var providers = (data.llm_providers || []).map(function (name) { return name.replace(/^groq\//, ""); }).join(" → ") || "chưa cấu hình";
      var search = data.embedding_model ? "từ khoá + ngữ nghĩa (" + data.embedding_model + ")" : "từ khoá (BM25)";
      els.serverStatus.innerHTML =
        "<strong>Model:</strong> " + escapeHtml(providers) +
        "<br><strong>Tìm kiếm:</strong> " + escapeHtml(search) +
        "<br><strong>Lưu trữ:</strong> " + escapeHtml(storage);
      if (data.storage_error) showBanner(data.storage_error);
      else if (!data.llm_configured) showBanner("Server chưa có GROQ_API_KEY nên chưa thể trả lời câu hỏi. Thêm biến này trong Environment Variables rồi Redeploy.");
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

    els.attachBtn.addEventListener("click", function () { els.fileInput.click(); });
    els.docsUpload.addEventListener("click", function () { els.fileInput.click(); });
    els.fileInput.addEventListener("change", function () {
      handleFiles(els.fileInput.files);
      els.fileInput.value = "";
    });

    els.docsBtn.addEventListener("click", function () {
      els.docsPanel.hidden = !els.docsPanel.hidden;
      if (!els.docsPanel.hidden) refreshDocs();
    });
    els.closeDocs.addEventListener("click", function () { els.docsPanel.hidden = true; });

    // Bấm chip [n] hoặc chip nguồn -> mở hộp văn bản gốc tại đoạn tương ứng
    els.inner.addEventListener("click", function (event) {
      var trigger = event.target.closest(".cite, .src-chip");
      if (!trigger) return;
      openCitePanel(trigger.closest(".msg").getAttribute("data-id"), trigger.getAttribute("data-cite"));
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
        if (!els.docsPanel.hidden) els.docsPanel.hidden = true;
        else if (state.cite.msgId) closeCitePanel();
        closeSidebarMobile();
      }
    });
  }

  // ------------------------------------------------------------------ init
  function init() {
    // Upload / câu hỏi bị gián đoạn do tải lại trang
    state.conversations.forEach(function (conv) {
      conv.messages.forEach(function (msg) {
        if (msg.role === "doc" && (msg.status === "uploading" || msg.status === "processing")) {
          msg.status = "error";
          msg.error = "Tải lên bị gián đoạn, hãy tải lại file.";
        }
        if (msg.role === "doc" && msg.overviewStatus === "loading") msg.overviewStatus = "error";
        if (msg.role === "assistant" && msg.status === "pending") {
          msg.status = "error";
          msg.error = "Câu trả lời bị gián đoạn do trang được tải lại.";
        }
      });
    });

    var last = null;
    try { last = localStorage.getItem(CURRENT_KEY); } catch (e) { last = null; }
    var exists = state.conversations.some(function (c) { return c.id === last; });
    state.currentId = exists ? last : newId();

    bindEvents();
    renderThemeToggle();
    renderAll();
    refreshDocs();
    loadHealth();
  }

  init();
})();
