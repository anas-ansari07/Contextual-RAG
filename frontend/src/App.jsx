
import { useRef, useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Something went wrong. Please try again."
    );
  }

  return data;
}

function Icon({ name, size = 18 }) {
  const paths = {
    sparkles: (
      <>
        <path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" />
        <path d="m19 14 1.1 2.9L23 18l-2.9 1.1L19 22l-1.1-2.9L15 18l2.9-1.1L19 14Z" />
      </>
    ),
    file: (
      <>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
        <path d="M14 2v6h6M8 13h8M8 17h8" />
      </>
    ),
    upload: (
      <>
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <path d="m17 8-5-5-5 5M12 3v12" />
      </>
    ),
    send: <path d="m22 2-7 20-4-9-9-4 20-7ZM22 2 11 13" />,
    plus: <path d="M12 5v14M5 12h14" />,
    chevron: <path d="m6 9 6 6 6-6" />,
    check: <path d="m5 12 4 4L19 6" />,
    book: (
      <>
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" />
      </>
    ),
    message: (
      <>
        <path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z" />
      </>
    ),
    close: <path d="m18 6-12 12M6 6l12 12" />,
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="source-card">
      <button
        className="source-toggle"
        type="button"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        <span className="source-icon">
          <Icon name="file" size={17} />
        </span>

        <span className="source-info">
          <strong>
            {source.filename || "PDF document"}
          </strong>
          <span>
            Page {source.page_number ?? "—"}
            {source.source_id != null && ` · Source ${source.source_id}`}
          </span>
        </span>

        <Icon name="chevron" size={17} />
      </button>

      {expanded && (
        <div className="source-details">
          <p>{source.text || "No passage text available."}</p>
          {source.distance != null && (
            <small>
              Retrieval distance:{" "}
              {typeof source.distance === "number"
                ? source.distance.toFixed(4)
                : source.distance}
            </small>
          )}
          {source.chunk_index != null && (
            <small>Chunk {source.chunk_index}</small>
          )}
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [documentInfo, setDocumentInfo] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [error, setError] = useState("");
  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const bottomRef = useRef(null);

  const busy = uploadBusy || chatBusy;

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0] ?? null;

    if (selectedFile && !selectedFile.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      setFile(null);
      return;
    }

    setError("");
    setFile(selectedFile);
    setDocumentInfo(null);
  }

  async function uploadDocument() {
    if (!file || busy) return;

    setUploadBusy(true);
    setError("");

    try {
      const body = new FormData();
      body.append("file", file);

      const response = await fetch(`${API}/documents/upload`, {
        method: "POST",
        body,
      });

      const data = await readResponse(response);
      setDocumentInfo(data);
      setMessages([]);
    } catch (err) {
      setError(err.message || "Unable to upload this PDF.");
    } finally {
      setUploadBusy(false);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();

    const submittedQuestion = question.trim();
    if (!submittedQuestion || chatBusy || uploadBusy) return;

    setError("");
    setQuestion("");
    setChatBusy(true);

    setMessages((current) => [
      ...current,
      {
        id: `${Date.now()}-user`,
        role: "user",
        content: submittedQuestion,
      },
    ]);

    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    });

    try {
      const response = await fetch(`${API}/chat/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: submittedQuestion,
          top_k: 5,
          retrievalMode,
        }),
      });

      const data = await readResponse(response);

      setMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          content: data.answer || "No answer was returned.",
          sources: Array.isArray(data.sources) ? data.sources : [],
          mode: data.retrieval_mode || retrievalMode,
        },
      ]);

      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      });
    } catch (err) {
      setError(err.message || "Unable to get an answer.");
    } finally {
      setChatBusy(false);
      textareaRef.current?.focus();
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  function startNewChat() {
    if (busy) return;
    setMessages([]);
    setQuestion("");
    setError("");
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={(event) => event.preventDefault()}>
          <span className="brand-mark">
            <Icon name="sparkles" size={23} />
          </span>
          <span>
            <strong>ContextIQ</strong>
            <small>PERSONAL KNOWLEDGE AI</small>
          </span>
        </a>

        <button className="new-chat-button" onClick={startNewChat} disabled={busy}>
          <Icon name="plus" size={18} />
          New conversation
        </button>

        <div className="sidebar-section-label">WORKSPACE</div>

        <div className="sidebar-document">
          <span className="sidebar-document-icon">
            <Icon name="book" size={18} />
          </span>
          <span className="sidebar-document-copy">
            <strong>Your knowledge base</strong>
            <small>
              {documentInfo
                ? documentInfo.filename
                : "No document indexed yet"}
            </small>
          </span>
          {documentInfo && (
            <span className="status-dot" title="Document indexed" />
          )}
        </div>

        <div className="sidebar-bottom">
          <span className="online-indicator" />
          <span>Local API connection</span>
          <span className="sidebar-version">BETA</span>
        </div>
      </aside>

      <section className="main-workspace">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>Document chat</strong>
          </div>

          <div className="topbar-status">
            <span className="status-dot" />
            AI assistant
          </div>
        </header>

        <div className={`chat-area ${messages.length ? "has-messages" : ""}`}>
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">
                <Icon name="sparkles" size={28} />
              </div>

              <span className="eyebrow">YOUR DOCUMENTS, UNDERSTOOD</span>
              <h1>Knowledge, without<br />the digging.</h1>
              <p className="welcome-description">
                Upload a PDF, ask a question, and get grounded answers
                with evidence you can verify.
              </p>

              <div className="welcome-cards">
                <button
                  className="welcome-card"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <span className="welcome-card-icon purple">
                    <Icon name="upload" size={19} />
                  </span>
                  <strong>Bring your knowledge</strong>
                  <span>Upload a PDF and index its content.</span>
                </button>

                <button
                  className="welcome-card"
                  onClick={() => {
                    setQuestion("Summarize the key points in my document.");
                    textareaRef.current?.focus();
                  }}
                >
                  <span className="welcome-card-icon blue">
                    <Icon name="message" size={19} />
                  </span>
                  <strong>Explore your document</strong>
                  <span>Ask questions and inspect source passages.</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="conversation">
              <div className="conversation-heading">
                <div>
                  <span className="eyebrow">CONVERSATION</span>
                  <h2>Let's explore your knowledge</h2>
                </div>
                <span className="message-count">
                  {messages.filter((message) => message.role === "user").length}{" "}
                  {messages.filter((message) => message.role === "user").length === 1
                    ? "question"
                    : "questions"}
                </span>
              </div>

              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <div className={`message-avatar ${message.role}`}>
                    {message.role === "assistant" ? (
                      <Icon name="sparkles" size={18} />
                    ) : (
                      <span>Y</span>
                    )}
                  </div>

                  <div className="message-body">
                    <div className="message-meta">
                      <strong>
                        {message.role === "assistant" ? "ContextIQ" : "You"}
                      </strong>
                      {message.mode && (
                        <span className="mode-chip">
                          {message.mode} retrieval
                        </span>
                      )}
                    </div>

                    <div className="message-content">{message.content}</div>

                    {message.role === "assistant" && message.sources?.length > 0 && (
                      <div className="sources-section">
                        <div className="sources-heading">
                          <span className="sources-heading-icon">
                            <Icon name="book" size={16} />
                          </span>
                          <strong>Sources</strong>
                          <span>{message.sources.length} passages</span>
                        </div>

                        <div className="sources-list">
                          {message.sources.map((source, index) => (
                            <SourceCard
                              key={`${source.source_id ?? index}-${index}`}
                              source={source}
                              index={index}
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </article>
              ))}

              {chatBusy && (
                <div className="message assistant">
                  <div className="message-avatar assistant">
                    <Icon name="sparkles" size={18} />
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>ContextIQ</strong>
                      <span className="thinking-label">Thinking</span>
                    </div>
                    <div className="thinking-indicator">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <footer className="composer-area">
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button
                type="button"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <Icon name="close" size={16} />
              </button>
            </div>
          )}

          <section className="document-bar">
            <div className="document-picker">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
                hidden
              />

              <button
                type="button"
                className="file-select-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={busy}
              >
                <span className="file-select-icon">
                  <Icon name="file" size={17} />
                </span>
                <span className="file-select-text">
                  <strong>
                    {file?.name || documentInfo?.filename || "Choose a PDF"}
                  </strong>
                  <small>
                    {file
                      ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to upload`
                      : documentInfo
                        ? `${documentInfo.chunks_indexed ?? " "} chunks indexed`
                        : "PDF documents only"}
                  </small>
                </span>
              </button>

              <button
                type="button"
                className="upload-button"
                onClick={uploadDocument}
                disabled={!file || busy}
              >
                {uploadBusy ? (
                  "Indexing..."
                ) : documentInfo && documentInfo.filename === file?.name ? (
                  <>
                    <Icon name="check" size={15} />
                    Indexed
                  </>
                ) : (
                  <>
                    <Icon name="upload" size={15} />
                    Index PDF
                  </>
                )}
              </button>
            </div>

            <div className="retrieval-picker">
              <label htmlFor="retrieval-mode">RETRIEVAL</label>
              <select
                id="retrieval-mode"
                value={retrievalMode}
                onChange={(event) => setRetrievalMode(event.target.value)}
                disabled={busy}
              >
                <option value="baseline">Baseline RAG</option>
                <option value="contextual">Contextual RAG</option>
                <option value="hybrid">Hybrid RAG</option>
              </select>
            </div>
          </section>

          <form className="composer" onSubmit={askQuestion}>
            <textarea
              ref={textareaRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="Ask anything about your documents..."
              rows={2}
              disabled={busy}
              aria-label="Your question"
            />

            <div className="composer-bottom">
              <span className="composer-hint">
                <kbd>Enter</kbd> to send
                <span className="hint-divider">·</span>
                <kbd>Shift + Enter</kbd> for a new line
              </span>

              <button
                className="send-button"
                type="submit"
                disabled={busy || !question.trim()}
                aria-label="Send question"
              >
                {chatBusy ? (
                  <span className="send-spinner" />
                ) : (
                  <Icon name="send" size={17} />
                )}
                <span>{chatBusy ? "Thinking" : "Ask"}</span>
              </button>
            </div>
          </form>

          <p className="disclaimer">
            ContextIQ can make mistakes. Verify important information against
            the cited source passages.
          </p>
        </footer>
      </section>
    </main>
  );
}
