import { useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [documentInfo, setDocumentInfo] = useState(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function uploadDocument() {
    if (!file) return;

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("file", file);

      const response = await fetch(`${API}/documents/upload`, {
        method: "POST",
        body,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setDocumentInfo(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    if (!question.trim()) return;

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API}/chat/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5 }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Question failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <header>
        <span className="eyebrow">AI KNOWLEDGE ASSISTANT</span>
        <h1>ContextIQ</h1>
        <p>Ask questions. Explore your documents. Verify every answer.</p>
      </header>

      <section className="panel">
        <h2>1. Your document</h2>
        <input
          type="file"
          accept=".pdf,application/pdf"
          onChange={(event) => {
            setFile(event.target.files?.[0] ?? null);
            setDocumentInfo(null);
            setResult(null);
          }}
        />
        <button onClick={uploadDocument} disabled={!file || busy}>
          {busy ? "Processing..." : "Upload and index PDF"}
        </button>

        {documentInfo && (
          <p className="success">
            Indexed {documentInfo.filename}:{" "}
            {documentInfo.chunks_indexed} chunks
          </p>
        )}
      </section>

      <section className="panel">
        <h2>2. Ask your document</h2>
        <form onSubmit={askQuestion}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question about your uploaded PDF..."
            rows={3}
          />
          <button disabled={busy || !question.trim()}>
            Ask ContextIQ
          </button>
        </form>
      </section>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="panel">
          <h2>Answer</h2>
          <p className="answer">{result.answer}</p>

          <h3>Retrieved evidence</h3>
          {result.sources.map((source) => (
            <article className="source" key={source.source_id}>
              <strong>
                [{source.source_id}] {source.filename}
                {" — Page "}{source.page_number}
              </strong>
              <p>{source.text}</p>
              <small>
                Retrieval distance: {source.distance}
              </small>
            </article>
          ))}
        </section>
      )}
    </main>
  );
}
