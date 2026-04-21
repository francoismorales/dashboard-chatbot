// src/ChatBot.jsx
// Widget flotante de chat. Se abre con un botón circular abajo a la derecha.
// Envía preguntas al endpoint /api/chat y muestra respuestas con animación de typing.

import { useEffect, useRef, useState } from "react";
import "./ChatBot.css";

const BASE_URL = "http://localhost:8000";

// Preguntas sugeridas que aparecen al abrir el chat por primera vez
const SUGGESTED_QUESTIONS = [
  "¿Cuál fue el pico más grande de aperturas?",
  "¿Cómo se comporta la actividad a las 8pm?",
  "¿Qué día de la semana tiene más cierres?",
  "Compara el lunes vs el domingo",
];

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll al último mensaje
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // Focus al input cuando se abre
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const sendMessage = async (text) => {
    const userMsg = { role: "user", content: text };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          // Mandamos historial previo para dar contexto al LLM
          history: messages,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Error desconocido" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg = {
        role: "assistant",
        content: data.response,
        toolCalls: data.tool_calls,
      };
      setMessages([...newHistory, assistantMsg]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    sendMessage(text);
  };

  const handleSuggestion = (q) => {
    sendMessage(q);
  };

  // ---------------- Render ----------------
  if (!isOpen) {
    return (
      <button
        className="chat-fab"
        onClick={() => setIsOpen(true)}
        title="Asistente de análisis"
        aria-label="Abrir chatbot"
      >
        <span className="chat-fab-icon">💬</span>
        <span className="chat-fab-pulse" />
      </button>
    );
  }

  return (
    <div className="chat-panel">
      <header className="chat-header">
        <div>
          <div className="chat-title">🤖 Asistente de análisis</div>
          <div className="chat-subtitle">Pregúntame sobre los datos del dashboard</div>
        </div>
        <button className="chat-close" onClick={() => setIsOpen(false)} aria-label="Cerrar">
          ×
        </button>
      </header>

      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="chat-welcome">
            <div className="chat-welcome-icon">✨</div>
            <p>Hola, analizo los datos de disponibilidad de tiendas Rappi. Puedo consultar estadísticas por hora, día, comparar períodos o encontrar eventos extremos.</p>
            <div className="chat-suggestions">
              <div className="chat-suggestions-label">Prueba preguntando:</div>
              {SUGGESTED_QUESTIONS.map((q, i) => (
                <button key={i} className="chat-suggestion" onClick={() => handleSuggestion(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`chat-msg chat-msg-${m.role}`}>
            <div className="chat-bubble">{m.content}</div>
            {m.toolCalls && m.toolCalls.length > 0 && (
              <div className="chat-tools">
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-msg chat-msg-assistant">
            <div className="chat-bubble chat-typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}

        {error && (
          <div className="chat-error">
            ❌ {error}
          </div>
        )}
      </div>

      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          placeholder="Escribe tu pregunta..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="chat-send" disabled={loading || !input.trim()}>
          ➤
        </button>
      </form>
    </div>
  );
}
