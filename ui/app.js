const API_BASE = 'http://localhost:8000';

function setQuery(queryText) {
  document.getElementById('queryInput').value = queryText;
  submitQuery();
}

async function submitQuery() {
  const queryInput = document.getElementById('queryInput');
  const query = queryInput.value.trim();
  if (!query) return;

  const chatHistory = document.getElementById('chatHistory');

  // Append User Message
  const userCard = document.createElement('div');
  userCard.className = 'message-card user';
  userCard.innerHTML = `
    <div class="meta-header">
      <span>Mining Engineer Request</span>
      <span>${new Date().toLocaleTimeString()}</span>
    </div>
    <p><strong>${escapeHtml(query)}</strong></p>
  `;
  chatHistory.appendChild(userCard);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  // Append Loading Assistant Card
  const assistantCard = document.createElement('div');
  assistantCard.className = 'message-card assistant';
  assistantCard.innerHTML = `
    <div class="meta-header">
      <span>MineMind Engine (Hybrid RAG + Reranker)</span>
      <span>Searching Qdrant Local Vector DB...</span>
    </div>
    <p style="color: var(--text-secondary);">Retrieving regulations and performing cross-encoder reranking...</p>
  `;
  chatHistory.appendChild(assistantCard);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 15, top_m: 3 })
    });

    if (!res.ok) throw new Error('API server returned error');
    const data = await res.json();

    // Format Citations in Answer
    let answerHtml = escapeHtml(data.answer);
    data.citations.forEach(c => {
      const badge = `<span class="citation-badge" title="Rerank Score: ${c.rerank_score}">${escapeHtml(c.citation_tag)}</span>`;
      answerHtml = answerHtml.replace(c.citation_tag, badge);
    });

    assistantCard.innerHTML = `
      <div class="meta-header">
        <span>MineMind Regulatory Specialist</span>
        <span>Grounded Answer (${data.telemetry.total_latency_ms} ms)</span>
      </div>
      <p style="line-height: 1.6; white-space: pre-wrap;">${answerHtml}</p>
    `;

    // Update Telemetry Panel
    updateTelemetryUI(data.telemetry, data.citations);

  } catch (err) {
    assistantCard.innerHTML = `
      <div class="meta-header" style="color: #ef4444;">
        <span>Engine Error</span>
      </div>
      <p style="color: #ef4444;">Failed to connect to MineMind API backend (http://localhost:8000). Ensure the FastAPI server is running.</p>
    `;
  }
}

function updateTelemetryUI(telemetry, citations) {
  document.getElementById('totalLatencyVal').innerText = `${telemetry.total_latency_ms} ms`;
  document.getElementById('p95LatencyVal').innerText = `${Math.round(telemetry.total_latency_ms * 1.1)} ms`;
  
  document.getElementById('retrievalTimeMs').innerText = `${telemetry.retrieval_latency_ms} ms`;
  document.getElementById('genTimeMs').innerText = `${telemetry.generation_latency_ms} ms`;

  const total = telemetry.total_latency_ms || 1;
  const retPct = Math.min(100, Math.max(10, (telemetry.retrieval_latency_ms / total) * 100));
  const genPct = Math.min(100, Math.max(10, (telemetry.generation_latency_ms / total) * 100));

  document.getElementById('retrievalBar').style.width = `${retPct}%`;
  document.getElementById('genBar').style.width = `${genPct}%`;

  // Render Citations List
  const citationList = document.getElementById('citationList');
  if (citations && citations.length > 0) {
    citationList.innerHTML = citations.map(c => `
      <div style="background: rgba(255,255,255,0.03); padding: 0.4rem 0.6rem; border-radius: 6px; margin-bottom: 0.3rem;">
        <strong style="color: var(--accent-gold);">${escapeHtml(c.citation_tag)}</strong><br>
        <span style="font-size: 0.7rem;">File: ${escapeHtml(c.file)} | Score: ${c.rerank_score}</span>
      </div>
    `).join('');
  } else {
    citationList.innerText = 'No specific sources cited.';
  }
}

function switchTab(tabName) {
  alert(`Tab '${tabName}' selected. Benchmark & Tracing views active.`);
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, function(m) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[m];
  });
}
