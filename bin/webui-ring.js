// Context ring for Open WebUI — injected into frontend/index.html by bin/webui.
//
// Docks in the prompt-bar toolbar, just left of the model selector dropdown: a
// small ring that fills as the chat's context grows against the model's
// 262,144-token window. Hover for numbers; click to compact the conversation now
// via Open WebUI's native endpoint. Auto-compaction is also enabled server-side,
// so the ring is awareness + a manual trigger, not the only safety net.
//
// The prompt bar is rebuilt on every SPA navigation — and doesn't exist yet when
// this script first runs — so a MutationObserver keeps re-running dock(): it
// re-attaches the ring when it falls out of the DOM, upgrades it out of the
// floating fallback once the toolbar renders, and moves it up the anchor list if
// a better anchor appears. If every known anchor id vanishes in an upgrade, the
// ring falls back to a fixed corner position.
//
// Data: GET /api/v1/chats/{id} -> context_usage {tokens, threshold}, using the
// user's own session token from localStorage.
(() => {
  const MAX_CTX = 262144;
  const POLL_MS = 8000;
  let busy = false;
  let lastTokens = null;
  let ring = null;
  let tip = null;

  const css = `
  #globus-ring{width:26px;height:26px;cursor:pointer;opacity:.8;flex:none;
    align-self:center;margin:0 6px;transition:opacity .15s;display:none}
  #globus-ring:hover{opacity:1}
  #globus-ring.floating{position:fixed;right:14px;bottom:86px;z-index:2147483000}
  #globus-ring.busy{animation:globus-spin 1s linear infinite}
  @keyframes globus-spin{to{transform:rotate(360deg)}}
  #globus-ring-tip{position:fixed;z-index:2147483001;max-width:280px;
    padding:.5rem .7rem;border-radius:8px;font:12.5px/1.45 system-ui,sans-serif;
    background:#1B1F26;color:#E4E8EC;border:1px solid #2A3038;display:none;
    box-shadow:0 4px 14px rgba(0,0,0,.25);pointer-events:none}
  @media (prefers-color-scheme: light){
    #globus-ring-tip{background:#fff;color:#1B1F26;border-color:#DCE1E7}}
  `;

  function fill(frac) {
    const arc = ring.querySelector('.arc');
    const C = 2 * Math.PI * 10;
    arc.setAttribute('stroke-dasharray', `${C * Math.min(1, frac)} ${C}`);
    arc.setAttribute('stroke',
      frac > 0.85 ? '#B4423C' : frac > 0.6 ? '#C08A1E' : '#2FA98C');
  }

  function chatId() {
    const m = location.pathname.match(/\/c\/([0-9a-fA-F-]+)/);
    return m ? m[1] : null;
  }
  const token = () => localStorage.getItem('token');

  function placeTip() {
    const r = ring.getBoundingClientRect();
    tip.style.right = Math.max(8, window.innerWidth - r.right) + 'px';
    tip.style.bottom = Math.max(8, window.innerHeight - r.top + 10) + 'px';
  }
  function setTip(html) {
    tip.innerHTML = html;
    if (tip.style.display === 'block') placeTip();
  }
  function tipDefault() {
    if (lastTokens == null) return;
    const pct = ((100 * lastTokens) / MAX_CTX).toFixed(1);
    setTip(`<b>${lastTokens.toLocaleString()} / ${MAX_CTX.toLocaleString()}</b> ` +
      `context tokens (${pct}%)<br>auto-compacts at 100,000 &middot; ` +
      `<u>click to compact now</u>`);
  }

  async function refresh() {
    const id = chatId();
    if (!id || !token()) { ring.style.display = 'none'; return; }
    try {
      const r = await fetch(`/api/v1/chats/${id}`,
        { headers: { Authorization: `Bearer ${token()}` } });
      if (!r.ok) { ring.style.display = 'none'; return; }
      const d = await r.json();
      const u = d && d.context_usage;
      lastTokens = u && typeof u.tokens === 'number' ? u.tokens : null;
      if (lastTokens == null) { ring.style.display = 'none'; return; }
      ring.style.display = 'block';
      fill(lastTokens / MAX_CTX);
      tipDefault();
    } catch (e) { /* transient — keep last state */ }
  }

  async function compact() {
    const id = chatId();
    if (!id || busy) return;
    busy = true;
    ring.classList.add('busy');
    setTip('Compacting&hellip; (summarises the conversation; can take a minute)');
    try {
      const r = await fetch(`/api/v1/chats/${id}/compact`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token()}`,
                   'Content-Type': 'application/json' },
        body: '{}',
      });
      if (r.status === 409) {
        setTip('Wait for the current response to finish, then try again.');
      } else if (!r.ok) {
        setTip(`Compaction failed (HTTP ${r.status}).`);
      } else {
        const d = await r.json();
        const u = d && d.context_usage;
        if (u && typeof u.tokens === 'number') { lastTokens = u.tokens; fill(u.tokens / MAX_CTX); }
        setTip(d && d.compacted
          ? 'Compacted. Older turns are now a summary; recent turns kept verbatim.'
          : 'Nothing to compact yet — the chat is below the threshold.');
      }
    } catch (e) {
      setTip('Compaction failed: ' + e);
    }
    busy = false;
    ring.classList.remove('busy');
    setTimeout(tipDefault, 6000);
  }

  function dock() {
    // Best anchor first: just left of the model selector dropdown in the prompt
    // bar, then the Dictate (voice) button, then the send button. Idempotent —
    // safe to call on every mutation tick; it only touches the DOM when the ring
    // isn't already sitting before the best available anchor.
    const anchor = document.getElementById('model-selector-0-button')
                || document.getElementById('voice-input-button')
                || document.getElementById('send-message-button');
    if (anchor && anchor.parentElement) {
      if (anchor.previousElementSibling === ring) return;
      ring.classList.remove('floating');
      anchor.parentElement.insertBefore(ring, anchor);
    } else if (!ring.isConnected) {
      ring.classList.add('floating');
      document.body.appendChild(ring);
    }
  }

  function mount() {
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);

    tip = document.createElement('div');
    tip.id = 'globus-ring-tip';
    document.body.appendChild(tip);

    ring = document.createElement('div');
    ring.id = 'globus-ring';
    ring.innerHTML =
      `<svg viewBox="0 0 26 26" width="26" height="26">
        <circle cx="13" cy="13" r="10" fill="none" stroke="#8884" stroke-width="3.5"/>
        <circle class="arc" cx="13" cy="13" r="10" fill="none" stroke="#2FA98C"
          stroke-width="3.5" stroke-linecap="round" stroke-dasharray="0 63"
          transform="rotate(-90 13 13)"/>
      </svg>`;
    ring.addEventListener('mouseenter', () => { tip.style.display = 'block'; placeTip(); });
    ring.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    ring.addEventListener('click', compact);
    dock();

    // The SPA rebuilds the input bar constantly, and at first mount it hasn't
    // rendered at all (which used to strand the ring in the floating corner).
    // Re-run dock() on mutations, throttled; dock() itself no-ops when placed.
    let pending = false;
    new MutationObserver(() => {
      if (pending) return;
      pending = true;
      setTimeout(() => { pending = false; dock(); }, 250);
    }).observe(document.body, { childList: true, subtree: true });

    refresh();
    setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
    const push = history.pushState.bind(history);
    history.pushState = (...a) => { push(...a); setTimeout(() => { dock(); refresh(); }, 400); };
    window.addEventListener('popstate', () => setTimeout(() => { dock(); refresh(); }, 400));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
