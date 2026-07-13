// TailorTwin scan GUI — client logic.
//
// Reads bootstrap config from the global window.TAILORTWIN_CFG that
// index.html renders, wires up the form, the SSE log stream, and
// the how-to modal.

(function () {
  const cfg = window.TAILORTWIN_CFG || {};
  const $ = (id) => document.getElementById(id);

  const slug = (s) =>
    (s || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "") || "scan";

  // --- Auto-derived output prefix ----------------------------------------
  let prefixEdited = false;
  $("out_prefix").addEventListener("input", () => (prefixEdited = true));
  function refreshPrefix() {
    if (prefixEdited) return;
    const name = slug($("person").value);
    const d = $("scan_date").value || cfg.today;
    $("out_prefix").value =
      cfg.defaultResults + "/" + name + "_" + d.replaceAll("-", "");
  }
  ["person", "scan_date"].forEach((id) =>
    $(id).addEventListener("input", refreshPrefix),
  );
  refreshPrefix();

  // --- Run / cancel + SSE log stream -------------------------------------
  let evtSource = null;

  // Toggle the action buttons. ``activeId`` is the button driving the
  // current job (gets the spinner + a status label); the others are just
  // disabled. Pass running=false to reset everything to idle.
  function setBusy(running, activeId, label) {
    $("run-btn").disabled = running;
    $("preflight-btn").disabled = running;
    $("cancel-btn").disabled = !running;
    $("run-btn").classList.toggle("loading", running && activeId === "run-btn");
    if (running && activeId) {
      $(activeId).querySelector(".btn-label").textContent = label;
    } else {
      $("run-btn").querySelector(".btn-label").textContent = "Run scan";
      $("preflight-btn").querySelector(".btn-label").textContent =
        "Check capture";
    }
  }

  // Start a backend job (scan or preflight) and stream its log via SSE.
  async function startJob(url, activeId, runLabel) {
    $("log").textContent = "";
    setLogVisible(true);
    setBusy(true, activeId, "Starting…");
    const data = new FormData($("form"));
    const r = await fetch(url, { method: "POST", body: data });
    const j = await r.json();
    if (!j.ok) {
      $("log").textContent = "ERROR: " + j.error + "\n";
      setBusy(false);
      return;
    }
    setBusy(true, activeId, runLabel);
    evtSource = new EventSource("/stream");
    evtSource.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.line !== undefined) {
        $("log").textContent += m.line;
        $("log").scrollTop = $("log").scrollHeight;
      }
      if (m.done) {
        evtSource.close();
        evtSource = null;
        setBusy(false);
      }
    };
  }

  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    startJob("/run", "run-btn", "Running…");
  });

  $("preflight-btn").addEventListener("click", () =>
    startJob("/preflight", "preflight-btn", "Checking…"),
  );

  $("cancel-btn").addEventListener("click", async () => {
    await fetch("/cancel", { method: "POST" });
    setBusy(true, null, "Cancelling…");
  });

  // --- Terminal toggle ---------------------------------------------------
const logToggle = $("log-toggle");
const logPanel = $("log-panel");
function setLogVisible(visible) {
  logPanel.hidden = !visible;
  logToggle.setAttribute("aria-expanded", String(visible));
  logToggle.querySelector(".log-toggle-label").textContent =
    visible ? "Hide terminal" : "Show terminal";
}
logToggle.addEventListener("click", () =>
  setLogVisible(logPanel.hidden));

// --- How-to modal ------------------------------------------------------
  const howto = $("howto");
  const openHowto = () => {
    howto.hidden = false;
    document.body.style.overflow = "hidden";
  };
  const closeHowto = () => {
    howto.hidden = true;
    document.body.style.overflow = "";
  };
  $("howto-btn").addEventListener("click", openHowto);
  $("howto-close").addEventListener("click", closeHowto);
  howto
    .querySelector(".modal-backdrop")
    .addEventListener("click", closeHowto);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !howto.hidden) closeHowto();
  });
})();
