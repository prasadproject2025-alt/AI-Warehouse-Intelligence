import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, BarChart3, Bot, CheckCircle2, ClipboardCheck, Crosshair, Eye,
  FileVideo, GraduationCap, Layers, Loader2, MapPin, RefreshCw, Search, Send,
  Radio, ShieldAlert, Sparkles, Square, Upload, Video, XCircle,
} from 'lucide-react';

/* ------------------------------------------------------------------ helpers */

const RISK_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

const titleCase = (s) => (s || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
const fmtSec = (s) => `${Number(s || 0).toFixed(2)}s`;

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON error body */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

async function apiSend(path, body, method = 'POST') {
  const res = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON error body */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

/** Minimal markdown renderer for assistant answers (bold, code, headings, lists). */
function Markdown({ text }) {
  const blocks = useMemo(() => {
    const inline = (s) => {
      const parts = [];
      const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
      let last = 0, m;
      while ((m = re.exec(s)) !== null) {
        if (m.index > last) parts.push(s.slice(last, m.index));
        const tok = m[0];
        if (tok.startsWith('**')) parts.push(<strong key={parts.length}>{tok.slice(2, -2)}</strong>);
        else if (tok.startsWith('`')) parts.push(<code key={parts.length} className="md-code">{tok.slice(1, -1)}</code>);
        else parts.push(<em key={parts.length}>{tok.slice(1, -1)}</em>);
        last = m.index + tok.length;
      }
      if (last < s.length) parts.push(s.slice(last));
      return parts;
    };
    return (text || '').split('\n').map((line, i) => {
      if (line.startsWith('### ')) return <h4 key={i} className="md-h">{inline(line.slice(4))}</h4>;
      if (line.startsWith('## ')) return <h3 key={i} className="md-h">{inline(line.slice(3))}</h3>;
      if (/^\s*[-*]\s/.test(line)) return <div key={i} className="md-li">{inline(line.replace(/^\s*[-*]\s/, ''))}</div>;
      if (/^\s*\d+\.\s/.test(line)) return <div key={i} className="md-li">{inline(line.trim())}</div>;
      if (!line.trim()) return <div key={i} className="md-gap" />;
      return <div key={i} className="md-p">{inline(line)}</div>;
    });
  }, [text]);
  return <div className="md">{blocks}</div>;
}

function RiskTag({ level }) {
  return <span className={`risk-tag ${level}`}>{level}</span>;
}

function StatusPill({ status }) {
  const map = {
    IMPLEMENTED: ['ok', 'Implemented'],
    PARTIALLY_IMPLEMENTED: ['warn', 'Partial'],
    REQUIRES_ADDITIONAL_FOOTAGE: ['info', 'Needs footage'],
    REQUIRES_MODEL_TRAINING: ['info', 'Needs training'],
    REQUIRES_ZONE_CONFIGURATION: ['info', 'Needs config'],
  };
  const [cls, label] = map[status] || ['info', status];
  return <span className={`status-pill ${cls}`}>{label}</span>;
}

function EmptyState({ icon: Icon, title, hint }) {
  return (
    <div className="empty-state">
      <Icon size={30} />
      <span className="empty-title">{title}</span>
      {hint && <span className="empty-hint">{hint}</span>}
    </div>
  );
}

function Bar({ value, max, tone = 'blue' }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="bar-track"><div className={`bar-fill ${tone}`} style={{ width: `${pct}%` }} /></div>
  );
}

/* --------------------------------------------------------------------- app */

export default function App() {
  const [activeTab, setActiveTab] = useState('operations');
  const [health, setHealth] = useState(null);
  const [apiError, setApiError] = useState(null);

  const [videos, setVideos] = useState([]);
  const [videosLoading, setVideosLoading] = useState(true);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [incidentsLoading, setIncidentsLoading] = useState(false);
  const [selectedIncident, setSelectedIncident] = useState(null);

  const [analytics, setAnalytics] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [prevention, setPrevention] = useState(null);

  const [riskFilter, setRiskFilter] = useState('ALL');
  const [behaviourFilter, setBehaviourFilter] = useState('ALL');
  const [search, setSearch] = useState('');
  const [showAnnotated, setShowAnnotated] = useState(true);

  const [messages, setMessages] = useState([{
    role: 'assistant',
    text: 'I answer only from events the vision pipeline actually recorded. If there is no evidence for a question, I will say so rather than guess.\n\nTry a question below, or ask your own.',
  }]);
  const [inputQuery, setInputQuery] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');


  const [uploadError, setUploadError] = useState('');
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [sceneForm, setSceneForm] = useState({
    bay: 'Dock 01', shift: 'Shift A', camera_id: 'CAM-01',
    floor_condition: 'unknown', dock_transfer: false,
  });

  const videoRef = useRef(null);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  /* ------------------------------------------------------------- data load */

  const refreshAnalytics = useCallback(async () => {
    try {
      const [a, p] = await Promise.all([apiGet('/api/analytics'), apiGet('/api/prevention')]);
      setAnalytics(a); setPrevention(p); setApiError(null);
    } catch (err) { setApiError(err.message); }
  }, []);

  const selectVideo = useCallback(async (video) => {
    setSelectedVideo(video);
    setIncidentsLoading(true);
    try {
      const data = await apiGet(`/api/videos/${video.id}`);
      setIncidents(data.incidents || []);
      setSelectedIncident(data.incidents?.[0] || null);
      setApiError(null);
    } catch (err) {
      setApiError(err.message); setIncidents([]); setSelectedIncident(null);
    } finally { setIncidentsLoading(false); }
  }, []);

  const refreshVideos = useCallback(async (keepSelection = true) => {
    setVideosLoading(true);
    try {
      const data = await apiGet('/api/videos');
      setVideos(data.videos || []);
      setApiError(null);
      if (data.videos?.length) {
        const keep = keepSelection && selectedVideo
          ? data.videos.find((v) => v.id === selectedVideo.id)
          : null;
        if (!keep) await selectVideo(data.videos[0]);
      } else {
        setSelectedVideo(null); setIncidents([]); setSelectedIncident(null);
      }
    } catch (err) { setApiError(err.message); }
    finally { setVideosLoading(false); }
  }, [selectVideo, selectedVideo]);

  // ---- live monitor -------------------------------------------------------
  const [liveSources, setLiveSources] = useState(null);
  const [liveSession, setLiveSession] = useState(null);
  const [liveError, setLiveError] = useState('');
  const [liveStarting, setLiveStarting] = useState(false);
  const [liveForm, setLiveForm] = useState({
    source_kind: 'file', source: '', camera_id: 'CAM-LIVE-01',
    bay: 'Dock 09 - Inside', shift: 'Shift A',
    floor_condition: 'unknown', dock_transfer: false,
  });

  useEffect(() => {
    if (activeTab !== 'live' || liveSources) return;
    fetch('/api/live/sources').then(r => r.json()).then(d => {
      setLiveSources(d);
      setLiveForm(f => ({ ...f, source: f.source || (d.library && d.library[0]) || '' }));
      const running = (d.active || []).find(x => x.status === 'running');
      if (running) setLiveSession(running);
    }).catch(() => setLiveError('Could not reach the server.'));
  }, [activeTab, liveSources]);

  // Poll the running session for telemetry and new alerts.
  useEffect(() => {
    if (!liveSession || !['running', 'starting'].includes(liveSession.status)) return;
    const id = setInterval(async () => {
      try {
        const r = await fetch(`/api/live/${liveSession.session_id}`);
        if (!r.ok) return;
        const d = await r.json();
        setLiveSession(d);
        if (d.event_count > 0) { refreshAnalytics(); }
      } catch { /* transient; the next tick retries */ }
    }, 2000);
    return () => clearInterval(id);
  }, [liveSession, refreshAnalytics]);

  const startLive = async () => {
    setLiveStarting(true); setLiveError('');
    try {
      const r = await fetch('/api/live/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(liveForm),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Could not start the live session');
      setLiveSession(d);
    } catch (e) { setLiveError(e.message); }
    finally { setLiveStarting(false); }
  };

  const stopLive = async () => {
    if (!liveSession) return;
    try {
      const r = await fetch(`/api/live/${liveSession.session_id}/stop`, { method: 'POST' });
      setLiveSession(await r.json());
    } catch { setLiveError('Could not stop the session.'); }
    refreshVideos(); refreshAnalytics();
  };

  useEffect(() => {
    (async () => {
      try { setHealth(await apiGet('/api/health')); } catch (err) { setApiError(err.message); }
      try { setCapabilities(await apiGet('/api/capabilities')); } catch { /* non-fatal */ }
      await refreshVideos(false);
      await refreshAnalytics();
    })();
    // Intentionally run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeTab === 'assistant') {
      const t = setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);
      return () => clearTimeout(t);
    }
  }, [messages, isChatLoading, activeTab]);

  /* Poll the analysis task while one is running. */
  useEffect(() => {
    if (!activeTaskId) return undefined;
    const id = setInterval(async () => {
      try {
        const data = await apiGet(`/api/videos/${activeTaskId}/status`);
        if (data.status === 'processing') {
          setUploadProgress(data.progress_percent || 1);
          setUploadStatus(
            `${data.stage || 'analysing'} — ${data.progress_percent || 0}% ` +
            `(frame ${data.current_frame || 0}/${data.total_frames || '?'}), ` +
            `${data.incidents_count || 0} risk event(s) so far`
          );
        } else if (data.status === 'completed') {
          setUploadProgress(100);
          setUploadStatus(`Analysis complete — ${data.incidents_count} risk event(s) detected.`);
          setActiveTaskId(null); setIsUploading(false);
          await refreshVideos(false); await refreshAnalytics();
          setCapabilities(await apiGet('/api/capabilities'));
          setActiveTab('operations');
        } else if (data.status === 'failed' || data.status === 'not_found') {
          setActiveTaskId(null); setIsUploading(false);
          setUploadError(data.error || 'Analysis failed. Check the server log for details.');
          setUploadStatus('');
        }
      } catch (err) { setUploadError(err.message); }
    }, 1500);
    return () => clearInterval(id);
  }, [activeTaskId, refreshVideos, refreshAnalytics]);

  /* ------------------------------------------------------------- handlers */

  const seekTo = (sec) => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = Math.max(0, sec - 1.0);
    el.play().catch(() => { /* autoplay may be blocked; user can press play */ });
  };

  const handleIncidentClick = (inc) => { setSelectedIncident(inc); seekTo(inc.timestamp_sec); };

  const sendAssistantMessage = async (queryText) => {
    const q = (queryText ?? inputQuery).trim();
    if (!q || isChatLoading) return;
    const next = [...messages, { role: 'user', text: q }];
    setMessages(next);
    if (!queryText) setInputQuery('');
    setIsChatLoading(true);
    try {
      const data = await apiSend('/api/assistant/chat', { query: q, video_id: selectedVideo?.id });
      setMessages([...next, {
        role: 'assistant', text: data.response,
        meta: { count: data.relevant_count, intent: data.intent, total: data.total_incidents_in_db },
      }]);
    } catch (err) {
      setMessages([...next, { role: 'assistant', text: `Could not reach the assistant service. ${err.message}`, error: true }]);
    } finally { setIsChatLoading(false); }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true); setUploadProgress(2); setUploadError('');
    setUploadStatus(`Uploading ${file.name}…`);
    const form = new FormData();
    form.append('file', file);
    Object.entries(sceneForm).forEach(([k, v]) => form.append(k, String(v)));
    try {
      const res = await fetch('/api/videos/upload', { method: 'POST', body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
      setUploadStatus(data.message || 'Upload accepted. Analysis started.');
      setActiveTaskId(data.video_id);
    } catch (err) {
      setUploadError(err.message); setIsUploading(false); setUploadStatus('');
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const reviewIncident = async (status) => {
    if (!selectedIncident) return;
    try {
      const updated = await apiSend(`/api/incidents/${selectedIncident.id}/review`, { status }, 'PATCH');
      setSelectedIncident(updated);
      setIncidents((list) => list.map((i) => (i.id === updated.id ? updated : i)));
      await refreshAnalytics();
    } catch (err) { setApiError(err.message); }
  };

  /* --------------------------------------------------------------- derived */

  const behaviourOptions = useMemo(
    () => Array.from(new Set(incidents.map((i) => i.behaviour_type))).sort(),
    [incidents],
  );

  const filteredIncidents = useMemo(() => {
    const q = search.trim().toLowerCase();
    return incidents.filter((i) => {
      if (riskFilter !== 'ALL' && i.risk_level !== riskFilter) return false;
      if (behaviourFilter !== 'ALL' && i.behaviour_type !== behaviourFilter) return false;
      if (q && !(`${i.behaviour_type} ${i.evidence_description} ${i.root_cause} ${i.bay}`)
        .toLowerCase().includes(q)) return false;
      return true;
    });
  }, [incidents, riskFilter, behaviourFilter, search]);

  const activeSrc = showAnnotated && selectedVideo?.annotated_video_url
    ? selectedVideo.annotated_video_url
    : selectedVideo?.video_url;

  /* ----------------------------------------------------------------- views */

  const live = liveSession;
  const liveRunning = !!live && ['running', 'starting'].includes(live.status);

  return (
    <div className="app-container">
      <header className="navbar">
        <div className="nav-brand">
          <div className="brand-badge">GEG HACKATHON</div>
          <div className="brand-title">
            <span>VisionGuard</span>
            <span className="brand-tag">WAREHOUSE FIELD INTELLIGENCE</span>
          </div>
        </div>
        <div className="nav-actions">
          <div className={`metric-pill ${health ? '' : 'offline'}`}>
            <span className={`dot ${health ? 'live' : 'dead'}`} />
            <span>{health ? 'Backend online' : 'Backend unreachable'}</span>
          </div>
          {health && (
            <div className="metric-pill">
              <span>Perception:&nbsp;<strong>
                {health.open_vocabulary ? 'YOLO-World open-vocabulary' : 'YOLOv8 COCO (fallback)'}
              </strong></span>
            </div>
          )}
          <button className="icon-btn" title="Refresh data"
            onClick={() => { refreshVideos(true); refreshAnalytics(); }}>
            <RefreshCw size={15} />
          </button>
        </div>
      </header>

      {apiError && (
        <div className="alert-bar error">
          <XCircle size={15} /> <span>{apiError}</span>
          <button onClick={() => setApiError(null)}>Dismiss</button>
        </div>
      )}

      <nav className="tab-row">
        {[
          ['operations', Video, 'Operations & Replay'],
          ['live', Radio, 'Live Monitor'],
          ['analytics', BarChart3, 'Shift Analytics'],
          ['prevention', GraduationCap, 'Prevention & Learning'],
          ['coverage', ClipboardCheck, 'Detection Coverage'],
          ['assistant', Bot, 'AI Assistant'],
          ['upload', Upload, 'Ingest Video'],
        ].map(([key, Icon, label]) => (
          <button key={key} className={`tab-btn ${activeTab === key ? 'active' : ''}`}
            onClick={() => setActiveTab(key)}>
            <Icon size={15} /> {label}
          </button>
        ))}
      </nav>

      <section className="kpi-row">
        <KpiCard label="Critical risk events" value={analytics?.risk_breakdown?.CRITICAL ?? '—'}
          sub="Immediate supervisor attention" tone="critical" />
        <KpiCard label="High risk events" value={analytics?.risk_breakdown?.HIGH ?? '—'}
          sub="Intervene this shift" tone="high" />
        <KpiCard label="Medium risk events" value={analytics?.risk_breakdown?.MEDIUM ?? '—'}
          sub="Coach at next briefing" tone="med" />
        <KpiCard label="Intervention opportunities" value={analytics?.intervention_opportunities ?? '—'}
          sub="High + critical, before damage occurs" tone="accent" />
        <KpiCard label="Elevated events / minute"
          value={analytics ? analytics.high_risk_events_per_minute.toFixed(2) : '—'}
          sub={analytics ? `Baseline over ${analytics.total_footage_minutes.toFixed(1)} min analysed` : 'Baseline rate'}
          tone="low" />
      </section>

      {activeTab === 'operations' && (
        <main className="dashboard-content">
          <div className="video-section">
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <Crosshair size={17} />
                  <span>{selectedVideo?.filename || 'No video selected'}</span>
                </div>
                <div className="header-controls">
                  {selectedVideo?.annotated_video_url && (
                    <div className="toggle-group">
                      <button className={showAnnotated ? 'active' : ''} onClick={() => setShowAnnotated(true)}>AI overlay</button>
                      <button className={!showAnnotated ? 'active' : ''} onClick={() => setShowAnnotated(false)}>Original</button>
                    </div>
                  )}
                  {selectedIncident && <RiskTag level={selectedIncident.risk_level} />}
                </div>
              </div>

              <div className="video-container">
                {videosLoading ? (
                  <div className="video-placeholder"><Loader2 className="spin" size={26} /><span>Loading footage…</span></div>
                ) : selectedVideo ? (
                  <video ref={videoRef} className="main-video" controls playsInline preload="metadata"
                    key={activeSrc} src={activeSrc} />
                ) : (
                  <EmptyState icon={FileVideo} title="No footage analysed yet"
                    hint="Use the Ingest Video tab to upload warehouse loading/unloading footage." />
                )}
              </div>

              {selectedVideo && (
                <>
                  <IncidentScrubber
                    incidents={incidents}
                    duration={selectedVideo.duration_sec}
                    selectedId={selectedIncident?.id}
                    onSelect={handleIncidentClick}
                  />
                  <div className="video-meta">
                    <span><MapPin size={12} /> {selectedVideo.bay || 'Unassigned bay'}</span>
                    <span>{selectedVideo.shift || 'Unassigned shift'}</span>
                    <span>{selectedVideo.camera_id || 'CAM'}</span>
                    <span>{Math.round(selectedVideo.width)}×{Math.round(selectedVideo.height)} @ {Math.round(selectedVideo.fps)}fps</span>
                    <span>{Number(selectedVideo.duration_sec || 0).toFixed(1)}s</span>
                    {selectedVideo.frames_analysed != null && <span>{selectedVideo.frames_analysed} frames analysed</span>}
                    {selectedVideo.detector_backend && <span>backend: {selectedVideo.detector_backend}</span>}
                  </div>
                </>
              )}

              <div className="video-selector-bar">
                {videos.map((v) => (
                  <button key={v.id}
                    className={`video-chip ${selectedVideo?.id === v.id ? 'active' : ''}`}
                    onClick={() => selectVideo(v)}>
                    <Video size={13} />
                    <span className="chip-name">{v.filename}</span>
                    <span className="chip-count">{v.incident_count || 0}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <ShieldAlert size={17} />
                  <span>Behaviour timeline — {filteredIncidents.length} of {incidents.length} event(s)</span>
                </div>
              </div>

              <div className="filter-bar">
                <div className="search-box">
                  <Search size={14} />
                  <input placeholder="Search behaviour, finding or bay…" value={search}
                    onChange={(e) => setSearch(e.target.value)} />
                </div>
                <div className="chip-filters">
                  {['ALL', ...RISK_ORDER].map((lvl) => (
                    <button key={lvl} className={`filter-chip ${riskFilter === lvl ? 'active' : ''}`}
                      onClick={() => setRiskFilter(lvl)}>{lvl}</button>
                  ))}
                </div>
                <select className="select" value={behaviourFilter} onChange={(e) => setBehaviourFilter(e.target.value)}>
                  <option value="ALL">All behaviours</option>
                  {behaviourOptions.map((b) => <option key={b} value={b}>{titleCase(b)}</option>)}
                </select>
              </div>

              <div className="timeline-list">
                {incidentsLoading ? (
                  <div className="empty-state"><Loader2 className="spin" size={22} /><span>Loading events…</span></div>
                ) : incidents.length === 0 ? (
                  <EmptyState icon={CheckCircle2} title="No risky handling behaviour detected in this video"
                    hint="That is a real result, not a placeholder — the pipeline analysed the footage and found nothing meeting the evidence thresholds." />
                ) : filteredIncidents.length === 0 ? (
                  <EmptyState icon={Search} title="No events match these filters"
                    hint="Clear the search box or reset the risk filter." />
                ) : filteredIncidents.map((inc) => (
                  <button key={inc.id}
                    className={`incident-card ${selectedIncident?.id === inc.id ? 'active' : ''}`}
                    onClick={() => handleIncidentClick(inc)}>
                    <div className="incident-left">
                      <RiskTag level={inc.risk_level} />
                      <div className="incident-info">
                        <span className="incident-name">{titleCase(inc.behaviour_type)}</span>
                        <span className="incident-time">
                          {fmtSec(inc.timestamp_sec)} · object #{inc.object_track_id ?? '—'}
                          {inc.duration_sec > 0 && ` · ${inc.duration_sec.toFixed(1)}s`}
                        </span>
                      </div>
                    </div>
                    <div className="incident-right">
                      <span className="score">{Math.round(inc.risk_score)}</span>
                      <span className="score-label">risk score</span>
                      {inc.review_status && inc.review_status !== 'PENDING_REVIEW' && (
                        <span className="reviewed-tag">reviewed</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <aside className="inspector-panel">
            <div className="card">
              <div className="card-header">
                <div className="card-title"><Eye size={17} /><span>Evidence inspector</span></div>
              </div>
              <div className="inspector-body">
                {!selectedIncident ? (
                  <EmptyState icon={Eye} title="Select an event"
                    hint="Pick an event from the timeline to inspect its evidence and reasoning." />
                ) : (
                  <IncidentDetail incident={selectedIncident} onReview={reviewIncident} onSeek={seekTo} />
                )}
              </div>
            </div>
          </aside>
        </main>
      )}

      {activeTab === 'analytics' && (
        <main className="dashboard-content single">
          <AnalyticsView analytics={analytics} />
        </main>
      )}

      {activeTab === 'prevention' && (
        <main className="dashboard-content single">
          <PreventionView prevention={prevention} />
        </main>
      )}

      {activeTab === 'coverage' && (
        <main className="dashboard-content single">
          <CoverageView capabilities={capabilities} />
        </main>
      )}

      {activeTab === 'assistant' && (
        <main className="dashboard-content single">
          <div className="card assistant-panel">
            <div className="card-header">
              <div className="card-title"><Bot size={18} /><span>AI operations assistant</span></div>
              <span className="brand-tag">ANSWERS RETRIEVED FROM THE INCIDENT DATABASE</span>
            </div>
            <div className="chat-history">
              {messages.map((m, idx) => (
                <div key={idx} className={`chat-bubble ${m.role} ${m.error ? 'error' : ''}`}>
                  <Markdown text={m.text} />
                  {m.meta && (
                    <div className="chat-meta">
                      grounded in {m.meta.count} retrieved event(s) · intent: {m.meta.intent} · {m.meta.total} events in database
                    </div>
                  )}
                </div>
              ))}
              {isChatLoading && (
                <div className="chat-bubble assistant loading">
                  <Loader2 size={15} className="spin" /> <span>Querying the incident database…</span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            <div className="quick-prompts">
              {[
                'Show me all high-risk handling events',
                'What were the three most common risky behaviours?',
                'Which loading bay had the highest number of risky events?',
                'Why was this event classified as high risk?',
                'What corrective action is recommended?',
                'How many product drops were detected?',
              ].map((qp) => (
                <button key={qp} className="prompt-pill" onClick={() => sendAssistantMessage(qp)} disabled={isChatLoading}>
                  {qp}
                </button>
              ))}
            </div>
            <div className="chat-input-bar">
              <input className="chat-input" placeholder="Ask about incidents, bays, shifts, root causes or corrective actions…"
                value={inputQuery} onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendAssistantMessage()} />
              <button className="chat-send-btn" onClick={() => sendAssistantMessage()} disabled={isChatLoading}>
                <Send size={15} />
              </button>
            </div>
          </div>
        </main>
      )}

      {activeTab === 'live' && (
        <main className="dashboard-content">
          <div className="video-section">
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <Radio size={17} />
                  <span>Live monitor{live && live.camera_id ? " \u2014 " + live.camera_id : ""}</span>
                </div>
                {live && (
                  <span className={`risk-tag ${liveRunning ? "LOW" : "MEDIUM"}`}>
                    {liveRunning ? "ANALYSING" : String(live.status || "").toUpperCase()}
                  </span>
                )}
              </div>

              <div className="video-container">
                {liveRunning ? (
                  <img className="main-video" alt="Live annotated feed"
                       src={`/api/live/${live.session_id}/stream`} />
                ) : (
                  <div className="video-placeholder">
                    <Radio size={30} style={{ margin: "0 auto 10px" }} />
                    <div>No live session running.</div>
                    <div style={{ fontSize: "0.78rem", marginTop: 6 }}>
                      Choose a source below and start analysis.
                    </div>
                  </div>
                )}
              </div>

              {live && (
                <div className="video-meta">
                  <span><span className="key">Analysed</span> <strong>{live.frames_analysed}</strong> frames</span>
                  <span><span className="key">Rate</span> <strong>{live.analysed_fps}</strong> fps</span>
                  <span><span className="key">Skipped to stay live</span> <strong>{live.frames_dropped}</strong></span>
                  <span><span className="key">Tracks</span> <strong>{live.active_tracks}</strong></span>
                  <span><span className="key">Alerts</span> <strong>{live.event_count}</strong></span>
                  <span><span className="key">Uptime</span> <strong>{live.uptime_sec}s</strong></span>
                </div>
              )}
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title"><Crosshair size={17} /><span>Source and scene context</span></div>
              </div>
              <div className="pad">
                <div className="form-grid">
                  <label>Source type
                    <select value={liveForm.source_kind} disabled={liveRunning}
                      onChange={e => setLiveForm({ ...liveForm, source_kind: e.target.value, source: "" })}>
                      <option value="file">Stored video (replayed live)</option>
                      <option value="camera">Attached camera</option>
                      <option value="stream">CCTV stream (RTSP/HTTP)</option>
                    </select>
                  </label>

                  {liveForm.source_kind === "file" && (
                    <label>Video
                      <select value={liveForm.source} disabled={liveRunning}
                        onChange={e => setLiveForm({ ...liveForm, source: e.target.value })}>
                        {(liveSources && liveSources.library ? liveSources.library : []).map(f => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                    </label>
                  )}
                  {liveForm.source_kind === "camera" && (
                    <label>Camera index
                      <input type="number" min="0" max="8" value={liveForm.source || "0"} disabled={liveRunning}
                        onChange={e => setLiveForm({ ...liveForm, source: e.target.value })} />
                    </label>
                  )}
                  {liveForm.source_kind === "stream" && (
                    <label>Stream URL
                      <input type="text" placeholder="rtsp://camera/stream" value={liveForm.source}
                        disabled={liveRunning}
                        onChange={e => setLiveForm({ ...liveForm, source: e.target.value })} />
                    </label>
                  )}

                  <label>Camera ID
                    <input value={liveForm.camera_id} disabled={liveRunning}
                      onChange={e => setLiveForm({ ...liveForm, camera_id: e.target.value })} />
                  </label>
                  <label>Loading bay
                    <input value={liveForm.bay} disabled={liveRunning}
                      onChange={e => setLiveForm({ ...liveForm, bay: e.target.value })} />
                  </label>
                  <label>Shift
                    <input value={liveForm.shift} disabled={liveRunning}
                      onChange={e => setLiveForm({ ...liveForm, shift: e.target.value })} />
                  </label>
                  <label>Floor condition
                    <select value={liveForm.floor_condition} disabled={liveRunning}
                      onChange={e => setLiveForm({ ...liveForm, floor_condition: e.target.value })}>
                      <option value="unknown">Unknown</option>
                      <option value="dry">Dry</option>
                      <option value="wet">Wet</option>
                    </select>
                  </label>
                  <label className="checkbox-label">
                    <input type="checkbox" checked={liveForm.dock_transfer} disabled={liveRunning}
                      onChange={e => setLiveForm({ ...liveForm, dock_transfer: e.target.checked })} />
                    Covers a dock transfer point
                  </label>
                </div>

                <div style={{ display: "flex", gap: 10, marginTop: 18, alignItems: "center" }}>
                  {!liveRunning ? (
                    <button className={`primary-btn ${liveStarting ? "disabled" : ""}`}
                      onClick={startLive} disabled={liveStarting}>
                      {liveStarting
                        ? (<><Loader2 size={16} className="spin" /> Starting analysis</>)
                        : (<>Start live analysis</>)}
                    </button>
                  ) : (
                    <button className="primary-btn" onClick={stopLive}>
                      <Square size={14} /> Stop analysis
                    </button>
                  )}
                </div>

                {liveError && <div className="status-box error" style={{ marginTop: 14 }}>{liveError}</div>}

                <p className="section-note" style={{ marginTop: 16, marginBottom: 0 }}>
                  Live analysis runs the same detection, tracking, behaviour and risk code as recorded
                  analysis, and its alerts are written to the same incident database. On CPU the
                  pipeline analyses only a few frames per second, so frames are deliberately skipped
                  to keep alerts anchored to the present rather than falling further behind. The
                  skipped count shown above is the real figure, not an estimate.
                </p>
              </div>
            </div>
          </div>

          <aside className="inspector-panel">
            <div className="card">
              <div className="card-header">
                <div className="card-title"><ShieldAlert size={17} /><span>Live alerts</span></div>
              </div>
              {live && live.events && live.events.length > 0 ? (
                <div className="timeline-list">
                  {live.events.map(ev => (
                    <div key={ev.id} className="incident-card">
                      <div className="incident-left">
                        <span className={`risk-tag ${ev.risk_level}`}>{ev.risk_level}</span>
                        <div className="incident-info">
                          <span className="incident-name">
                            {String(ev.behaviour_type).replace(/_/g, " ").toUpperCase()}
                          </span>
                          <span className="incident-time">
                            {Number(ev.timestamp_sec).toFixed(1)}s into session
                            {ev.bay ? " \u00b7 " + ev.bay : ""}
                          </span>
                        </div>
                      </div>
                      <div className="incident-right">
                        <span className="score">{ev.risk_score}</span>
                        <div className="score-label">score</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <ShieldAlert size={26} />
                  <div className="empty-title">No alerts yet</div>
                  <div className="empty-hint">
                    {liveRunning
                      ? "Analysis is running. Risky handling appears here the moment it is detected."
                      : "Alerts raised during a live session appear here, and are saved to the incident database alongside recorded findings."}
                  </div>
                </div>
              )}
            </div>
          </aside>
        </main>
      )}

      {activeTab === 'upload' && (
        <main className="dashboard-content single">
          <div className="card ingest-card">
            <div className="card-header">
              <div className="card-title"><Upload size={17} /><span>Ingest warehouse footage</span></div>
            </div>
            <div className="ingest-body">
              <p className="ingest-lead">
                Upload recorded CCTV or smartphone footage of a loading/unloading operation.
                The scene context below is what the wet-floor, dock and staging-zone detectors
                reason against — it is site knowledge the camera cannot infer.
              </p>

              <div className="form-grid">
                <label>Loading bay
                  <input value={sceneForm.bay} onChange={(e) => setSceneForm({ ...sceneForm, bay: e.target.value })} />
                </label>
                <label>Shift
                  <select value={sceneForm.shift} onChange={(e) => setSceneForm({ ...sceneForm, shift: e.target.value })}>
                    <option>Shift A</option><option>Shift B</option><option>Shift C</option>
                    <option>Unassigned Shift</option>
                  </select>
                </label>
                <label>Camera ID
                  <input value={sceneForm.camera_id} onChange={(e) => setSceneForm({ ...sceneForm, camera_id: e.target.value })} />
                </label>
                <label>Floor condition
                  <select value={sceneForm.floor_condition}
                    onChange={(e) => setSceneForm({ ...sceneForm, floor_condition: e.target.value })}>
                    <option value="unknown">Unknown / not reported</option>
                    <option value="dry">Dry</option>
                    <option value="wet">Wet</option>
                  </select>
                </label>
                <label className="checkbox-label">
                  <input type="checkbox" checked={sceneForm.dock_transfer}
                    onChange={(e) => setSceneForm({ ...sceneForm, dock_transfer: e.target.checked })} />
                  Camera covers a vehicle dock transition
                </label>
              </div>

              <input ref={fileInputRef} type="file" accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm"
                id="video-file-input" style={{ display: 'none' }} onChange={handleFileUpload} disabled={isUploading} />
              <label htmlFor="video-file-input" className={`primary-btn ${isUploading ? 'disabled' : ''}`}>
                {isUploading ? (<><Loader2 size={17} className="spin" /> Analysing…</>) : (<><Upload size={17} /> Select video file</>)}
              </label>

              {isUploading && (
                <div className="progress-wrap">
                  <div className="progress-track"><div className="progress-fill" style={{ width: `${uploadProgress}%` }} /></div>
                  <div className="progress-label">{uploadProgress}%</div>
                </div>
              )}
              {uploadStatus && <div className="status-box">{uploadStatus}</div>}
              {uploadError && (
                <div className="status-box error"><AlertTriangle size={14} /> {uploadError}</div>
              )}
            </div>
          </div>
        </main>
      )}
    </div>
  );
}

/* -------------------------------------------------------------- components */

function KpiCard({ label, value, sub, tone }) {
  return (
    <div className={`kpi-card ${tone}`}>
      <span className="kpi-label">{label}</span>
      <span className="kpi-val">{value}</span>
      <span className="kpi-sub">{sub}</span>
    </div>
  );
}

/** Timeline scrubber showing every incident as a clickable marker. */
function IncidentScrubber({ incidents, duration, selectedId, onSelect }) {
  if (!duration || duration <= 0 || incidents.length === 0) return null;
  return (
    <div className="scrubber">
      <div className="scrubber-track">
        {incidents.map((inc) => (
          <button key={inc.id}
            className={`marker ${inc.risk_level} ${selectedId === inc.id ? 'active' : ''}`}
            style={{ left: `${Math.min(99, (inc.timestamp_sec / duration) * 100)}%` }}
            title={`${titleCase(inc.behaviour_type)} — ${inc.risk_level} at ${fmtSec(inc.timestamp_sec)}`}
            onClick={() => onSelect(inc)} />
        ))}
      </div>
      <div className="scrubber-labels"><span>0s</span><span>{duration.toFixed(1)}s</span></div>
    </div>
  );
}

function IncidentDetail({ incident, onReview, onSeek }) {
  const tierLabel = {
    OBSERVED_BEHAVIOUR: 'Observed behaviour',
    POTENTIAL_RISK: 'Potential damage risk',
    CONFIRMED_DAMAGE: 'Damage confirmed by reviewer',
  }[incident.evidence_tier] || 'Observed behaviour';

  return (
    <>
      <div className="detail-head">
        <RiskTag level={incident.risk_level} />
        <span className="detail-behaviour">{titleCase(incident.behaviour_type)}</span>
        <button className="link-btn" onClick={() => onSeek(incident.timestamp_sec)}>
          Replay at {fmtSec(incident.timestamp_sec)}
        </button>
      </div>

      <div className="tier-banner">
        <AlertTriangle size={13} />
        <span><strong>{tierLabel}</strong> — the system reports handling risk, not confirmed damage.
          Physical inspection is required before any damage conclusion.</span>
      </div>

      {incident.evidence_image_url && (
        <div className="evidence-block">
          <span className="block-label">Annotated evidence frame</span>
          <img src={incident.evidence_image_url} alt="Incident evidence" className="evidence-img" />
        </div>
      )}

      {incident.evidence_clip_url && (
        <div className="evidence-block">
          <span className="block-label">Incident replay clip</span>
          <video className="evidence-clip" controls preload="metadata" src={incident.evidence_clip_url} />
        </div>
      )}

      <div className="info-block">
        <span className="info-block-title">What was observed</span>
        <span className="info-block-text">{incident.evidence_description}</span>
      </div>

      {incident.evidence_stages?.length > 0 && (
        <div className="info-block">
          <span className="info-block-title">Temporal sequence (why this is not a single-frame guess)</span>
          <div className="stage-chain">
            {incident.evidence_stages.map((s, i) => (
              <React.Fragment key={i}>
                <span className="stage">{s.stage}<em>{s.at_sec}s</em></span>
                {i < incident.evidence_stages.length - 1 && <span className="stage-arrow">→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>
      )}

      {incident.risk_factors?.length > 0 && (
        <div className="info-block">
          <span className="info-block-title">
            Risk score breakdown — {Math.round(incident.risk_score)}/100
          </span>
          <div className="factor-list">
            {incident.risk_factors.map((f, i) => (
              <div key={i} className="factor">
                <span className={`factor-pts ${f.points >= 0 ? 'pos' : 'neg'}`}>
                  {f.points >= 0 ? '+' : ''}{Math.round(f.points)}
                </span>
                <div className="factor-body">
                  <span className="factor-name">{f.name}</span>
                  <span className="factor-detail">{f.detail}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="info-block">
        <span className="info-block-title">Operational root cause</span>
        <span className="info-block-text">{incident.root_cause}</span>
      </div>

      <div className="info-block rec-box">
        <span className="info-block-title">Recommended supervisor action</span>
        <span className="info-block-text strong">{incident.recommended_action}</span>
      </div>

      <div className="info-block">
        <span className="info-block-title">Human review</span>
        <span className="info-block-text muted">
          Current status: <strong>{titleCase(incident.review_status || 'PENDING_REVIEW')}</strong>.
          Only a reviewer can mark damage as confirmed — the AI never does.
        </span>
        <div className="review-actions">
          <button onClick={() => onReview('CONFIRMED_BY_SUPERVISOR')}>Confirm behaviour</button>
          <button onClick={() => onReview('FALSE_POSITIVE')}>Mark false positive</button>
          <button onClick={() => onReview('DAMAGE_CONFIRMED')}>Damage confirmed on inspection</button>
          <button onClick={() => onReview('NO_ACTION_NEEDED')}>No action needed</button>
        </div>
      </div>

      <div className="meta-row">
        <span>Bay: {incident.bay || '—'}</span>
        <span>Shift: {incident.shift || '—'}</span>
        <span>Camera: {incident.camera_id || '—'}</span>
        <span>Detection confidence: {(incident.confidence * 100).toFixed(0)}%</span>
      </div>
    </>
  );
}

function AnalyticsView({ analytics }) {
  if (!analytics) return <div className="card"><EmptyState icon={Loader2} title="Loading analytics…" /></div>;
  if (analytics.total_incidents === 0) {
    return (
      <div className="card">
        <EmptyState icon={BarChart3} title="No analytics yet"
          hint="Analytics are computed from recorded events only. Ingest footage to populate this view." />
      </div>
    );
  }
  const behaviours = Object.entries(analytics.top_behaviours || {});
  const maxB = Math.max(...behaviours.map(([, c]) => c), 1);
  const maxBay = Math.max(...(analytics.by_bay || []).map((b) => b.total), 1);

  return (
    <>
      <div className="card pad">
        <h2 className="section-title"><BarChart3 size={18} /> Behaviour Pareto</h2>
        <p className="section-note">
          Counts are of detected events across {analytics.total_footage_minutes.toFixed(1)} minutes
          of analysed footage. Focusing on the top bars removes most of the risk.
        </p>
        <div className="chart-list">
          {behaviours.map(([b, c]) => (
            <div key={b} className="chart-row">
              <div className="chart-label">
                <span>{titleCase(b)}</span>
                <span className="chart-value">{c} ({Math.round((c / analytics.total_incidents) * 100)}%)</span>
              </div>
              <Bar value={c} max={maxB} />
            </div>
          ))}
        </div>
      </div>

      <div className="grid-2">
        <div className="card pad">
          <h2 className="section-title"><MapPin size={17} /> Risk by loading bay</h2>
          {(analytics.by_bay || []).length === 0 ? (
            <EmptyState icon={MapPin} title="No bay data" />
          ) : (
            <div className="chart-list">
              {analytics.by_bay.map((b) => (
                <div key={b.bay} className="chart-row">
                  <div className="chart-label">
                    <span>{b.bay}</span>
                    <span className="chart-value">{b.total} events · {b.high_risk} elevated</span>
                  </div>
                  <Bar value={b.total} max={maxBay} tone="amber" />
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card pad">
          <h2 className="section-title"><Layers size={17} /> Risk mix &amp; review status</h2>
          <div className="stat-grid">
            {RISK_ORDER.map((lvl) => (
              <div key={lvl} className="stat-tile">
                <span className={`stat-val ${lvl}`}>{analytics.risk_breakdown[lvl]}</span>
                <span className="stat-lbl">{lvl}</span>
              </div>
            ))}
          </div>
          <div className="review-summary">
            {Object.entries(analytics.review_breakdown || {}).map(([k, v]) => (
              <div key={k} className="review-line"><span>{titleCase(k)}</span><strong>{v}</strong></div>
            ))}
          </div>
        </div>
      </div>

      <div className="card pad">
        <h2 className="section-title"><Video size={17} /> Per-video results</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead><tr><th>Video</th><th>Duration</th><th>Events</th><th>Elevated risk</th><th>Events / min</th></tr></thead>
            <tbody>
              {(analytics.by_video || []).map((v) => (
                <tr key={v.video_id}>
                  <td className="wrap">{v.filename}</td>
                  <td>{v.duration_sec.toFixed(1)}s</td>
                  <td>{v.total}</td>
                  <td>{v.high_risk}</td>
                  <td>{v.duration_sec > 0 ? (v.total / (v.duration_sec / 60)).toFixed(1) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="card pad">
          <h2 className="section-title">Shift comparison</h2>
          <div className="chart-list">
            {(analytics.by_shift || []).map((s) => (
              <div key={s.shift} className="chart-row">
                <div className="chart-label"><span>{s.shift}</span>
                  <span className="chart-value">{s.total} events · {s.high_risk} elevated</span></div>
                <Bar value={s.total} max={Math.max(...analytics.by_shift.map((x) => x.total), 1)} tone="violet" />
              </div>
            ))}
          </div>
        </div>
        <div className="card pad">
          <h2 className="section-title">Damage prevention framing</h2>
          <p className="prevention-statement">
            <strong>{analytics.intervention_opportunities}</strong> high-risk handling events were
            identified across <strong>{analytics.total_footage_minutes.toFixed(1)} minutes</strong> of
            footage — each one an opportunity to intervene <em>before</em> damage occurs.
          </p>
          <p className="section-note">
            This is deliberately not phrased as “N products damaged”. The system observes handling
            behaviour and infers risk; whether damage actually resulted can only be established by
            physical inspection, recorded through the human-review workflow.
          </p>
        </div>
      </div>
    </>
  );
}

function PreventionView({ prevention }) {
  if (!prevention) return <div className="card"><EmptyState icon={Loader2} title="Loading prevention insights…" /></div>;
  const { recurring_behaviours: recurring = [], high_risk_locations: hotspots = [], baseline } = prevention;
  return (
    <>
      <div className="card pad">
        <h2 className="section-title"><RefreshCw size={17} /> Recurring behaviours &amp; training opportunities</h2>
        {recurring.length === 0 ? (
          <EmptyState icon={CheckCircle2} title="No behaviour has recurred yet"
            hint="A behaviour appears here once it has been detected at least twice." />
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Behaviour</th><th>Occurrences</th><th>Share</th><th>Suggested coaching topic</th></tr></thead>
              <tbody>
                {recurring.map((r) => (
                  <tr key={r.behaviour_type}>
                    <td><strong>{titleCase(r.behaviour_type)}</strong></td>
                    <td>{r.occurrences}</td>
                    <td>{r.share_percent}%</td>
                    <td className="wrap">{r.training_topic}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="grid-2">
        <div className="card pad">
          <h2 className="section-title"><MapPin size={17} /> High-risk locations</h2>
          {hotspots.length === 0 ? <EmptyState icon={MapPin} title="No elevated-risk location recorded" /> : (
            <div className="chart-list">
              {hotspots.map((h) => (
                <div key={h.bay} className="hotspot">
                  <span className="hotspot-name">{h.bay}</span>
                  <span className="hotspot-val">{h.high_risk} elevated of {h.total} events</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="card pad">
          <h2 className="section-title"><Sparkles size={17} /> Improvement tracking</h2>
          <div className="baseline-box">
            <span className="baseline-val">{baseline.high_risk_events_per_minute.toFixed(2)}</span>
            <span className="baseline-lbl">elevated-risk events per minute of footage</span>
          </div>
          <p className="section-note">{baseline.note}</p>
          <p className="section-note">
            Measured over {baseline.total_footage_minutes.toFixed(1)} minutes. Re-analyse footage from a
            later shift with the same bays to see whether coaching moved this number.
          </p>
        </div>
      </div>
    </>
  );
}

function CoverageView({ capabilities }) {
  if (!capabilities) return <div className="card"><EmptyState icon={Loader2} title="Loading capability report…" /></div>;
  const { behaviours = [], counts = {} } = capabilities;
  return (
    <div className="card pad">
      <h2 className="section-title"><ClipboardCheck size={18} /> Detection coverage — honest status</h2>
      <p className="section-note">
        This table is generated from the detector implementations themselves, so it cannot claim a
        capability the code does not have. “Events recorded” is the real count from the database —
        a detector can be fully implemented and still show zero if the footage contains no such behaviour.
      </p>
      <div className="coverage-counts">
        <span><strong>{counts.implemented}</strong> implemented</span>
        <span><strong>{counts.partial}</strong> partial</span>
        <span><strong>{counts.requires_config}</strong> need configuration or footage</span>
        <span><strong>{counts.total}</strong> behaviours defined</span>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr><th>Behaviour</th><th>Detection method</th><th>Status</th><th>Events recorded</th><th>Known limitations</th></tr>
          </thead>
          <tbody>
            {behaviours.map((b) => (
              <tr key={b.behaviour_type}>
                <td><strong>{b.label}</strong></td>
                <td className="wrap dim">{b.method}</td>
                <td><StatusPill status={b.status} /></td>
                <td className={b.events_recorded > 0 ? 'ok-num' : 'dim'}>{b.events_recorded}</td>
                <td className="wrap dim small">{b.limitations}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
