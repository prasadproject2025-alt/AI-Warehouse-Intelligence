import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldAlert, 
  Video, 
  Bot, 
  BarChart3, 
  AlertTriangle, 
  CheckCircle2, 
  Play, 
  Pause, 
  RotateCcw, 
  Upload, 
  Send,
  Eye,
  Crosshair,
  Flame,
  ArrowRight,
  TrendingDown,
  Layers,
  Sparkles
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('operations'); // 'operations', 'analytics', 'assistant', 'upload'
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [riskFilter, setRiskFilter] = useState('ALL');
  
  // Assistant chat state
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "👋 Welcome to **VisionGuard AI Field Intelligence Assistant**. I am actively monitoring warehouse material-handling operations. Ask me about detected high-risk behaviors, drop events, shift summaries, or root-cause recommendations!"
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');

  const videoRef = useRef(null);

  // Load initial videos and analytics
  useEffect(() => {
    fetchVideos();
    fetchAnalytics();
    const interval = setInterval(() => {
      fetchAnalytics();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchVideos = async () => {
    try {
      const res = await fetch('/api/videos');
      const data = await res.json();
      if (data.videos && data.videos.length > 0) {
        setVideos(data.videos);
        if (!selectedVideo) {
          selectVideo(data.videos[0]);
        }
      }
    } catch (err) {
      console.error("Failed to load videos", err);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await fetch('/api/analytics');
      const data = await res.json();
      setAnalytics(data);
    } catch (err) {
      console.error("Failed to load analytics", err);
    }
  };

  const selectVideo = async (video) => {
    setSelectedVideo(video);
    try {
      const res = await fetch(`/api/videos/${video.id}`);
      const data = await res.json();
      setIncidents(data.incidents || []);
      if (data.incidents && data.incidents.length > 0) {
        setSelectedIncident(data.incidents[0]);
      } else {
        setSelectedIncident(null);
      }
    } catch (err) {
      console.error("Failed to load video details", err);
    }
  };

  const seekToTimestamp = (sec) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, sec - 0.2);
      videoRef.current.play();
    }
  };

  const handleIncidentClick = (inc) => {
    setSelectedIncident(inc);
    seekToTimestamp(inc.timestamp_sec);
  };

  const sendAssistantMessage = async (queryText) => {
    const q = queryText || inputQuery;
    if (!q.trim()) return;

    const newMsgs = [...messages, { role: 'user', text: q }];
    setMessages(newMsgs);
    if (!queryText) setInputQuery('');
    setIsChatLoading(true);

    try {
      const res = await fetch('/api/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, video_id: selectedVideo?.id })
      });
      const data = await res.json();
      setMessages([...newMsgs, { role: 'assistant', text: data.response }]);
    } catch (err) {
      setMessages([...newMsgs, { role: 'assistant', text: "⚠️ Error contacting AI Assistant service." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(`Uploading and analyzing ${file.name} with YOLOv8 & Behaviour Engine...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/videos/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setUploadStatus(`Success! Identified ${data.result.incidents_count} incidents.`);
      fetchVideos();
      fetchAnalytics();
    } catch (err) {
      setUploadStatus(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const filteredIncidents = incidents.filter(i => {
    if (riskFilter === 'ALL') return true;
    return i.risk_level === riskFilter;
  });

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="nav-brand">
          <div className="brand-badge">GEG HACKATHON</div>
          <div className="brand-title">
            <span>VisionGuard</span>
            <span className="brand-tag">FIELD INTELLIGENCE ASSISTANT</span>
          </div>
        </div>

        <div className="nav-actions">
          <div className="metric-pill">
            <span className="dot"></span>
            <span>AI Perception Pipeline: <strong>ONLINE (YOLOv8 + ByteTrack)</strong></span>
          </div>
          {analytics && (
            <div className="metric-pill">
              <span>Handling Discipline: <strong>{analytics.handling_discipline_score}%</strong></span>
            </div>
          )}
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="tab-row">
        <button 
          className={`tab-btn ${activeTab === 'operations' ? 'active' : ''}`}
          onClick={() => setActiveTab('operations')}
        >
          <Video size={16} /> Operations & Live Replay
        </button>
        <button 
          className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <BarChart3 size={16} /> Shift Behaviour Analytics
        </button>
        <button 
          className={`tab-btn ${activeTab === 'assistant' ? 'active' : ''}`}
          onClick={() => setActiveTab('assistant')}
        >
          <Bot size={16} /> AI Warehouse Assistant
        </button>
        <button 
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          <Upload size={16} /> Ingest Warehouse Video
        </button>
      </div>

      {/* KPI Stats Strip */}
      <div className="dashboard-content" style={{ paddingBottom: '0' }}>
        <div className="kpi-row">
          <div className="kpi-card critical">
            <span className="kpi-label">Critical Incidents</span>
            <span className="kpi-val" style={{ color: 'var(--risk-critical)' }}>
              {analytics?.risk_breakdown?.CRITICAL || 0}
            </span>
            <span className="kpi-sub">Severe drop or crushing risk</span>
          </div>
          <div className="kpi-card high">
            <span className="kpi-label">High-Risk Events</span>
            <span className="kpi-val" style={{ color: 'var(--risk-high)' }}>
              {analytics?.risk_breakdown?.HIGH || 0}
            </span>
            <span className="kpi-sub">Immediate intervention needed</span>
          </div>
          <div className="kpi-card med">
            <span className="kpi-label">Medium Risk</span>
            <span className="kpi-val" style={{ color: 'var(--risk-med)' }}>
              {analytics?.risk_breakdown?.MEDIUM || 0}
            </span>
            <span className="kpi-sub">Floor dragging / posture</span>
          </div>
          <div className="kpi-card low">
            <span className="kpi-label">Discipline Index</span>
            <span className="kpi-val" style={{ color: 'var(--risk-low)' }}>
              {analytics?.handling_discipline_score || 100}%
            </span>
            <span className="kpi-sub">Shift compliance rating</span>
          </div>
          <div className="kpi-card score">
            <span className="kpi-label">Damage Prevention</span>
            <span className="kpi-val" style={{ color: 'var(--accent-cyan)' }}>
              {(analytics?.risk_breakdown?.CRITICAL || 0) + (analytics?.risk_breakdown?.HIGH || 0)}
            </span>
            <span className="kpi-sub">Proactive Interventions</span>
          </div>
        </div>
      </div>

      {/* Main Tab Views */}
      {activeTab === 'operations' && (
        <main className="dashboard-content">
          {/* Left Column: Video Player & Selector */}
          <div className="video-section">
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <Crosshair size={18} color="var(--accent-cyan)" />
                  <span>AI Video Perception: {selectedVideo?.filename || "Loading video..."}</span>
                </div>
                {selectedIncident && (
                  <span className={`risk-tag ${selectedIncident.risk_level}`}>
                    {selectedIncident.risk_level} RISK
                  </span>
                )}
              </div>

              {/* Video Player Box */}
              <div className="video-container">
                {selectedVideo ? (
                  <video 
                    ref={videoRef}
                    className="main-video"
                    controls
                    playsInline
                    src={`/static/processed/${selectedVideo.annotated_filepath?.split(/[/\\]/).pop() || selectedVideo.filename}`}
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)' }}>Select or ingest a warehouse video to begin.</div>
                )}
                
                <div className="video-hud-overlay">
                  <span className="rec-badge">AI LIVE INFERENCE</span>
                  <span>720P HD @ 30 FPS</span>
                  <span>BYTE-TRACK ID ACTIVE</span>
                </div>
              </div>

              {/* Video Switcher Chips */}
              <div className="video-selector-bar">
                {videos.map(v => (
                  <div 
                    key={v.id} 
                    className={`video-chip ${selectedVideo?.id === v.id ? 'active' : ''}`}
                    onClick={() => selectVideo(v)}
                  >
                    <Video size={14} />
                    <span>{v.filename}</span>
                    <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>({v.incident_count || 0} alerts)</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Timeline Filter and List */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <ShieldAlert size={18} color="var(--risk-critical)" />
                  <span>Detected Behaviour Timeline ({filteredIncidents.length} events)</span>
                </div>

                <div style={{ display: 'flex', gap: '6px' }}>
                  {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(lvl => (
                    <button
                      key={lvl}
                      onClick={() => setRiskFilter(lvl)}
                      style={{
                        padding: '4px 10px',
                        fontSize: '0.72rem',
                        borderRadius: '4px',
                        border: '1px solid var(--border-color)',
                        background: riskFilter === lvl ? 'var(--accent-blue)' : 'var(--bg-surface)',
                        color: '#fff',
                        cursor: 'pointer'
                      }}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>
              </div>

              <div className="timeline-list">
                {filteredIncidents.length === 0 ? (
                  <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No events detected matching filter '{riskFilter}'.
                  </div>
                ) : (
                  filteredIncidents.map(inc => (
                    <div 
                      key={inc.id}
                      className={`incident-card ${selectedIncident?.id === inc.id ? 'active' : ''}`}
                      onClick={() => handleIncidentClick(inc)}
                    >
                      <div className="incident-left">
                        <span className={`risk-tag ${inc.risk_level}`}>
                          {inc.risk_level}
                        </span>
                        <div className="incident-info">
                          <span className="incident-name">
                            {inc.behaviour_type.replace(/_/g, ' ').toUpperCase()}
                          </span>
                          <span className="incident-time">
                            ⏱ {inc.timestamp_sec.toFixed(2)}s | Object #{inc.object_track_id || 'N/A'}
                          </span>
                        </div>
                      </div>

                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '0.78rem', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                          Score: {inc.risk_score}
                        </span>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                          Click to Replay ❯
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Evidence Inspector & Prescriptive Actions */}
          <aside className="inspector-panel">
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <Eye size={18} color="var(--accent-cyan)" />
                  <span>Incident Evidence Inspector</span>
                </div>
              </div>

              <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {selectedIncident ? (
                  <>
                    {/* Evidence Image Snapshot */}
                    {selectedIncident.evidence_image_path && (
                      <div>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px', display: 'block' }}>
                          ANNOTATED EVIDENCE FRAME (T: {selectedIncident.timestamp_sec}s)
                        </span>
                        <img 
                          src={`/static/evidence/${selectedIncident.evidence_image_path.split(/[/\\]/).pop()}`} 
                          alt="Incident Evidence"
                          className="evidence-preview-img"
                        />
                      </div>
                    )}

                    {/* Operational Details */}
                    <div className="info-block">
                      <span className="info-block-title">Observed Behavior & Physical Metrics</span>
                      <span className="info-block-text">{selectedIncident.evidence_description}</span>
                    </div>

                    <div className="info-block">
                      <span className="info-block-title">Operational Root Cause</span>
                      <span className="info-block-text">{selectedIncident.root_cause}</span>
                    </div>

                    {/* Prescriptive Corrective Intervention */}
                    <div className="info-block rec-box">
                      <span className="info-block-title" style={{ color: 'var(--risk-low)' }}>
                        Recommended Supervisor Intervention
                      </span>
                      <span className="info-block-text" style={{ fontWeight: 600 }}>
                        {selectedIncident.recommended_action}
                      </span>
                    </div>
                  </>
                ) : (
                  <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    Click an incident on the timeline to inspect evidence frame and root-cause analysis.
                  </div>
                )}
              </div>
            </div>
          </aside>
        </main>
      )}

      {/* Analytics Tab */}
      {activeTab === 'analytics' && (
        <main className="dashboard-content" style={{ gridTemplateColumns: '1fr' }}>
          <div className="card" style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BarChart3 color="var(--accent-cyan)" />
              Shift Behaviour & Damage Prevention Analytics
            </h2>

            {analytics && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                {/* Pareto Chart */}
                <div className="info-block">
                  <span className="info-block-title">Top Detected Risky Behaviours (Pareto Breakdown)</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '12px' }}>
                    {Object.entries(analytics.top_behaviours || {}).map(([bName, count]) => {
                      const pct = Math.round((count / Math.max(1, analytics.total_incidents)) * 100);
                      return (
                        <div key={bName} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem' }}>
                            <span style={{ fontWeight: 600 }}>{bName.replace(/_/g, ' ').toUpperCase()}</span>
                            <span style={{ color: 'var(--text-muted)' }}>{count} events ({pct}%)</span>
                          </div>
                          <div style={{ width: '100%', height: '8px', background: 'var(--bg-main)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: 'linear-gradient(90deg, #38bdf8, #2563eb)' }}></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* 10 Scenarios Coverage Table */}
                <div className="info-block">
                  <span className="info-block-title">GEG Challenge: 10 Target Behaviour Taxonomy Coverage</span>
                  <table style={{ width: '100%', marginTop: '12px', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '6px' }}>Behaviour Scenario</th>
                        <th style={{ padding: '6px' }}>Detection Type</th>
                        <th style={{ padding: '6px' }}>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ['Product Dropping', 'Downward Acceleration & Impact Deceleration', 'Active'],
                        ['Product Dragging', 'Floor Plane Horizontal Translation', 'Active'],
                        ['Product Throwing / Pushing', 'Release Velocity & Spatial Detachment', 'Active'],
                        ['Rolling Cartons / Mattresses', 'Aspect-Ratio Tumbling Cycles', 'Active'],
                        ['Improper Stacking / Inversion', 'Heavy-on-Light Vertical Inversion', 'Active'],
                        ['Stepping on Cartons', 'Operator Foot Contact on Product Top', 'Active'],
                        ['Using Straps to Pull', 'Tensile Strap Pull Without Base Support', 'Active'],
                        ['Dragging on Wet Floor', 'Floor Moisture Contact Zone', 'Active'],
                        ['Vertical Product Kept Flat', 'Upright Aspect-Ratio Deviation', 'Active'],
                        ['Dock Level Hazard', 'Uneven Bed-to-Dock Transition Shock', 'Active']
                      ].map(([name, method, stat]) => (
                        <tr key={name} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '8px 6px', fontWeight: 600 }}>{name}</td>
                          <td style={{ padding: '8px 6px', color: 'var(--text-dim)' }}>{method}</td>
                          <td style={{ padding: '8px 6px', color: 'var(--risk-low)', fontWeight: 700 }}>✓ {stat}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </main>
      )}

      {/* AI Assistant Tab */}
      {activeTab === 'assistant' && (
        <main className="dashboard-content" style={{ gridTemplateColumns: '1fr' }}>
          <div className="card assistant-panel">
            <div className="card-header">
              <div className="card-title">
                <Bot size={20} color="var(--accent-cyan)" />
                <span>AI Warehouse Operations Assistant (Grounded Factual Reasoning)</span>
              </div>
              <span className="brand-tag">ZERO HALLUCINATION GUARANTEE</span>
            </div>

            {/* Message History */}
            <div className="chat-history">
              {messages.map((m, idx) => (
                <div key={idx} className={`chat-bubble ${m.role}`}>
                  <div style={{ whiteSpace: 'pre-wrap' }}>
                    {m.text}
                  </div>
                </div>
              ))}
              {isChatLoading && (
                <div className="chat-bubble assistant" style={{ color: 'var(--accent-cyan)' }}>
                  Analyzing warehouse event database...
                </div>
              )}
            </div>

            {/* Quick Prompts */}
            <div className="quick-prompts">
              {[
                "Show me all high-risk handling events",
                "What were the three most common risky behaviours?",
                "How many product drops were detected?",
                "Why was this event classified as high risk?",
                "Summarize shift handling discipline"
              ].map(qp => (
                <span 
                  key={qp} 
                  className="prompt-pill"
                  onClick={() => sendAssistantMessage(qp)}
                >
                  {qp}
                </span>
              ))}
            </div>

            {/* Input Bar */}
            <div className="chat-input-bar">
              <input 
                type="text"
                className="chat-input"
                placeholder="Ask the AI supervisor assistant about warehouse incidents, root causes, or training needs..."
                value={inputQuery}
                onChange={e => setInputQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && sendAssistantMessage()}
              />
              <button className="chat-send-btn" onClick={() => sendAssistantMessage()}>
                <Send size={16} />
              </button>
            </div>
          </div>
        </main>
      )}

      {/* Ingest / Upload Tab */}
      {activeTab === 'upload' && (
        <main className="dashboard-content" style={{ gridTemplateColumns: '1fr' }}>
          <div className="card" style={{ padding: '36px', textAlign: 'center' }}>
            <Upload size={48} color="var(--accent-cyan)" style={{ margin: '0 auto 16px auto' }} />
            <h2 style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: '8px' }}>
              Ingest Warehouse Video for AI Analysis
            </h2>
            <p style={{ color: 'var(--text-muted)', maxWidth: '540px', margin: '0 auto 24px auto', fontSize: '0.9rem' }}>
              Upload any recorded CCTV or smartphone footage from loading/unloading bays. VisionGuard will automatically detect objects, track persistent IDs, identify risky handling behaviours, and generate actionable evidence.
            </p>

            <input 
              type="file" 
              accept="video/mp4,video/avi,video/mov" 
              id="video-file-input"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
            />
            <label 
              htmlFor="video-file-input" 
              className="chat-send-btn"
              style={{ display: 'inline-flex', padding: '12px 28px', fontSize: '0.95rem', cursor: 'pointer', margin: '0 auto' }}
            >
              Select MP4 Video File
            </label>

            {uploadStatus && (
              <div style={{ marginTop: '20px', padding: '14px', background: 'var(--bg-surface)', borderRadius: '8px', maxWidth: '600px', margin: '20px auto 0 auto', color: 'var(--accent-cyan)', fontFamily: 'JetBrains Mono', fontSize: '0.85rem' }}>
                {uploadStatus}
              </div>
            )}
          </div>
        </main>
      )}
    </div>
  );
}
