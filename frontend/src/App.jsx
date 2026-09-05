import { useState, useEffect, useRef, useCallback } from 'react';

/* ══════════════════════════════════════════════════════════════
   CONSTANTS
   ══════════════════════════════════════════════════════════════ */
const BACKEND_URL = 'http://localhost:5001';
const CHUNK_INTERVAL = 2000; // 2 seconds
const RECORD_DURATION = 1800; // 1.8s recording within 2s window

// Generate fake unknown phone numbers for simulation
const FAKE_NUMBERS = [
  '+1 (809) 555-0147',
  '+1 (312) 555-0198',
  '+91 98765 43210',
  '+44 7911 123456',
  '+1 (646) 555-0173',
  '+91 87654 32109',
  '+1 (415) 555-0162',
];

const getRandomNumber = () => FAKE_NUMBERS[Math.floor(Math.random() * FAKE_NUMBERS.length)];

/* ══════════════════════════════════════════════════════════════
   WAVEFORM VISUALIZER — Real Web Audio API analyser
   ══════════════════════════════════════════════════════════════ */
function WaveformVisualizer({ analyserNode, active, risk }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const barCount = 48;

  const getColor = () => {
    if (!active) return 'rgba(0, 240, 255, 0.3)';
    if (risk === 'HIGH') return '#ff4d4d';
    if (risk === 'SUSPICIOUS') return '#ffae00';
    return '#00f0ff';
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    const draw = () => {
      ctx.clearRect(0, 0, W, H);

      if (analyserNode && active) {
        const bufLen = analyserNode.frequencyBinCount;
        const data = new Uint8Array(bufLen);
        analyserNode.getByteFrequencyData(data);

        const barW = (W / barCount) - 2;
        const step = Math.floor(bufLen / barCount);
        const color = getColor();

        for (let i = 0; i < barCount; i++) {
          const val = data[i * step] / 255;
          const barH = Math.max(2, val * H * 0.85);
          const x = i * (barW + 2);
          const y = (H - barH) / 2;

          ctx.fillStyle = color;
          ctx.shadowColor = color;
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.roundRect(x, y, barW, barH, 1.5);
          ctx.fill();
        }
      } else {
        // Idle state — minimal bars
        const barW = (W / barCount) - 2;
        for (let i = 0; i < barCount; i++) {
          const x = i * (barW + 2);
          const y = (H - 3) / 2;
          ctx.fillStyle = 'rgba(0, 240, 255, 0.15)';
          ctx.fillRect(x, y, barW, 3);
        }
      }

      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [analyserNode, active, risk]);

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={80}
      style={{ width: '100%', height: '80px', borderRadius: '8px' }}
    />
  );
}

/* ══════════════════════════════════════════════════════════════
   TRUST SCORE PANEL
   ══════════════════════════════════════════════════════════════ */
function TrustScorePanel({ trustScore, aiLikelihood, risk, signals }) {
  const getRiskColor = () => {
    if (risk === 'HIGH') return 'var(--red)';
    if (risk === 'SUSPICIOUS') return 'var(--amber)';
    return 'var(--green)';
  };

  return (
    <div>
      {/* Score Gauges */}
      <div className="trust-gauge">
        <div className="gauge-item" style={{ borderColor: `${getRiskColor()}33` }}>
          <div className="gauge-value" style={{ color: 'var(--green)' }}>{trustScore}</div>
          <div className="gauge-label">Trust Score</div>
        </div>
        <div className="gauge-item" style={{ borderColor: `${getRiskColor()}33` }}>
          <div className="gauge-value" style={{ color: aiLikelihood > 60 ? 'var(--red)' : aiLikelihood > 35 ? 'var(--amber)' : 'var(--cyan)' }}>
            {aiLikelihood}
          </div>
          <div className="gauge-label">AI Likelihood</div>
        </div>
      </div>

      {/* Risk Badge */}
      <div style={{ textAlign: 'center', margin: '12px 0' }}>
        <span className={`risk-badge ${risk.toLowerCase()}`}>
          {risk === 'HIGH' && '🚨 '}
          {risk === 'SUSPICIOUS' && '⚠️ '}
          {risk === 'SAFE' && '✅ '}
          {risk}
        </span>
      </div>

      {/* Signals */}
      {signals && signals.length > 0 && (
        <div>
          <div className="section-divider">Signal Analysis</div>
          <ul className="signal-list">
            {signals.map((s, i) => (
              <li key={i} className="signal-item">
                <span className="signal-dot" style={{ background: getRiskColor() }} />
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   TIMELINE PANEL
   ══════════════════════════════════════════════════════════════ */
function TimelinePanel({ entries }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  if (!entries || entries.length === 0) return null;

  return (
    <div>
      <div className="section-divider">Chunk Timeline</div>
      <div className="timeline-panel" ref={scrollRef}>
        {entries.map((entry, i) => (
          <div key={i} className="timeline-entry">
            <span className="timeline-time">[{entry.time}]</span>
            <span className={`timeline-badge ${entry.risk.toLowerCase()}`}>{entry.risk}</span>
            <span className="timeline-score">Trust: {entry.trustScore}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   HOME SCREEN
   ══════════════════════════════════════════════════════════════ */
function HomeScreen({ onSimulateCall, callHistory }) {
  return (
    <div className="relative min-h-screen flex flex-col items-center px-4 pt-20 pb-16">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-black/80 backdrop-blur-md border-b border-cyan-500/10">
        <div className="flex items-center gap-3">
          <div className="neon-text-cyan font-mono text-sm tracking-widest font-bold">ECHO_LOCATOR</div>
        </div>
        <div className="status-badge">
          <div className="status-dot bg-cyan-500" />
          <span className="font-mono text-[10px] tracking-widest">STANDBY</span>
        </div>
      </header>

      <div className="w-full max-w-xl">
        {/* Branding */}
        <div className="text-center mb-10">
          <div className="text-5xl mb-4">🛡️</div>
          <h1 className="text-4xl md:text-5xl font-black mb-3 tracking-tight">
            EchoLocator
          </h1>
          <p className="text-gray-400 font-medium text-sm max-w-md mx-auto">
            Real-Time Call Safety Platform — Detecting AI-driven fake emergency calls using acoustic and behavioral voice analysis.
          </p>
        </div>

        {/* Main Card */}
        <main className="glass-card p-6 sm:p-8 relative overflow-hidden">
          <div className="corner-tl" /><div className="corner-tr" />
          <div className="corner-bl" /><div className="corner-br" />

          {/* Simulate Call Button */}
          <button
            id="simulate-call-btn"
            onClick={onSimulateCall}
            className="w-full py-4 bg-cyan-500 hover:bg-cyan-400 text-black font-black flex items-center justify-center gap-3 rounded-lg transition-all active:scale-95 shadow-[0_0_20px_rgba(6,182,212,0.4)] mb-8"
          >
            <span className="text-lg">📞</span> SIMULATE INCOMING CALL
          </button>

          {/* How it Works */}
          <div className="section-divider">How It Works</div>
          <div className="space-y-3 mb-6">
            {[
              { num: '01', title: 'Incoming Call', desc: 'A simulated call appears with an unknown number' },
              { num: '02', title: 'Accept & Protect', desc: 'Accept the call and enable AI protection' },
              { num: '03', title: 'Real-Time Analysis', desc: 'Every 2 seconds your audio is analyzed for AI voice patterns' },
              { num: '04', title: 'Live Trust Scoring', desc: 'Get instant risk assessment with detailed signal breakdown' },
            ].map((step) => (
              <div key={step.num} className="flex items-start gap-4 p-3 rounded-lg bg-white/[0.02] border border-white/5 hover:border-cyan-500/20 transition-colors">
                <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center text-cyan-400 font-mono text-xs font-bold flex-shrink-0">{step.num}</div>
                <div>
                  <h4 className="text-sm font-bold">{step.title}</h4>
                  <p className="text-xs text-gray-500">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Recent Suspicious Calls */}
          {callHistory.length > 0 && (
            <div>
              <div className="section-divider">Recent Suspicious Calls</div>
              {callHistory.slice(0, 5).map((call, i) => (
                <div key={i} className="call-history-item">
                  <div>
                    <div className="font-mono text-sm">{call.number}</div>
                    <div className="text-xs text-gray-500">{call.time}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`risk-badge ${call.risk.toLowerCase()}`} style={{ padding: '3px 10px', fontSize: '0.65rem' }}>
                      {call.risk}
                    </span>
                    {call.count > 1 && (
                      <span className="repeat-count" style={{ fontSize: '0.7rem' }}>×{call.count}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>

        <footer className="mt-12 text-center text-[10px] font-mono text-gray-600 tracking-widest uppercase">
          Powered by AASIST Neural Network — Protected by EchoLocator v2.0
        </footer>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   INCOMING CALL SCREEN
   ══════════════════════════════════════════════════════════════ */
function IncomingCallScreen({ phoneNumber, onAccept, onReject }) {
  return (
    <div className="incoming-call-screen">
      <div className="caller-avatar">
        <span style={{ fontSize: '48px', opacity: 0.6 }}>👤</span>
      </div>

      <div className="font-mono text-xs tracking-widest text-gray-500 uppercase mb-2">Incoming Call</div>
      <div className="text-2xl font-bold font-mono tracking-wide mb-1">{phoneNumber}</div>
      <div className="text-sm text-gray-500">Unknown Number</div>

      <div className="ringtone-dots">
        <div className="ringtone-dot" />
        <div className="ringtone-dot" />
        <div className="ringtone-dot" />
      </div>

      <div className="call-actions">
        <div className="text-center">
          <button
            id="reject-call-btn"
            className="call-btn call-btn-reject"
            onClick={onReject}
            aria-label="Reject call"
          >
            📵
          </button>
          <div className="text-xs text-gray-500 mt-3 font-mono">Decline</div>
        </div>
        <div className="text-center">
          <button
            id="accept-call-btn"
            className="call-btn call-btn-accept"
            onClick={onAccept}
            aria-label="Accept call"
          >
            📞
          </button>
          <div className="text-xs text-gray-500 mt-3 font-mono">Accept</div>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   PERMISSION MODAL
   ══════════════════════════════════════════════════════════════ */
function PermissionModal({ onEnable, onDismiss }) {
  return (
    <div className="permission-overlay">
      <div className="permission-card">
        <div className="permission-icon">🛡️</div>
        <h2 className="text-xl font-bold mb-2">Enable AI Protection?</h2>
        <p className="text-sm text-gray-400 mb-6 leading-relaxed">
          This uses your microphone to analyze the call in real time. Your audio is processed locally and sent to a secure backend for AI voice detection. No recordings are stored.
        </p>
        <button
          id="enable-protection-btn"
          className="permission-btn-primary"
          onClick={onEnable}
        >
          🔒 Enable Protection
        </button>
        <button
          id="dismiss-protection-btn"
          className="permission-btn-secondary"
          onClick={onDismiss}
        >
          Not Now
        </button>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   LIVE CALL SCREEN
   ══════════════════════════════════════════════════════════════ */
function LiveCallScreen({
  phoneNumber,
  callDuration,
  trustScore,
  aiLikelihood,
  risk,
  confidence,
  signals,
  repeatedNumber,
  repeatCallCount,
  timeline,
  analyserNode,
  onEndCall,
}) {
  const formatDuration = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="relative min-h-screen flex flex-col items-center px-4 pt-20 pb-16">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-black/80 backdrop-blur-md border-b border-cyan-500/10">
        <div className="flex items-center gap-3">
          <div className="neon-text-cyan font-mono text-sm tracking-widest font-bold">ECHO_LOCATOR</div>
        </div>
        <div className="status-badge" style={{
          borderColor: risk === 'HIGH' ? 'rgba(255,77,77,0.4)' : risk === 'SUSPICIOUS' ? 'rgba(255,174,0,0.4)' : undefined,
          color: risk === 'HIGH' ? 'var(--red)' : risk === 'SUSPICIOUS' ? 'var(--amber)' : undefined,
        }}>
          <div className="status-dot" style={{
            background: risk === 'HIGH' ? 'var(--red)' : risk === 'SUSPICIOUS' ? 'var(--amber)' : undefined,
            animation: 'pulseDot 1s ease-in-out infinite',
          }} />
          <span className="font-mono text-[10px] tracking-widest">PROTECTED</span>
        </div>
      </header>

      <div className="w-full max-w-xl">
        <main className="glass-card p-5 sm:p-6 relative overflow-hidden">
          <div className="corner-tl" /><div className="corner-tr" />
          <div className="corner-bl" /><div className="corner-br" />
          <div className="scan-line-container"><div className="scan-line" /></div>

          {/* Call Header */}
          <div className="live-call-header">
            <div>
              <div className="font-mono text-xs text-gray-500">Caller</div>
              <div className="font-mono text-lg font-bold">{phoneNumber}</div>
            </div>
            <div className="text-right">
              <div className="call-timer">{formatDuration(callDuration)}</div>
            </div>
          </div>

          {/* Protection + Mic Status */}
          <div className="flex items-center justify-between mb-4">
            <div className="protection-badge active">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
              </span>
              AI Protection Active
            </div>
            <div className="mic-indicator">
              <div className="mic-dot" />
              Listening
            </div>
          </div>

          {/* Waveform */}
          <div className="p-4 rounded-xl bg-black/40 border border-white/5 mb-4">
            <WaveformVisualizer analyserNode={analyserNode} active={true} risk={risk} />
          </div>

          {/* Repeat Number Warning */}
          {repeatedNumber && (
            <div className="repeat-warning">
              <span>⚠️</span>
              <span>This number has called multiple times recently</span>
              <span className="repeat-count">{repeatCallCount}</span>
            </div>
          )}

          {/* Trust Score Panel */}
          <TrustScorePanel
            trustScore={trustScore}
            aiLikelihood={aiLikelihood}
            risk={risk}
            signals={signals}
          />

          {/* Timeline */}
          <TimelinePanel entries={timeline} />

          {/* Confidence */}
          <div className="mt-4 flex items-center justify-center gap-2">
            <span className="text-[10px] font-mono text-gray-600 tracking-widest uppercase">Confidence</span>
            <div className="w-32 h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${confidence}%`,
                  background: `linear-gradient(90deg, var(--cyan), var(--electric))`,
                }}
              />
            </div>
            <span className="font-mono text-xs text-gray-500">{confidence}%</span>
          </div>

          {/* End Call */}
          <button
            id="end-call-btn"
            className="end-call-btn"
            onClick={onEndCall}
          >
            📵 END CALL
          </button>
        </main>

        <footer className="mt-8 text-center text-[10px] font-mono text-gray-600 tracking-widest uppercase">
          AASIST Neural Analysis — EchoLocator v2.0
        </footer>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   MAIN APP — State Machine
   ══════════════════════════════════════════════════════════════ */
export default function App() {
  // Screen states: HOME | INCOMING_CALL | PERMISSION | LIVE_CALL
  const [screen, setScreen] = useState('HOME');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [sessionId, setSessionId] = useState('');

  // Detection results
  const [trustScore, setTrustScore] = useState(85);
  const [aiLikelihood, setAiLikelihood] = useState(5);
  const [risk, setRisk] = useState('SAFE');
  const [confidence, setConfidence] = useState(50);
  const [signals, setSignals] = useState([]);
  const [repeatedNumber, setRepeatedNumber] = useState(false);
  const [repeatCallCount, setRepeatCallCount] = useState(0);

  // Call state
  const [callDuration, setCallDuration] = useState(0);
  const [timeline, setTimeline] = useState([]);
  const [callHistory, setCallHistory] = useState([]);

  // Audio refs
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const timerRef = useRef(null);
  const analyserRef = useRef(null);
  const audioContextRef = useRef(null);
  const incomingTimerRef = useRef(null);

  // Load call history from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem('echolocator_call_history');
      if (stored) setCallHistory(JSON.parse(stored));
    } catch (e) {
      console.warn('Failed to load call history:', e);
    }
  }, []);

  // Save call history to localStorage
  const saveCallHistory = useCallback((history) => {
    setCallHistory(history);
    try {
      localStorage.setItem('echolocator_call_history', JSON.stringify(history.slice(0, 50)));
    } catch (e) {
      console.warn('Failed to save call history:', e);
    }
  }, []);

  // ── Simulate Incoming Call ──
  const simulateIncomingCall = useCallback(() => {
    const number = getRandomNumber();
    setPhoneNumber(number);
    setSessionId(crypto.randomUUID());

    // Short delay before showing incoming call (simulate ring delay)
    incomingTimerRef.current = setTimeout(() => {
      setScreen('INCOMING_CALL');
    }, 800);
  }, []);

  // ── Accept Call ──
  const acceptCall = useCallback(() => {
    setScreen('PERMISSION');
  }, []);

  // ── Reject Call ──
  const rejectCall = useCallback(() => {
    setScreen('HOME');
  }, []);

  // ── Enable Protection ──
  const enableProtection = useCallback(async () => {
    try {
      // Request microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up Web Audio API analyser for waveform visualization
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.7;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Record call for repeat number tracking
      try {
        await fetch(`${BACKEND_URL}/record-call`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ phone_number: phoneNumber }),
        });
      } catch (e) {
        console.warn('Backend not reachable for call recording:', e);
      }

      // Check repeat number from localStorage
      const storedNumbers = JSON.parse(localStorage.getItem('echolocator_numbers') || '{}');
      const currentCount = (storedNumbers[phoneNumber] || 0) + 1;
      storedNumbers[phoneNumber] = currentCount;
      localStorage.setItem('echolocator_numbers', JSON.stringify(storedNumbers));

      if (currentCount > 1) {
        setRepeatedNumber(true);
        setRepeatCallCount(currentCount);
      }

      // Switch to live call screen
      setScreen('LIVE_CALL');
      setCallDuration(0);
      setTimeline([]);
      setTrustScore(85);
      setAiLikelihood(5);
      setRisk('SAFE');
      setConfidence(50);
      setSignals([]);

      // Start call timer
      timerRef.current = setInterval(() => {
        setCallDuration(prev => prev + 1);
      }, 1000);

      // Start audio chunk capture loop
      startDetectionLoop(stream);

    } catch (err) {
      console.error('Microphone access denied:', err);
      alert('Microphone access is required for real-time analysis.');
      setScreen('HOME');
    }
  }, [phoneNumber, sessionId]);

  // ── Detection Loop ──
  const startDetectionLoop = useCallback((stream) => {
    const captureAndAnalyze = () => {
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm'
      });
      const chunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        if (chunks.length === 0) return;
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
        await analyzeChunk(blob);
      };

      mediaRecorder.start();
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }, RECORD_DURATION);
    };

    // Initial capture
    captureAndAnalyze();

    // Continuous loop every 2 seconds
    intervalRef.current = setInterval(captureAndAnalyze, CHUNK_INTERVAL);
  }, [sessionId, phoneNumber]);

  // ── Analyze Chunk ──
  const analyzeChunk = useCallback(async (audioBlob) => {
    try {
      // Convert to WAV for backend
      const wavBlob = await convertToWav(audioBlob);

      const formData = new FormData();
      formData.append('audio', wavBlob, 'chunk.wav');
      formData.append('session_id', sessionId);
      formData.append('phone_number', phoneNumber);
      formData.append('timestamp', new Date().toISOString());

      const response = await fetch(`${BACKEND_URL}/analyze-chunk`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`Backend returned ${response.status}`);

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Update state with backend results
      setTrustScore(data.trust_score);
      setAiLikelihood(data.ai_likelihood);
      setRisk(data.risk);
      setConfidence(data.confidence);
      setSignals(data.signals || []);
      if (data.repeated_number) {
        setRepeatedNumber(true);
        setRepeatCallCount(data.repeat_call_count);
      }

      // Add to timeline
      setTimeline(prev => [...prev, {
        time: formatElapsed(prev.length * 2 + 2),
        risk: data.risk,
        trustScore: data.trust_score,
      }]);

    } catch (err) {
      console.warn('Backend unavailable, using local fallback:', err);
      fallbackDetection();
    }
  }, [sessionId, phoneNumber]);

  // ── Fallback Detection (when backend is offline) ──
  const fallbackDetection = useCallback(() => {
    // Use Web Audio API features for basic local analysis
    if (analyserRef.current) {
      const analyser = analyserRef.current;
      const bufLen = analyser.frequencyBinCount;
      const freqData = new Uint8Array(bufLen);
      analyser.getByteFrequencyData(freqData);

      // Basic spectral analysis
      const avgEnergy = freqData.reduce((a, b) => a + b, 0) / bufLen;
      const variance = freqData.reduce((a, b) => a + Math.pow(b - avgEnergy, 2), 0) / bufLen;

      // Simple heuristic scoring from frequency data
      const normalizedVar = Math.min(100, variance / 20);
      const hasVoice = avgEnergy > 10;

      if (hasVoice) {
        const localAiLikelihood = Math.max(5, Math.min(95, 50 - normalizedVar / 2 + Math.random() * 10));
        const localTrustScore = Math.max(5, 100 - localAiLikelihood);
        const localRisk = localAiLikelihood > 65 ? 'HIGH' : localAiLikelihood > 40 ? 'SUSPICIOUS' : 'SAFE';

        setAiLikelihood(Math.round(localAiLikelihood));
        setTrustScore(Math.round(localTrustScore));
        setRisk(localRisk);
        setConfidence(Math.min(70, 40 + Math.floor(Math.random() * 20)));
        setSignals([
          'Acoustic heuristic analysis active (backend offline)',
          normalizedVar < 30 ? 'Pitch variation appears limited' : 'Natural pitch variation detected',
          avgEnergy < 30 ? 'Low signal energy detected' : 'Normal signal energy',
        ]);
      }
    }

    setTimeline(prev => [...prev, {
      time: formatElapsed(prev.length * 2 + 2),
      risk: risk,
      trustScore: trustScore,
    }]);
  }, [risk, trustScore]);

  // ── Convert audio blob to WAV ──
  const convertToWav = async (blob) => {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const channelData = audioBuffer.getChannelData(0);
      const sampleRate = audioBuffer.sampleRate;

      // Encode to WAV
      const buffer = new ArrayBuffer(44 + channelData.length * 2);
      const view = new DataView(buffer);

      const writeStr = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
      };

      writeStr(0, 'RIFF');
      view.setUint32(4, 36 + channelData.length * 2, true);
      writeStr(8, 'WAVE');
      writeStr(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeStr(36, 'data');
      view.setUint32(40, channelData.length * 2, true);

      let offset = 44;
      for (let i = 0; i < channelData.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }

      await audioCtx.close();
      return new Blob([view], { type: 'audio/wav' });
    } catch (e) {
      console.warn('WAV conversion failed, sending raw blob:', e);
      return blob;
    }
  };

  // ── End Call ──
  const endCall = useCallback(() => {
    // Stop all audio
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }

    // Save to call history
    const entry = {
      number: phoneNumber,
      risk: risk,
      trustScore: trustScore,
      time: new Date().toLocaleString(),
      count: repeatCallCount || 1,
    };
    saveCallHistory([entry, ...callHistory]);

    // Reset
    setRepeatedNumber(false);
    setRepeatCallCount(0);
    analyserRef.current = null;
    setScreen('HOME');
  }, [phoneNumber, risk, trustScore, repeatCallCount, callHistory, saveCallHistory]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      if (incomingTimerRef.current) clearTimeout(incomingTimerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // ── Render ──
  return (
    <>
      {/* Ambient orbs */}
      <div className="orb orb-1" aria-hidden="true" />
      <div className="orb orb-2" aria-hidden="true" />

      {screen === 'HOME' && (
        <HomeScreen
          onSimulateCall={simulateIncomingCall}
          callHistory={callHistory}
        />
      )}

      {screen === 'INCOMING_CALL' && (
        <IncomingCallScreen
          phoneNumber={phoneNumber}
          onAccept={acceptCall}
          onReject={rejectCall}
        />
      )}

      {screen === 'PERMISSION' && (
        <PermissionModal
          onEnable={enableProtection}
          onDismiss={() => setScreen('HOME')}
        />
      )}

      {screen === 'LIVE_CALL' && (
        <LiveCallScreen
          phoneNumber={phoneNumber}
          callDuration={callDuration}
          trustScore={trustScore}
          aiLikelihood={aiLikelihood}
          risk={risk}
          confidence={confidence}
          signals={signals}
          repeatedNumber={repeatedNumber}
          repeatCallCount={repeatCallCount}
          timeline={timeline}
          analyserNode={analyserRef.current}
          onEndCall={endCall}
        />
      )}
    </>
  );
}


/* ── Helper ── */
function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = (seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}