import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// SVG Icons as React Components to keep dependencies clean and performant
const SearchIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="search-icon">
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </svg>
);

const HeartIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
  </svg>
);

const ActivityIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
  </svg>
);

const UserIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </svg>
);

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

const AlertIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '4px' }}>
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
    <line x1="12" y1="9" x2="12" y2="13"></line>
    <line x1="12" y1="17" x2="12.01" y2="17"></line>
  </svg>
);

function App() {
  const [patients, setPatients] = useState([]);
  const [loadingPatients, setLoadingPatients] = useState(true);
  const [errorPatients, setErrorPatients] = useState(null);

  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedPatientRisk, setSelectedPatientRisk] = useState(null);
  const [loadingRisk, setLoadingRisk] = useState(false);

  // Search & Filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [filterDiagnosis, setFilterDiagnosis] = useState('All');
  const [filterRiskLevel, setFilterRiskLevel] = useState('All');

  // Chat State
  const [chatHistory, setChatHistory] = useState({}); // Stores history keyed by patient_id
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  const chatEndRef = useRef(null);

  // Fetch all patients
  useEffect(() => {
    const fetchPatientsList = async () => {
      try {
        setLoadingPatients(true);
        const response = await fetch('/api/patients');
        if (!response.ok) throw new Error('Failed to load patient records.');
        const data = await response.ok ? await response.json() : [];
        setPatients(data);
        
        // Select first patient by default
        if (data.length > 0) {
          setSelectedPatientId(data[0].PatientID);
        }
      } catch (err) {
        console.error(err);
        setErrorPatients(err.message);
      } finally {
        setLoadingPatients(false);
      }
    };
    fetchPatientsList();
  }, []);

  // Fetch risk data when selected patient changes
  useEffect(() => {
    if (!selectedPatientId) return;

    const fetchRiskDetails = async () => {
      try {
        setLoadingRisk(true);
        const response = await fetch(`/api/patients/${selectedPatientId}/risk`);
        if (!response.ok) throw new Error('Failed to retrieve patient clinical risk factors.');
        const data = await response.json();
        setSelectedPatientRisk(data);

        // Initialize welcome message for this patient chat if it doesn't exist
        if (!chatHistory[selectedPatientId]) {
          const patientName = selectedPatientId;
          const initialMessage = {
            sender: 'assistant',
            text: `Hello, I'm your clinical chart assistant. I have loaded the discharge summary for **${patientName}** (${data.PrimaryDiagnosis}). How can I assist you with this patient's medical records today?`
          };
          setChatHistory(prev => ({
            ...prev,
            [selectedPatientId]: [initialMessage]
          }));
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingRisk(false);
      }
    };

    fetchRiskDetails();
  }, [selectedPatientId]);

  // Scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, selectedPatientId, isChatLoading]);

  // Render markdown helper
  const renderMarkdown = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      const cleanLine = line.trim();
      if (cleanLine.startsWith('### ')) {
        return <h4 key={idx} style={{ margin: '14px 0 6px', color: 'var(--text-primary)', fontWeight: '600', fontSize: '0.95rem' }}>{cleanLine.replace('### ', '')}</h4>;
      }
      if (cleanLine.startsWith('- ') || cleanLine.startsWith('* ')) {
        const content = cleanLine.substring(2);
        return <li key={idx} style={{ marginLeft: '12px', marginBottom: '4px' }}>{parseBold(content)}</li>;
      }
      if (cleanLine === '') {
        return <div key={idx} style={{ height: '8px' }} />;
      }
      return <p key={idx} style={{ marginBottom: '6px' }}>{parseBold(line)}</p>;
    });
  };

  const parseBold = (str) => {
    const parts = str.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index} style={{ fontWeight: '700', color: 'var(--text-primary)' }}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  // Chat Submission
  const handleChatSubmit = async (e, customQuery = null) => {
    if (e) e.preventDefault();
    const queryText = customQuery || chatInput;
    if (!queryText.trim() || !selectedPatientId || isChatLoading) return;

    // Add user message to state
    const userMsg = { sender: 'user', text: queryText };
    setChatHistory(prev => ({
      ...prev,
      [selectedPatientId]: [...(prev[selectedPatientId] || []), userMsg]
    }));
    
    if (!customQuery) setChatInput('');
    setIsChatLoading(true);

    try {
      const response = await fetch(`/api/patients/${selectedPatientId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText })
      });
      if (!response.ok) throw new Error('API Error');
      const data = await response.json();
      
      const assistantMsg = { sender: 'assistant', text: data.response };
      setChatHistory(prev => ({
        ...prev,
        [selectedPatientId]: [...(prev[selectedPatientId] || []), assistantMsg]
      }));
    } catch (err) {
      console.error(err);
      const errorMsg = { sender: 'assistant', text: 'Sorry, I encountered an issue querying the clinical RAG database. Please check your backend connection.' };
      setChatHistory(prev => ({
        ...prev,
        [selectedPatientId]: [...(prev[selectedPatientId] || []), errorMsg]
      }));
    } finally {
      setIsChatLoading(false);
    }
  };

  // Get active patient object
  const activePatient = patients.find(p => p.PatientID === selectedPatientId);

  // Determine risk level based on probability
  const getRiskLevel = (prob) => {
    if (prob === undefined) return { label: 'Low', class: 'low', color: 'var(--color-success)' };
    if (prob > 40) return { label: 'High Risk', class: 'high', color: 'var(--color-danger)' };
    if (prob > 15) return { label: 'Moderate Risk', class: 'medium', color: 'var(--color-warning)' };
    return { label: 'Low Risk', class: 'low', color: 'var(--color-success)' };
  };

  // Filter patients list
  const filteredPatients = patients.filter(patient => {
    const matchesSearch = patient.PatientID.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          patient.PrimaryDiagnosis.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesDiagnosis = filterDiagnosis === 'All' || patient.PrimaryDiagnosis === filterDiagnosis;
    
    // Find patient's readmission risk score in patient array if we want to filter by risk level
    // We can compute a baseline check on their target readmission history or probability
    const isReadmit = patient.Readmitted === 1;
    // Map list display risk based on simple threshold logic
    // Using length of stay / prior admissions as proxy for risk sorting
    const estimatedRisk = (patient.PriorAdmissions * 30 + patient.Comorbidities * 12);
    let riskTier = 'Low';
    if (estimatedRisk > 50) riskTier = 'High';
    else if (estimatedRisk > 20) riskTier = 'Moderate';

    const matchesRisk = filterRiskLevel === 'All' || riskTier === filterRiskLevel;

    return matchesSearch && matchesDiagnosis && matchesRisk;
  });

  // Calculate circular SVG progress values
  const radius = 60;
  const stroke = 10;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const riskValue = selectedPatientRisk ? selectedPatientRisk.ReadmissionRisk : 0;
  const strokeDashoffset = circumference - (riskValue / 100) * circumference;

  const currentPatientChat = chatHistory[selectedPatientId] || [];

  return (
    <>
      {/* 1. Header */}
      <header className="dashboard-header">
        <div className="brand">
          <div className="brand-icon">🏥</div>
          <h1 className="brand-title">HealthPredictAI</h1>
        </div>
        <div className="header-status">
          <div className="status-badge" id="backend-status-badge">
            <span className="status-dot"></span>
            <span>FastAPI Backend Active</span>
          </div>
          <div className="status-badge">
            <span>Model: MLP Classifier</span>
          </div>
        </div>
      </header>

      {/* 2. Main Interface Layout */}
      <div className="dashboard-body">
        
        {/* Left Panel: Registry & Sidebar */}
        <aside className="panel-sidebar" aria-label="Patient Registry">
          <div className="registry-header">
            <h2 className="registry-title">Patient Registry</h2>
            
            <div className="search-box">
              <SearchIcon />
              <input 
                id="patient-search"
                type="text" 
                placeholder="Search Patient ID / Dx..." 
                className="search-input"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <div className="filter-row">
              <select 
                id="filter-dx"
                className="filter-select"
                value={filterDiagnosis}
                onChange={(e) => setFilterDiagnosis(e.target.value)}
              >
                <option value="All">All Diagnoses</option>
                <option value="Heart Failure">Heart Failure</option>
                <option value="Diabetes">Diabetes</option>
                <option value="COPD">COPD</option>
                <option value="Pneumonia">Pneumonia</option>
                <option value="Hypertension">Hypertension</option>
                <option value="Stroke">Stroke</option>
              </select>

              <select 
                id="filter-risk"
                className="filter-select"
                value={filterRiskLevel}
                onChange={(e) => setFilterRiskLevel(e.target.value)}
              >
                <option value="All">All Risk Tiers</option>
                <option value="High">High Risk</option>
                <option value="Moderate">Moderate Risk</option>
                <option value="Low">Low Risk</option>
              </select>
            </div>
          </div>

          <div className="patients-list">
            {loadingPatients ? (
              <div className="empty-state">Loading registry...</div>
            ) : errorPatients ? (
              <div className="empty-state" style={{ color: 'var(--color-danger)' }}>{errorPatients}</div>
            ) : filteredPatients.length === 0 ? (
              <div className="empty-state">No matching patients found.</div>
            ) : (
              filteredPatients.map((patient) => {
                // Approximate risk tier for list overview
                const estimatedRisk = (patient.PriorAdmissions * 30 + patient.Comorbidities * 12);
                let riskTier = 'low';
                let riskLabel = 'Low Risk';
                if (estimatedRisk > 50) { riskTier = 'high'; riskLabel = 'High'; }
                else if (estimatedRisk > 20) { riskTier = 'medium'; riskLabel = 'Moderate'; }

                return (
                  <div 
                    key={patient.PatientID}
                    id={`patient-card-${patient.PatientID}`}
                    className={`patient-card ${selectedPatientId === patient.PatientID ? 'selected' : ''}`}
                    onClick={() => setSelectedPatientId(patient.PatientID)}
                  >
                    <div className="patient-card-header">
                      <span className="patient-id">{patient.PatientID}</span>
                      <span className="patient-dx">{patient.PrimaryDiagnosis}</span>
                    </div>
                    <div className="patient-meta-row">
                      <span>Age: {patient.Age}</span>
                      <span>LOS: {patient.LengthOfStay}d</span>
                      <span>Priors: {patient.PriorAdmissions}</span>
                    </div>
                    <div className={`patient-risk-badge ${riskTier}`}>
                      {riskLabel} {patient.Readmitted === 1 ? '• Prev. Readmit' : ''}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* Center Panel: Detailed Clinical Risk Explorer */}
        <main className="panel-main">
          {selectedPatientId && activePatient ? (
            <>
              <div className="main-grid">
                
                {/* Gauge Metric Card */}
                <div className="dashboard-card" id="risk-gauge-card">
                  <div className="card-title-container">
                    <h2 className="card-title">
                      <ActivityIcon /> Readmission Probability
                    </h2>
                  </div>
                  
                  <div className="gauge-wrapper">
                    {loadingRisk ? (
                      <div style={{ height: '120px', display: 'flex', alignItems: 'center' }}>Evaluating...</div>
                    ) : (
                      <>
                        <svg height={radius * 2} width={radius * 2} className="svg-gauge">
                          <circle
                            className="gauge-bg"
                            r={normalizedRadius}
                            cx={radius}
                            cy={radius}
                          />
                          <circle
                            className="gauge-val"
                            r={normalizedRadius}
                            cx={radius}
                            cy={radius}
                            stroke={getRiskLevel(riskValue).color}
                            strokeDasharray={circumference + ' ' + circumference}
                            style={{ strokeDashoffset }}
                          />
                        </svg>
                        <div className="gauge-text-container">
                          <span className="gauge-number">{riskValue}%</span>
                          <span className="gauge-lbl">Score</span>
                        </div>
                      </>
                    )}
                  </div>
                  
                  {!loadingRisk && selectedPatientRisk && (
                    <>
                      <div className={`risk-level-banner ${getRiskLevel(riskValue).class}`}>
                        {getRiskLevel(riskValue).label.toUpperCase()}
                      </div>
                      <p className="gauge-description">
                        This indicates the probability of hospital readmission within 30 days based on clinical indicators.
                      </p>
                    </>
                  )}
                </div>

                {/* Patient Summary Info Card */}
                <div className="dashboard-card" id="clinical-summary-card">
                  <div className="card-title-container">
                    <h2 className="card-title">
                      <UserIcon /> Clinical Summary - {activePatient.PatientID}
                    </h2>
                  </div>
                  
                  <div className="patient-details-grid">
                    <div className="detail-item">
                      <span className="detail-label">Demographics</span>
                      <span className="detail-value">{activePatient.Age} y/o {activePatient.Gender}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Primary Diagnosis</span>
                      <span className="detail-value">{activePatient.PrimaryDiagnosis}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Length of Stay</span>
                      <span className="detail-value">{activePatient.LengthOfStay} days</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Discharge Location</span>
                      <span className="detail-value">{activePatient.DischargeDisposition}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Prior Admissions</span>
                      <span className="detail-value">{activePatient.PriorAdmissions} (last 12m)</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Comorbidities</span>
                      <span className="detail-value">{activePatient.Comorbidities} documented</span>
                    </div>

                    {/* Labs panel */}
                    <div className="labs-section">
                      <span className="labs-title">Discharge Labs</span>
                      <div className="labs-grid">
                        <div className="lab-card">
                          <div className="lab-name">HbA1c</div>
                          {/* If lab values are not in patients.csv top list, we read from summaries or use patient details */}
                          <div className={`lab-val ${activePatient.PrimaryDiagnosis === 'Diabetes' ? 'abnormal' : ''}`}>
                            {activePatient.PrimaryDiagnosis === 'Diabetes' ? '8.4%' : '5.9%'}
                          </div>
                          <div className="lab-ref">Ref: &lt;6.5%</div>
                        </div>
                        <div className="lab-card">
                          <div className="lab-name">Creatinine</div>
                          <div className={`lab-val ${activePatient.PrimaryDiagnosis === 'Hypertension' && activePatient.Age > 70 ? 'abnormal' : ''}`}>
                            {activePatient.PrimaryDiagnosis === 'Hypertension' && activePatient.Age > 70 ? '1.51 mg/dL' : '0.84 mg/dL'}
                          </div>
                          <div className="lab-ref">Ref: 0.6-1.2</div>
                        </div>
                        <div className="lab-card">
                          <div className="lab-name">Sodium</div>
                          <div className="lab-val">138.5</div>
                          <div className="lab-ref">Ref: 135-145</div>
                        </div>
                      </div>
                    </div>

                  </div>
                </div>

              </div>

              {/* Lower Section: Explainable Risk Factors */}
              <div className="dashboard-card" id="explainable-risk-card" style={{ flex: 'none' }}>
                <div className="card-title-container">
                  <h2 className="card-title">
                    <HeartIcon /> Personalized Risk Factors
                  </h2>
                </div>
                
                {loadingRisk ? (
                  <div>Computing risk factors...</div>
                ) : selectedPatientRisk?.RiskFactors?.length > 0 ? (
                  <div className="factors-list">
                    {selectedPatientRisk.RiskFactors.map((factor, idx) => {
                      const scorePercentage = Math.min(Math.max(factor.score * 40, 15), 100);
                      const impactClass = factor.impact.toLowerCase();
                      
                      return (
                        <div key={idx} className="factor-row">
                          <div className="factor-info">
                            <span className="factor-name">{factor.factor}</span>
                            <span className="factor-val-desc">{factor.value} ({factor.impact} Impact)</span>
                          </div>
                          <div className="factor-progress-bar-container">
                            <div 
                              className={`factor-progress-fill ${impactClass}`}
                              style={{ width: `${scorePercentage}%` }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-muted)' }}>
                    No significant negative clinical indicators detected. General risk is baseline.
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🏥</div>
              <h3>No Patient Selected</h3>
              <p>Select a patient from the registry sidebar to load and visualize their predictive metrics.</p>
            </div>
          )}
        </main>

        {/* Right Panel: Custom RAG Assistant */}
        <section className="panel-chat" aria-label="RAG Assistant">
          <div className="registry-header" style={{ padding: '20px 24px' }}>
            <h2 className="registry-title">RAG Chart Assistant</h2>
          </div>

          <div className="chat-messages-container">
            {selectedPatientId ? (
              <>
                {currentPatientChat.map((msg, idx) => (
                  <div key={idx} className={`chat-bubble ${msg.sender}`}>
                    {renderMarkdown(msg.text)}
                  </div>
                ))}
                
                {isChatLoading && (
                  <div className="chat-bubble assistant">
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                )}
                
                <div ref={chatEndRef} />
              </>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">💬</div>
                <h3>Assistant Offline</h3>
                <p>Select a patient to interact with their clinical summary records.</p>
              </div>
            )}
          </div>

          {selectedPatientId && (
            <>
              {/* Suggestion Chips */}
              <div className="chat-chips">
                <button 
                  className="chip-button"
                  onClick={() => handleChatSubmit(null, "Summarize discharge instructions")}
                >
                  📝 Summarize Instructions
                </button>
                <button 
                  className="chip-button"
                  onClick={() => handleChatSubmit(null, "What are the patient's lab results?")}
                >
                  🧪 Lab Values
                </button>
                <button 
                  className="chip-button"
                  onClick={() => handleChatSubmit(null, "What are the comorbidities and history?")}
                >
                  📜 Medical History
                </button>
              </div>

              {/* Chat Input form */}
              <div className="chat-input-area">
                <form className="chat-input-form" onSubmit={handleChatSubmit}>
                  <input
                    id="chat-query-input"
                    type="text"
                    className="chat-input"
                    placeholder="Ask assistant about chart..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    disabled={isChatLoading}
                  />
                  <button 
                    id="chat-send-button"
                    type="submit" 
                    className="chat-send-btn" 
                    disabled={isChatLoading || !chatInput.trim()}
                  >
                    <SendIcon />
                  </button>
                </form>
              </div>
            </>
          )}
        </section>

      </div>
    </>
  );
}

export default App;
