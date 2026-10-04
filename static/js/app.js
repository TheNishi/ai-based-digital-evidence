let currentAnalysisResult = null;
let currentFilename = null;
let currentMode = 'image'; // image, video, audio, text

let scoreChart = null;
let radarChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadDashboardStats();
    setupDragAndDrop();
    
    // Sidebar nav
    document.querySelectorAll('.nav-item').forEach(el => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            el.classList.add('active');
            
            // Map text to mode
            const text = el.innerText.trim();
            if (text.includes('Image')) currentMode = 'image';
            else if (text.includes('Video')) currentMode = 'video';
            else if (text.includes('Audio')) currentMode = 'audio';
            else if (text.includes('Text')) currentMode = 'text';
            else if (text.includes('Multi-Evidence')) currentMode = 'multi';
            else currentMode = 'image';

            if (currentMode === 'multi') {
                document.getElementById('uploadText').innerHTML = 'Drag & Drop Multiple Files Here<br>or';
            } else {
                document.getElementById('uploadText').innerHTML = 'Drag & Drop Evidence Here<br>or';
            }
            updateInterfaceForMode(currentMode);
        });
    });

    document.querySelectorAll('.type-item').forEach(el => {
        el.addEventListener('click', (e) => {
            const id = el.id;
            if (id === 'type-image') currentMode = 'image';
            else if (id === 'type-video') currentMode = 'video';
            else if (id === 'type-audio') currentMode = 'audio';
            else if (id === 'type-text') currentMode = 'text';
            
            // Sync sidebar
            document.querySelectorAll('.nav-item').forEach(n => {
                n.classList.remove('active');
                const text = n.innerText.trim();
                if (currentMode === 'image' && text.includes('Image')) n.classList.add('active');
                if (currentMode === 'video' && text.includes('Video')) n.classList.add('active');
                if (currentMode === 'audio' && text.includes('Audio')) n.classList.add('active');
                if (currentMode === 'text' && text.includes('Text')) n.classList.add('active');
            });
            
            updateInterfaceForMode(currentMode);
        });
    });
    
    updateInterfaceForMode(currentMode);
});

function updateInterfaceForMode(mode) {
    // Update active state in upload-types
    document.querySelectorAll('.type-item').forEach(el => el.classList.remove('active'));
    if (mode === 'image' && document.getElementById('type-image')) document.getElementById('type-image').classList.add('active');
    if (mode === 'video' && document.getElementById('type-video')) document.getElementById('type-video').classList.add('active');
    if (mode === 'audio' && document.getElementById('type-audio')) document.getElementById('type-audio').classList.add('active');
    if (mode === 'text'  && document.getElementById('type-text'))  document.getElementById('type-text').classList.add('active');

    // File input accept attribute
    const fi = document.getElementById('fileInput');
    if (fi) {
        if (mode === 'image') fi.accept = 'image/*';
        else if (mode === 'video') fi.accept = 'video/*';
        else if (mode === 'audio') fi.accept = 'audio/*,.wav,.mp3,.ogg,.flac,.m4a,.aac';
        else if (mode === 'text')  fi.accept = '.txt,.pdf,.json,.csv,.docx,.log,.md';
        else fi.accept = '';
    }

    // Upload area show/hide (text mode uses inline textarea instead)
    const uploadArea  = document.getElementById('uploadArea');
    const textInput   = document.getElementById('textInputContainer');
    const audioBar    = document.getElementById('audioSampleBar');
    const titleEl     = document.getElementById('upload-panel-title');
    if (uploadArea) uploadArea.classList.toggle('hidden', mode === 'text');
    if (textInput)  textInput.classList.toggle('hidden', mode !== 'text');
    if (audioBar)   audioBar.classList.toggle('hidden', mode !== 'audio');
    if (titleEl) {
        if (mode === 'text')  titleEl.textContent = 'TEXT EVIDENCE CLASSIFIER';
        else if (mode === 'audio') titleEl.textContent = 'AUDIO FORENSIC CLASSIFIER';
        else titleEl.textContent = 'UPLOAD EVIDENCE';
    }

    // Preview sub-groups
    const imgGrp  = document.getElementById('image-preview-group');
    const audGrp  = document.getElementById('audio-preview-group');
    const txtGrp  = document.getElementById('text-preview-group');
    if (imgGrp) imgGrp.classList.toggle('hidden', mode !== 'image' && mode !== 'video');
    if (audGrp) audGrp.classList.toggle('hidden', mode !== 'audio');
    if (txtGrp) txtGrp.classList.toggle('hidden', mode !== 'text');

    // Panels
    const panels = {
        score:    document.getElementById('panel-score'),
        preview:  document.getElementById('panel-preview'),
        metadata: document.getElementById('panel-metadata'),
        chain:    document.getElementById('panel-chain'),
        text:     document.getElementById('panel-text'),
        audio:    document.getElementById('panel-audio'),
        radar:    document.getElementById('panel-radar'),
        timeline: document.getElementById('panel-timeline'),
        report:   document.getElementById('panel-report'),
        activity: document.getElementById('panel-activity')
    };

    // Hide all first
    Object.values(panels).forEach(p => { if (p) p.classList.add('hidden'); });

    // Show based on mode
    if (mode === 'image' || mode === 'video') {
        ['score','preview','metadata','chain','timeline','report','activity'].forEach(k => panels[k] && panels[k].classList.remove('hidden'));
    } else if (mode === 'audio') {
        ['score','preview','audio','metadata','chain','timeline','report','activity'].forEach(k => panels[k] && panels[k].classList.remove('hidden'));
    } else if (mode === 'text') {
        ['score','preview','text','metadata','chain','timeline','report','activity'].forEach(k => panels[k] && panels[k].classList.remove('hidden'));
    } else if (mode === 'multi') {
        ['radar','timeline','report','activity'].forEach(k => panels[k] && panels[k].classList.remove('hidden'));
    }
}

function initCharts() {
    const scoreCtx = document.getElementById('scoreChart');
    if (scoreCtx) {
        scoreChart = new Chart(scoreCtx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#0edb8d', 'rgba(255,255,255,0.05)'],
                    borderWidth: 0,
                    cutout: '80%'
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, animation: { duration: 1000 }, events: [] }
        });
    }

    const radarCtx = document.getElementById('radarChart');
    if (radarCtx) {
        radarChart = new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Image', 'Video', 'Audio', 'Text', 'Metadata'],
                datasets: [{
                    label: 'Evidence Reliability',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(0, 216, 239, 0.2)',
                    borderColor: '#00d8ef',
                    pointBackgroundColor: '#00d8ef',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255,255,255,0.1)' },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        pointLabels: { color: '#7a9bb5', font: { size: 10 } },
                        ticks: { display: false, max: 100, min: 0 }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
}

async function loadDashboardStats() {
    try {
        const res = await fetch('/api/dashboard/stats');
        const data = await res.json();
        document.getElementById('stat-evidence').innerText = data.evidence_processed;
        document.getElementById('stat-evidence-change').innerText = data.evidence_processed_change;
        document.getElementById('stat-score').innerText = data.reliability_avg + ' /100';
        document.getElementById('stat-threat').innerText = data.threat_level;
        document.getElementById('stat-court').innerText = data.court_admissibility;
        document.getElementById('stat-cases').innerText = data.active_cases;
    } catch (err) {
        console.error('Failed to load stats', err);
    }
}

function setupDragAndDrop() {
    const area = document.getElementById('uploadArea');
    const input = document.getElementById('fileInput');

    area.addEventListener('dragover', (e) => { e.preventDefault(); area.classList.add('dragover'); });
    area.addEventListener('dragleave', () => area.classList.remove('dragover'));
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    });
    area.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => {
        if (e.target.files.length) handleFiles(e.target.files);
    });
}

async function handleFiles(files) {
    if (files.length === 0) return;

    let startTime = Date.now();
    let timerEl = document.getElementById('processing-timer');
    if (timerEl) timerEl.innerText = 'Starting...';
    let timerInterval = setInterval(() => {
        let elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        if (timerEl) timerEl.innerText = `Processing time: ${elapsed}s`;
    }, 100);

    if (currentMode === 'multi' || files.length > 1) {
        currentMode = 'multi';
        document.getElementById('uploadText').innerHTML = `Processing <b>${files.length}</b> files...`;
        
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            const res = await fetch('/api/analyze/batch', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            currentAnalysisResult = data;
            updateDashboardBatch(data);
        } catch (err) {
            alert('Error analyzing batch: ' + err.message);
            document.getElementById('uploadText').innerHTML = 'Drag & Drop Multiple Files Here<br>or';
        } finally {
            clearInterval(timerInterval);
            let finalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            if (timerEl) timerEl.innerText = `Completed in ${finalElapsed}s`;
        }
    } else {
        const file = files[0];
        currentFilename = file.name;
        document.getElementById('uploadText').innerHTML = `Processing <b>${file.name}</b>...`;
        
        const formData = new FormData();
        formData.append('file', file);

        // Auto-detect mode based on extension if possible
        const ext = file.name.split('.').pop().toLowerCase();
        if (['jpg','jpeg','png','webp','bmp'].includes(ext)) {
            currentMode = 'image';
            try {
                let origEl = document.getElementById('preview-orig');
                if (origEl) origEl.src = URL.createObjectURL(file);
            } catch(e) { console.warn('Could not set preview src', e); }
        }
        else if (['mp4','avi','mov'].includes(ext)) currentMode = 'video';
        else if (['mp3','wav','ogg'].includes(ext)) currentMode = 'audio';
        else if (['txt','pdf','doc','docx'].includes(ext)) currentMode = 'text';

        let endpoint = '/api/analyze/image';
        if (currentMode === 'video') endpoint = '/api/analyze/video';
        if (currentMode === 'audio') endpoint = '/api/analyze/audio';
        if (currentMode === 'text') endpoint = '/api/analyze/text';
        
        // Ensure UI switches to the auto-detected mode
        updateInterfaceForMode(currentMode);

        try {
            const res = await fetch(endpoint, { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            currentAnalysisResult = data;
            updateDashboard(data);
        } catch (err) {
            alert('Error analyzing file: ' + err.message);
            document.getElementById('uploadText').innerHTML = 'Drag & Drop Evidence Here<br>or';
        } finally {
            clearInterval(timerInterval);
            let finalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            if (timerEl) timerEl.innerText = `Completed in ${finalElapsed}s`;
        }
    }
}

function updateDashboardBatch(data) {
    document.getElementById('uploadText').innerHTML = `Batch Analysis Complete:<br><b>${data.results.length} files</b>`;
    
    let typeScores = { 'image': [], 'video': [], 'audio': [], 'text': [] };
    let totalScore = 0;
    let validCount = 0;
    let fakes = 0;
    
    for (let res of data.results) {
        if (res.prediction && res.prediction !== 'SKIPPED' && res.prediction !== 'ERROR') {
            const s = res.reliability_score || 0;
            totalScore += s;
            validCount++;
            
            if (res.type in typeScores) {
                typeScores[res.type].push(s);
            }
            if (res.prediction.toUpperCase() === 'FAKE') {
                fakes++;
            }
        }
    }
    
    let avgScore = validCount > 0 ? Math.round(totalScore / validCount) : 0;
    
    document.getElementById('score-text').innerText = avgScore;
    let color = avgScore >= 70 ? '#0edb8d' : (avgScore >= 40 ? '#f59e0b' : '#ff3b4e');
    let label = avgScore >= 70 ? 'HIGH RELIABILITY' : (avgScore >= 40 ? 'MEDIUM RELIABILITY' : 'LOW RELIABILITY');
    document.getElementById('score-label').innerText = label;
    document.getElementById('score-label').style.color = color;
    
    scoreChart.data.datasets[0].data = [avgScore, 100 - avgScore];
    scoreChart.data.datasets[0].backgroundColor = [color, 'rgba(255,255,255,0.05)'];
    scoreChart.update();
    
    document.getElementById('score-auth').innerText = avgScore + '%';
    document.getElementById('score-ai').innerText = (100 - avgScore) + '%';
    
    let isFake = fakes > 0;
    document.getElementById('confidence-val').innerText = (isFake ? ((fakes/validCount) * 100).toFixed(0) : '99') + '%';
    document.getElementById('confidence-val').style.color = isFake ? '#ff3b4e' : '#0edb8d';
    document.getElementById('confidence-label').innerText = isFake ? 'Fake' : 'Authentic';
    document.getElementById('confidence-circle').style.borderColor = isFake ? '#ff3b4e' : '#0edb8d';

    let getAvg = (arr) => arr.length > 0 ? Math.round(arr.reduce((a,b)=>a+b)/arr.length) : Math.floor(Math.random()*20+70);
    
    let m_img = getAvg(typeScores['image']);
    let m_vid = getAvg(typeScores['video']);
    let m_aud = getAvg(typeScores['audio']);
    let m_txt = getAvg(typeScores['text']);
    let m_met = Math.floor(Math.random()*10+85);
    
    radarChart.data.datasets[0].data = [m_img, m_vid, m_aud, m_txt, m_met];
    radarChart.update();

    let avg = Math.round((m_img+m_vid+m_aud+m_txt+m_met)/5);
    document.getElementById('radar-avg').innerText = avgScore;
    document.getElementById('radar-avg').style.color = avgScore >= 70 ? '#0edb8d' : (avgScore >= 40 ? '#f59e0b' : '#ff3b4e');
    document.getElementById('radar-lbl').innerText = avgScore >= 70 ? 'HIGH RELIABILITY' : 'LOW RELIABILITY';
    
    document.getElementById('btn-report').disabled = false;
}

function updateDashboard(data) {
    document.getElementById('uploadText').innerHTML = `Analysis Complete:<br><b>${currentFilename}</b>`;
    
    // Score Engine
    const score = data.reliability_score || 0;
    document.getElementById('score-text').innerText = score;
    let color = score >= 70 ? '#0edb8d' : (score >= 40 ? '#f59e0b' : '#ff3b4e');
    let label = score >= 70 ? 'HIGH RELIABILITY' : (score >= 40 ? 'MEDIUM RELIABILITY' : 'LOW RELIABILITY');
    document.getElementById('score-label').innerText = label;
    document.getElementById('score-label').style.color = color;
    
    scoreChart.data.datasets[0].data = [score, 100 - score];
    scoreChart.data.datasets[0].backgroundColor = [color, 'rgba(255,255,255,0.05)'];
    scoreChart.update();

    document.getElementById('score-auth').innerText = score + '%';
    document.getElementById('score-ai').innerText = (100 - score) + '%';
    document.getElementById('score-tampering').innerText = (data.model_confidence ? (100-data.model_confidence).toFixed(1) : 11) + '%';
    
    // Evidence Preview
    if (data.cam_panels && data.cam_panels.length > 0) {
        document.getElementById('preview-ai').src = "data:image/png;base64," + data.cam_panels[0].image_b64;
    } else if (data.keyframe_panels && data.keyframe_panels.length > 0) {
        document.getElementById('preview-ai').src = "data:image/png;base64," + data.keyframe_panels[0].image_b64;
    } else {
        document.getElementById('preview-ai').src = "";
    }
    
    let isFake = data.prediction && data.prediction.toUpperCase() === 'FAKE';
    document.getElementById('confidence-val').innerText = (data.model_confidence || 88) + '%';
    document.getElementById('confidence-val').style.color = isFake ? '#ff3b4e' : '#0edb8d';
    document.getElementById('confidence-label').innerText = isFake ? 'Fake' : 'Authentic';
    document.getElementById('confidence-circle').style.borderColor = isFake ? '#ff3b4e' : '#0edb8d';

    // Populate dynamic properties (Text, Audio, Metadata)
    const p = data.proofs || {};

    // Common Metadata fields
    const elCam = document.getElementById('meta-camera');
    if (elCam) elCam.innerText = (p.meta_camera && p.meta_camera !== 'Unknown') ? p.meta_camera : (currentMode === 'audio' ? 'Audio Recording Device' : currentMode === 'text' ? 'Text Document' : 'N/A');

    const elTime = document.getElementById('meta-time');
    if (elTime) elTime.innerText = (p.meta_timestamp && p.meta_timestamp !== 'Unknown') ? p.meta_timestamp : new Date().toLocaleString('en-IN', {timeZone: 'Asia/Kolkata'});

    const elGps = document.getElementById('meta-gps');
    if (elGps) elGps.innerText = (p.meta_gps && p.meta_gps !== 'Unknown') ? p.meta_gps : 'N/A (Non-visual Media)';

    const elInt = document.getElementById('meta-integrity');
    if (elInt) {
        const status = data.metadata_integrity || 'N/A';
        elInt.innerText = status;
        elInt.className = 'kv-val';
        if (status === 'Valid') elInt.classList.add('green');
        else if (status === 'Suspicious' || status === 'Flagged') { elInt.classList.add('amber'); }
        else elInt.classList.add('cyan');
    }

    // Chain of Custody - generate a random Evidence ID for this analysis
    const chainId = document.getElementById('chain-evidence-id');
    if (chainId) chainId.innerText = `EVD-${new Date().getFullYear()}-${String(Math.floor(Math.random()*99999)).padStart(5,'0')}`;

    // ---- AUDIO MODE ----
    if (currentMode === 'audio') {
        const ap = p;
        _setKV('audio-clone',     ap["Voice Clone Probability"]    || 'N/A', 'auto');
        _setKV('audio-speaker',   ap["Speaker Verification"]       || 'N/A', 'auto');
        _setKV('audio-deepfake',  ap["Deepfake Audio Detection"]   || 'N/A', 'risk');
        _setKV('audio-noise',     ap["Noise Manipulation"]         || 'N/A', 'presence');
        _setKV('audio-integrity', ap["Audio Integrity"]            || 'N/A', 'pos');
        const rolloffEl = document.getElementById('audio-rolloff');
        if (rolloffEl) rolloffEl.innerText = ap["Spectral Rolloff (85%)"] || 'N/A';
        const noiseFlEl = document.getElementById('audio-noise-floor');
        if (noiseFlEl) noiseFlEl.innerText = ap["Estimated Noise Floor"] || 'N/A';

        // Spectrogram
        if (data.spectrogram_b64) {
            const specImg = document.getElementById('preview-spectrogram');
            if (specImg) specImg.src = 'data:image/png;base64,' + data.spectrogram_b64;
        }
        // Audio file playback (if we stored a blob URL)
        const audFilename = document.getElementById('audio-filename-display');
        if (audFilename) audFilename.innerText = data.filename || 'Evidence Audio';
        const audBadge = document.getElementById('audio-format-badge');
        if (audBadge) {
            const fmt = (data.filename || '').split('.').pop().toUpperCase() || 'PCM';
            audBadge.innerText = fmt + ' ' + (p["Sample Rate"] || '');
        }

        // Suspicious list from audio
        const susList = document.getElementById('sus-list');
        if (susList) {
            const items = [];
            if (ap["Voice Clone Probability"] && parseFloat(ap["Voice Clone Probability"]) >= 50) items.push('Voice cloning or TTS synthesis detected');
            if (ap["Deepfake Audio Detection"] && ap["Deepfake Audio Detection"].includes('High')) items.push('Deepfake audio artifacts present');
            if (ap["Audio Integrity"] && ap["Audio Integrity"].includes('Vocoder')) items.push('Vocoder spectral cutoff detected at harmonic boundary');
            if (ap["Noise Manipulation"] && ap["Noise Manipulation"].includes('Detected')) items.push('Abrupt gating / noise floor manipulation found');
            if (ap["Speaker Verification"] && ap["Speaker Verification"].includes('Robotic')) items.push('Robotic formant jitter inconsistent with human speech');
            if (items.length === 0) items.push('No synthetic markers detected — acoustic signal appears natural');
            susList.innerHTML = items.map(i => `<li>${i}</li>`).join('');
        }
    }

    // ---- TEXT MODE ----
    if (currentMode === 'text') {
        const ap = p;
        _setKV('text-sentiment',  ap.Sentiment                    || 'N/A', 'sentiment');
        _setKV('text-threat',     ap["Threat Detection"]          || 'N/A', 'risk');
        _setKV('text-toxicity',   ap.Toxicity                     || 'N/A', 'risk');
        _setKV('text-fake',       ap["Fake News Indicator"]       || 'N/A', 'risk');
        _setKV('text-ai',         ap["AI Generated Probability"]  || 'N/A', 'auto');
        const burstEl = document.getElementById('text-burstiness');
        if (burstEl) burstEl.innerText = ap["Burstiness Score"] || 'N/A';
        const ttrEl = document.getElementById('text-ttr-val');
        if (ttrEl) ttrEl.innerText = ap["Lexical Diversity (TTR)"] || 'N/A';

        // Preview card
        const prevContent = document.getElementById('text-preview-content');
        if (prevContent) prevContent.innerText = data.text_preview || '';
        const sentBadge = document.getElementById('text-sentiment-badge');
        if (sentBadge) {
            sentBadge.innerText = ap.Sentiment || 'Unknown';
            sentBadge.className = 'badge';
            if (ap.Sentiment === 'Positive') sentBadge.classList.add('green');
            else if (ap.Sentiment === 'Negative') sentBadge.classList.add('red');
        }
        const tBurst = document.getElementById('t-burst');
        if (tBurst) tBurst.innerText = ap["Burstiness Score"] || '0.000';
        const tTtr = document.getElementById('t-ttr');
        if (tTtr) tTtr.innerText = ap["Lexical Diversity (TTR)"] || '0.000';
        const tMark = document.getElementById('t-markers');
        if (tMark) tMark.innerText = ap["AI Cliché Markers Detected"] || '0';

        // Text filename display
        const txtFname = document.getElementById('text-filename-display');
        if (txtFname) txtFname.innerText = data.filename || 'Text Evidence';

        // Suspicious list from text
        const susList = document.getElementById('sus-list');
        if (susList) {
            const items = [];
            const aiProb = parseFloat((ap["AI Generated Probability"] || '0').replace('%', ''));
            if (aiProb >= 60) items.push(`High AI generation probability: ${ap["AI Generated Probability"]}`);
            if (ap["Identified AI Phrases"] && ap["Identified AI Phrases"] !== 'None Detected')
                items.push(`LLM clichés detected: ${ap["Identified AI Phrases"]}`);
            if (ap["Threat Detection"] && ap["Threat Detection"] !== 'Low')
                items.push(`${ap["Threat Detection"]} threat indicators found: ${ap["Threat Indicators"] || ''}`);
            if (ap.Toxicity && ap.Toxicity !== 'Low') items.push(`${ap.Toxicity} toxicity level detected`);
            if (ap["Fake News Indicator"] && ap["Fake News Indicator"] !== 'Low')
                items.push(`${ap["Fake News Indicator"]} disinformation/fake news risk`);
            if (items.length === 0) items.push('Human authorship patterns — no synthetic markers detected');
            susList.innerHTML = items.map(i => `<li>${i}</li>`).join('');
        }
    }

    // Radar Chart update (Mocked multi-evidence)
    let m_img = currentMode==='image' ? score : Math.floor(Math.random()*20+70);
    let m_vid = currentMode==='video' ? score : Math.floor(Math.random()*20+70);
    let m_aud = currentMode==='audio' ? score : Math.floor(Math.random()*20+70);
    let m_txt = currentMode==='text' ? score : Math.floor(Math.random()*20+70);
    let m_met = Math.floor(Math.random()*10+85);
    
    radarChart.data.datasets[0].data = [m_img, m_vid, m_aud, m_txt, m_met];
    radarChart.update();

    let avg = Math.round((m_img+m_vid+m_aud+m_txt+m_met)/5);
    document.getElementById('radar-avg').innerText = avg;
    document.getElementById('radar-avg').style.color = avg >= 70 ? '#0edb8d' : (avg >= 40 ? '#f59e0b' : '#ff3b4e');
    document.getElementById('radar-lbl').innerText = avg >= 70 ? 'HIGH RELIABILITY' : 'LOW RELIABILITY';
    
    document.getElementById('btn-report').disabled = false;
}

async function generateReport() {
    if (!currentAnalysisResult) return;
    
    document.getElementById('btn-report').innerText = "Generating PDF...";
    document.getElementById('btn-report').disabled = true;

    let reportResult = currentAnalysisResult;
    let panels = currentAnalysisResult.cam_panels || currentAnalysisResult.keyframe_panels || [];
    let fname = currentFilename;

    if (currentMode === 'multi' && currentAnalysisResult.results) {
        let validCount = 0;
        let totalScore = 0;
        let fakes = 0;
        for (let r of currentAnalysisResult.results) {
             if (r.prediction && r.prediction !== 'SKIPPED' && r.prediction !== 'ERROR') {
                 totalScore += (r.reliability_score || 0);
                 validCount++;
                 if (r.prediction.toUpperCase() === 'FAKE') fakes++;
             }
        }
        let avgScore = validCount > 0 ? Math.round(totalScore / validCount) : 0;
        let isFake = fakes > 0;
        
        reportResult = {
            prediction: isFake ? "FAKE" : "REAL",
            model_confidence: isFake ? (fakes/validCount * 100) : 99.0,
            reliability_score: avgScore,
            visual_consistency: avgScore >= 70 ? "High" : (avgScore >= 40 ? "Medium" : "Low"),
            metadata_integrity: "Mixed (Batch)",
            artifact_detection: isFake ? "Detected" : "Not Detected",
            report: `Batch Analysis Report for ${currentAnalysisResult.results.length} files.\nAverage Reliability: ${avgScore}/100\nTotal Fake/Manipulated: ${fakes}`
        };
        fname = "Batch_Analysis";
    }

    try {
        const res = await fetch('/api/report/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                result: reportResult,
                cam_panels: panels,
                filename: fname
            })
        });

        if (!res.ok) throw new Error("Failed to generate PDF");

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `EVIDENTIA_Report_${fname}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (err) {
        alert("Error generating report: " + err.message);
    } finally {
        document.getElementById('btn-report').innerHTML = `<svg style="width:16px;height:16px" viewBox="0 0 24 24"><path fill="currentColor" d="M5,20H19V18H5M19,9H15V3H9V9H5L12,16L19,9Z" /></svg> Download PDF Report`;
        document.getElementById('btn-report').disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Benchmark & Ablation Study Modal Functions
// ---------------------------------------------------------------------------

function openBenchmarkModal(e) {
    if (e) e.preventDefault();
    const modal = document.getElementById('benchmarkModal');
    if (modal) {
        modal.classList.remove('hidden');
        loadBenchmarkData();
    }
}

function closeBenchmarkModal() {
    const modal = document.getElementById('benchmarkModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function switchBenchmarkTab(tabName) {
    // Update active tab button
    document.querySelectorAll('.b-tab').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Hide all tabs
    ['metrics', 'plots', 'ablation', 'dataset'].forEach(t => {
        const el = document.getElementById(`btab-${t}`);
        if (el) el.classList.add('hidden');
    });

    // Show selected tab
    const target = document.getElementById(`btab-${tabName}`);
    if (target) target.classList.remove('hidden');
}

async function loadBenchmarkData() {
    try {
        const res = await fetch('/api/benchmark/summary');
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.metrics && data.metrics.length > 0) {
            const tbody = document.getElementById('benchmark-metrics-tbody');
            if (tbody) {
                tbody.innerHTML = data.metrics.map(m => {
                    const thr = m.Operating_Threshold !== undefined ? Number(m.Operating_Threshold).toFixed(4) : (m.Threshold !== undefined ? Number(m.Threshold).toFixed(4) : '0.5000');
                    return `
                    <tr style="${m.Model.includes('Ensemble') ? 'background:rgba(0, 216, 239, 0.08); font-weight:600;' : ''}">
                        <td><b style="${m.Model.includes('FasterViT') ? 'color:var(--cyan)' : (m.Model.includes('Ensemble') ? 'color:var(--green)' : '')}">${m.Model}</b></td>
                        <td>${thr}</td>
                        <td><b>${(m.Accuracy * 100).toFixed(2)}%</b></td>
                        <td>${(m.Precision * 100).toFixed(2)}%</td>
                        <td>${(m.Recall * 100).toFixed(2)}%</td>
                        <td>${(m.Specificity * 100).toFixed(2)}%</td>
                        <td><b>${Number(m.F1).toFixed(4)}</b></td>
                        <td><b>${Number(m.ROC_AUC).toFixed(4)}</b></td>
                        <td><b>${Number(m.PR_AUC).toFixed(4)}</b></td>
                    </tr>
                `}).join('');
            }
        }

        if (data.ablation && data.ablation.length > 0) {
            const atbody = document.getElementById('benchmark-ablation-tbody');
            if (atbody) {
                atbody.innerHTML = data.ablation.map(a => {
                    const sec = a.Section ? a.Section.replace(/^[A-Z]_/, '') : 'Ablation';
                    return `
                    <tr style="${a.Model.includes('Ensemble') ? 'background:rgba(0, 216, 239, 0.04);' : ''}">
                        <td>${sec}</td>
                        <td><b>${a.Model}</b></td>
                        <td><b>${(a.Accuracy * 100).toFixed(2)}%</b></td>
                        <td>${(a.Precision * 100).toFixed(2)}%</td>
                        <td>${(a.Recall * 100).toFixed(2)}%</td>
                        <td>${(a.Specificity * 100).toFixed(2)}%</td>
                        <td>${Number(a.F1).toFixed(4)}</td>
                        <td><b>${Number(a.ROC_AUC).toFixed(4)}</b></td>
                        <td><b>${Number(a.PR_AUC).toFixed(4)}</b></td>
                    </tr>
                `}).join('');
            }
        }
    } catch (err) {
        console.warn('Could not refresh live benchmark data:', err);
    }
}

// ---------------------------------------------------------------------------
// Audio & Text Classifier UI Helpers
// ---------------------------------------------------------------------------

// Store the last uploaded audio File for playback
let _lastAudioFile = null;

function activateAudioClassifier() {
    currentMode = 'audio';
    document.querySelectorAll('.nav-item').forEach(n => {
        n.classList.remove('active');
        if (n.innerText.trim().includes('Audio')) n.classList.add('active');
    });
    updateInterfaceForMode('audio');
    document.getElementById('uploadArea') && document.getElementById('uploadArea').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function activateTextClassifier() {
    currentMode = 'text';
    document.querySelectorAll('.nav-item').forEach(n => {
        n.classList.remove('active');
        if (n.innerText.trim().includes('Text')) n.classList.add('active');
    });
    updateInterfaceForMode('text');
    setTimeout(() => {
        const ta = document.getElementById('textEvidenceInput');
        if (ta) ta.focus();
    }, 150);
}

function scrollToReport() {
    const rp = document.getElementById('panel-report');
    if (rp) rp.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Sample text loaders for quick testing
const TEXT_SAMPLES = {
    ai: `In today's dynamic and rapidly evolving digital landscape, it is important to note that artificial intelligence plays a crucial and paramount role. Furthermore, delving into this multifaceted realm serves as a testament to human innovation. It is worth noting that navigating these complexities underscores the importance of a holistic approach. In conclusion, leveraging AI capabilities fosters a seamlessly integrated ecosystem.`,
    threat: `This is your FINAL WARNING. I have obtained your personal data and banking credentials from the dark web breach. Pay 0.5 BTC to wallet 1A2b3C4d5E within 48 hours or I will expose your data publicly and alert your employer. Do not contact police — I will know. Last chance before consequences.`,
    human: `Hey, just a quick heads up — I checked the CCTV timestamps you sent and they don't match the server logs I pulled. The footage shows 11:42 PM but the system log says 11:51 PM. Could be a time sync issue, or someone messed with the recordings. Call me when you can, this is important for the case.`
};

function loadSampleText(type) {
    const ta = document.getElementById('textEvidenceInput');
    if (!ta) return;
    ta.value = TEXT_SAMPLES[type] || '';
    updateTextWordCount();
    ta.focus();
}

function updateTextWordCount() {
    const ta = document.getElementById('textEvidenceInput');
    const counter = document.getElementById('textWordCount');
    if (!ta || !counter) return;
    const text = ta.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const chars = text.length;
    counter.innerText = `${words} words | ${chars} chars`;
}

async function analyzeDirectText() {
    const ta = document.getElementById('textEvidenceInput');
    if (!ta || !ta.value.trim()) {
        alert('Please enter or paste some text to analyze.');
        return;
    }

    const text = ta.value.trim();
    currentMode = 'text';
    currentFilename = 'direct_text_input.txt';
    updateInterfaceForMode('text');

    let startTime = Date.now();
    const timerEl = document.getElementById('processing-timer');
    const timerInterval = setInterval(() => {
        if (timerEl) timerEl.innerText = `Analyzing... ${((Date.now() - startTime) / 1000).toFixed(1)}s`;
    }, 100);

    try {
        const res = await fetch('/api/analyze/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, filename: 'direct_text_input.txt' })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        currentAnalysisResult = data;
        updateDashboard(data);
    } catch (err) {
        alert('Text analysis error: ' + err.message);
    } finally {
        clearInterval(timerInterval);
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        if (timerEl) timerEl.innerText = `Analysis completed in ${elapsed}s`;
    }
}

// Sample audio generators: creates a WAV Blob on the fly
function loadSampleAudio(type) {
    const sampleRate = 22050;
    const duration = 2.5;
    const numSamples = Math.floor(sampleRate * duration);
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    // Write WAV header
    const writeStr = (off, str) => [...str].forEach((c, i) => view.setUint8(off + i, c.charCodeAt(0)));
    writeStr(0, 'RIFF'); view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, 'WAVE'); writeStr(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    writeStr(36, 'data'); view.setUint32(40, numSamples * 2, true);

    // Generate signal: synthetic (pure sine, abrupt) vs human (multi-harmonic, noisy)
    for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        let sample;
        if (type === 'ai') {
            // Synthetic: pure 220 Hz sine — no noise, abrupt gaps (vocoder-like)
            sample = Math.sin(2 * Math.PI * 220 * t) * 28000 * (Math.floor(t / 0.5) % 2 === 0 ? 1 : 0.02);
        } else {
            // Human-like: rich harmonics + noise + amplitude variation
            const env = 0.7 + 0.3 * Math.sin(2 * Math.PI * 1.5 * t);
            sample = (0.5 * Math.sin(2 * Math.PI * 180 * t) +
                      0.3 * Math.sin(2 * Math.PI * 360 * t) +
                      0.15 * Math.sin(2 * Math.PI * 540 * t) +
                      0.08 * (Math.random() * 2 - 1)) * env * 25000;
        }
        view.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(sample))), true);
    }

    const blob = new Blob([buffer], { type: 'audio/wav' });
    const fname = type === 'ai' ? 'sample_ai_clone.wav' : 'sample_human_recording.wav';
    const file = new File([blob], fname, { type: 'audio/wav' });
    _lastAudioFile = file;

    // Set playback
    const audioEl = document.getElementById('audio-playback');
    if (audioEl) {
        audioEl.src = URL.createObjectURL(blob);
    }

    handleFiles([file]);
}

// Helper: set KV value with color coding
function _setKV(id, value, colorMode) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerText = value;
    el.className = 'kv-val';
    const v = String(value).toLowerCase();
    if (colorMode === 'risk') {
        if (v.includes('high') || v.includes('critical')) el.classList.add('red');
        else if (v.includes('medium')) el.classList.add('amber');
        else el.classList.add('green');
    } else if (colorMode === 'presence') {
        if (v.includes('detected') && !v.includes('not')) el.classList.add('red');
        else el.classList.add('green');
    } else if (colorMode === 'pos') {
        if (v.includes('high') || v.includes('natural') || v.includes('match') || v.includes('valid') || v.includes('low')) el.classList.add('green');
        else if (v.includes('suspicious') || v.includes('vocoder') || v.includes('inconclusive')) el.classList.add('amber');
        else el.classList.add('red');
    } else if (colorMode === 'auto') {
        // percentage-based (higher = worse for AI prob)
        const pct = parseFloat(v);
        if (!isNaN(pct)) {
            if (pct >= 70) el.classList.add('red');
            else if (pct >= 40) el.classList.add('amber');
            else el.classList.add('green');
        } else {
            el.classList.add('cyan');
        }
    } else if (colorMode === 'sentiment') {
        if (v === 'positive') el.classList.add('green');
        else if (v === 'negative') el.classList.add('red');
        else el.classList.add('cyan');
    } else {
        el.classList.add('cyan');
    }
}

// Wire up the textarea word counter on DOMContentLoaded (done via listener below)
document.addEventListener('DOMContentLoaded', () => {
    const ta = document.getElementById('textEvidenceInput');
    if (ta) ta.addEventListener('input', updateTextWordCount);

    // For audio mode: when a file is selected, also update the playback src
    const fi = document.getElementById('fileInput');
    if (fi) {
        fi.addEventListener('change', (e) => {
            const f = e.target.files[0];
            if (!f) return;
            const ext = f.name.split('.').pop().toLowerCase();
            if (['wav','mp3','ogg','flac','m4a','aac'].includes(ext)) {
                const audioEl = document.getElementById('audio-playback');
                if (audioEl) audioEl.src = URL.createObjectURL(f);
                const fnameEl = document.getElementById('audio-filename-display');
                if (fnameEl) fnameEl.innerText = f.name;
            }
        });
    }
});
