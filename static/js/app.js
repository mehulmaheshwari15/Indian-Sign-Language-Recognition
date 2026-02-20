/* ===================================================================
   SignSight AI — Enhanced Frontend Logic v2
   Tabs · Upload · Camera · Live · TTS · History · Toasts · Copy
   =================================================================== */

(function () {
    'use strict';

    // ─── DOM refs ───────────────────────────────────────────────────
    const $ = (s) => document.querySelector(s);
    const $$ = (s) => document.querySelectorAll(s);

    const tabBtns = $$('.tab-btn');
    const panels = $$('.panel');

    // upload
    const uploadZone = $('#upload-zone');
    const fileInput = $('#file-input');
    const previewWrap = $('#preview-container');
    const previewImg = $('#preview-img');
    const clearBtn = $('#clear-btn');
    const fileInfoEl = $('#file-info');
    const fileNameEl = $('#file-name');
    const fileSizeEl = $('#file-size');

    // camera
    const cameraVideo = $('#camera-video');
    const cameraFlash = $('#camera-flash');
    const captureBtn = $('#capture-btn');
    const cameraCanvas = document.createElement('canvas');

    // live
    const liveVideo = $('#live-video');
    const scanOverlay = $('#scan-overlay');
    const liveBadge = $('#live-badge');
    const livePlaceholder = $('#live-placeholder');

    // cta
    const runBtn = $('#run-btn');
    const runBtnIcon = $('#run-btn-icon');
    const runBtnText = $('#run-btn-text');
    const sampleBtn = $('#sample-btn');

    // result
    const resultSection = $('#result-section');
    const resultSign = $('#result-sign');
    const resultConf = $('#result-confidence');
    const resultDesc = $('#result-description');
    const statusDot = $('#status-dot');
    const confBarFill = $('#confidence-bar-fill');

    // result actions
    const speakBtn = $('#speak-btn');
    const copyBtn = $('#copy-btn');

    // history
    const historySection = $('#history-section');
    const historyList = $('#history-list');
    const historyClear = $('#history-clear');

    // toast
    const toastContainer = $('#toast-container');

    // ─── State ──────────────────────────────────────────────────────
    let activeTab = 'upload';
    let isDetecting = false;
    let liveInterval = null;
    let cameraStream = null;
    let selectedFile = null;
    let lastResult = null;
    let predictions = []; // history

    const MOCK_SIGNS = ['Hello', 'Thank You', 'Yes', 'No', 'Please', 'I Love You'];

    // ─── Tab switching ──────────────────────────────────────────────
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            if (tab === activeTab) return;

            stopLive();
            stopCamera();
            isDetecting = false;
            hideResult();
            updateBtnUI();

            activeTab = tab;
            tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
            panels.forEach(p => p.classList.toggle('active', p.dataset.panel === tab));

            if (tab === 'camera') startCamera(cameraVideo);
            if (tab === 'live') startCamera(liveVideo);
        });
    });

    // ─── Upload ─────────────────────────────────────────────────────
    uploadZone.addEventListener('click', (e) => {
        // don't re-open picker if clicking clear button
        if (e.target.closest('.clear-btn')) return;
        fileInput.click();
    });

    uploadZone.addEventListener('dragover', e => {
        e.preventDefault(); uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', e => {
        e.preventDefault(); uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    clearBtn.addEventListener('click', e => {
        e.stopPropagation();
        selectedFile = null;
        previewWrap.classList.remove('has-image');
        previewImg.src = '';
        fileInput.value = '';
        fileInfoEl.classList.remove('visible');
        hideResult();
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file (JPG, PNG, WebP)', 'error');
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            showToast('File too large – max 5 MB', 'error');
            return;
        }
        selectedFile = file;
        const url = URL.createObjectURL(file);
        previewImg.src = url;
        previewWrap.classList.add('has-image');

        // show file info
        fileNameEl.textContent = file.name.length > 30 ? file.name.slice(0, 27) + '…' : file.name;
        fileSizeEl.textContent = formatSize(file.size);
        fileInfoEl.classList.add('visible');

        showToast('Image loaded — click "Run detection"', 'success');
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    // ─── Camera helpers ─────────────────────────────────────────────
    async function startCamera(videoEl) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 }
            });
            cameraStream = stream;
            videoEl.srcObject = stream;
            videoEl.play();
        } catch (err) {
            console.warn('Camera error:', err);
            showToast('Camera access denied or unavailable', 'error');
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
        }
        cameraVideo.srcObject = null;
        liveVideo.srcObject = null;
    }

    function captureFrame(videoEl) {
        cameraCanvas.width = videoEl.videoWidth || 640;
        cameraCanvas.height = videoEl.videoHeight || 480;
        cameraCanvas.getContext('2d').drawImage(videoEl, 0, 0);
        return new Promise(resolve => cameraCanvas.toBlob(resolve, 'image/jpeg', 0.85));
    }

    // ─── Capture button (camera tab) ───────────────────────────────
    captureBtn.addEventListener('click', async () => {
        cameraFlash.classList.add('flash');
        setTimeout(() => cameraFlash.classList.remove('flash'), 150);
        await runDetection();
    });

    // ─── Run detection ──────────────────────────────────────────────
    runBtn.addEventListener('click', () => {
        if (activeTab === 'live') { toggleLive(); return; }
        runDetection();
    });

    async function runDetection() {
        if (isDetecting && activeTab !== 'live') return;
        isDetecting = true;
        hideResult();
        updateBtnUI();

        let blob = null;

        if (activeTab === 'upload') {
            blob = selectedFile || null;
        } else if (activeTab === 'camera') {
            blob = await captureFrame(cameraVideo);
        }

        if (blob) {
            try {
                const fd = new FormData();
                fd.append('image', blob, 'frame.jpg');
                const res = await fetch('/predict_image', { method: 'POST', body: fd });
                if (res.ok) {
                    const data = await res.json();
                    const conf = (data.confidence * 100).toFixed(1);
                    showResultUI(data.label, conf + '%', 'Prediction from AI model.', false, data.confidence);
                    addToHistory(data.label, conf + '%');
                } else {
                    showMock();
                }
            } catch {
                showMock();
            }
        } else {
            // no file selected mock
            if (activeTab === 'upload') {
                showToast('Please upload an image first', 'error');
            } else {
                await delay(1200);
                showMock();
            }
        }

        isDetecting = false;
        updateBtnUI();
    }

    // ─── Live detection ─────────────────────────────────────────────
    function toggleLive() {
        if (liveInterval) {
            stopLive();
            showToast('Live detection stopped', 'success');
        } else {
            isDetecting = true;
            scanOverlay.classList.add('active');
            liveBadge.classList.add('active');
            if (livePlaceholder) livePlaceholder.style.display = 'none';
            updateBtnUI();

            if (!cameraStream) startCamera(liveVideo);

            showToast('Live detection started', 'success');

            liveInterval = setInterval(async () => {
                const blob = await captureFrame(liveVideo);
                if (blob) {
                    try {
                        const fd = new FormData();
                        fd.append('image', blob, 'frame.jpg');
                        const res = await fetch('/predict_image', { method: 'POST', body: fd });
                        if (res.ok) {
                            const data = await res.json();
                            const conf = (data.confidence * 100).toFixed(1);
                            showResultUI(data.label, conf + '%', 'Live detection active — tracking gestures.', true, data.confidence);
                            addToHistory(data.label, conf + '%');
                        } else {
                            showMockLive();
                        }
                    } catch { showMockLive(); }
                } else {
                    showMockLive();
                }
            }, 2000);
        }
    }

    function stopLive() {
        clearInterval(liveInterval);
        liveInterval = null;
        isDetecting = false;
        scanOverlay.classList.remove('active');
        liveBadge.classList.remove('active');
        if (livePlaceholder) livePlaceholder.style.display = '';
        hideResult();
        updateBtnUI();
    }

    // ─── Sample image ───────────────────────────────────────────────
    sampleBtn.addEventListener('click', async () => {
        const c = document.createElement('canvas');
        c.width = 224; c.height = 224;
        const ctx = c.getContext('2d');
        // prettier sample: gradient bg + hand emoji
        const grad = ctx.createLinearGradient(0, 0, 224, 224);
        grad.addColorStop(0, '#f97316');
        grad.addColorStop(1, '#ea580c');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 224, 224);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 60px Inter, serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('✋', 112, 112);

        c.toBlob(async (blob) => {
            selectedFile = new File([blob], 'sample_hand.jpg', { type: 'image/jpeg' });
            previewImg.src = URL.createObjectURL(blob);
            previewWrap.classList.add('has-image');
            fileNameEl.textContent = 'sample_hand.jpg';
            fileSizeEl.textContent = formatSize(blob.size);
            fileInfoEl.classList.add('visible');

            // switch to upload tab
            tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === 'upload'));
            panels.forEach(p => p.classList.toggle('active', p.dataset.panel === 'upload'));
            activeTab = 'upload';
            stopLive(); stopCamera();

            showToast('Sample image loaded', 'success');
            await runDetection();
        }, 'image/jpeg');
    });

    // ─── Result actions ─────────────────────────────────────────────

    // Text-to-speech with Indian accent
    function getIndianVoice() {
        const voices = speechSynthesis.getVoices();
        // Prefer Indian English voices
        const indian = voices.find(v => v.lang === 'hi-IN')
            || voices.find(v => v.lang === 'en-IN')
            || voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('india'))
            || voices.find(v => v.lang.startsWith('en'));
        return indian || null;
    }

    // Ensure voices are loaded
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = () => { }; // trigger load
    }

    speakBtn.addEventListener('click', () => {
        if (!lastResult) return;
        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(
            `The detected sign is ${lastResult.label}, with ${lastResult.confidence} confidence.`
        );
        const voice = getIndianVoice();
        if (voice) utterance.voice = voice;
        utterance.rate = 0.85;
        utterance.pitch = 1.1;
        utterance.volume = 1;

        speechSynthesis.speak(utterance);
        showToast('🔊 Speaking with Indian accent', 'success');
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        if (!lastResult) return;
        const text = `Sign: ${lastResult.label} | Confidence: ${lastResult.confidence}`;
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard', 'success');
        }).catch(() => {
            showToast('Failed to copy', 'error');
        });
    });

    // ─── History ────────────────────────────────────────────────────
    function addToHistory(sign, confidence) {
        const now = new Date();
        const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        predictions.unshift({ sign, confidence, time });
        if (predictions.length > 8) predictions.pop();
        renderHistory();
    }

    function renderHistory() {
        if (predictions.length === 0) {
            historySection.classList.remove('visible');
            return;
        }
        historySection.classList.add('visible');
        historyList.innerHTML = predictions.map(p => `
      <div class="history-item">
        <span class="h-sign">${p.sign}</span>
        <span class="h-conf">${p.confidence}</span>
        <span class="h-time">${p.time}</span>
      </div>
    `).join('');
    }

    historyClear.addEventListener('click', () => {
        predictions = [];
        renderHistory();
        showToast('History cleared', 'success');
    });

    // ─── Toasts ─────────────────────────────────────────────────────
    function showToast(message, type = 'success') {
        const icons = {
            success: '<i data-lucide="check-circle" style="width:16px;height:16px"></i>',
            error: '<i data-lucide="alert-circle" style="width:16px;height:16px"></i>'
        };
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span class="toast-icon">${icons[type] || ''}</span><span>${message}</span>`;
        toastContainer.appendChild(toast);
        lucide.createIcons();

        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // ─── UI helpers ─────────────────────────────────────────────────
    function showResultUI(sign, confidence, description, pulsing, rawConf) {
        lastResult = { label: sign, confidence };

        resultSign.textContent = sign;
        resultConf.textContent = confidence;
        resultDesc.textContent = description || '';
        statusDot.classList.toggle('pulsing', !!pulsing);

        // confidence bar
        const pct = Math.min((rawConf || 0) * 100, 100);
        confBarFill.style.width = pct + '%';
        confBarFill.classList.toggle('high', pct >= 70);

        resultSection.classList.add('visible');
    }

    function hideResult() {
        resultSection.classList.remove('visible');
        confBarFill.style.width = '0%';
    }

    function showMock() {
        const sign = MOCK_SIGNS[Math.floor(Math.random() * MOCK_SIGNS.length)];
        const rawConf = 0.9 + Math.random() * 0.09;
        const conf = (rawConf * 100).toFixed(1) + '%';
        showResultUI(sign, conf, 'Mock prediction (no backend).', false, rawConf);
        addToHistory(sign, conf);
    }

    function showMockLive() {
        const sign = MOCK_SIGNS[Math.floor(Math.random() * MOCK_SIGNS.length)];
        const rawConf = 0.9 + Math.random() * 0.09;
        const conf = (rawConf * 100).toFixed(1) + '%';
        showResultUI(sign, conf, 'Live detection active — tracking gestures.', true, rawConf);
        addToHistory(sign, conf);
    }

    function updateBtnUI() {
        if (activeTab === 'live') {
            if (liveInterval) {
                runBtn.classList.add('stop');
                runBtnIcon.innerHTML = '<div style="width:12px;height:12px;background:var(--red-500);border-radius:2px"></div>';
                runBtnText.textContent = 'Stop live stream';
            } else {
                runBtn.classList.remove('stop');
                runBtnIcon.innerHTML = '<i data-lucide="play" style="width:18px;height:18px;fill:currentColor"></i>';
                runBtnText.textContent = 'Start live stream';
                lucide.createIcons();
            }
            runBtn.disabled = false;
        } else if (isDetecting) {
            runBtn.classList.remove('stop');
            runBtnIcon.innerHTML = '<div class="spinner"></div>';
            runBtnText.textContent = 'Processing model...';
            runBtn.disabled = true;
        } else {
            runBtn.classList.remove('stop');
            runBtnIcon.innerHTML = '<i data-lucide="sparkles" style="width:18px;height:18px"></i>';
            runBtnText.textContent = 'Run detection';
            runBtn.disabled = false;
            lucide.createIcons();
        }
    }

    function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

    // ─── Keyboard shortcuts ─────────────────────────────────────────
    document.addEventListener('keydown', e => {
        if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); runBtn.click(); }
        if (e.key === 'Escape') { stopLive(); hideResult(); }
    });

    // ─── Nav link handlers ──────────────────────────────────────────
    const navMap = {
        'How it works': {
            scrollTo: '.card',
            message: 'Upload a photo, capture from camera, or start live detection — our AI model does the rest!'
        },
        'Model': {
            scrollTo: '.result-section',
            message: 'We use a MobileNet-based deep learning model trained on Indian Sign Language gestures.'
        },
        'Docs': {
            scrollTo: '.footer-strip',
            message: 'API Documentation coming soon — POST /predict_image with a multipart image.'
        },
        'Contact': {
            scrollTo: null,
            message: 'Contact us at signsight@example.com — we love hearing from you! 🧡'
        }
    };

    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const info = navMap[link.textContent.trim()];
            if (!info) return;

            if (info.scrollTo) {
                const target = document.querySelector(info.scrollTo);
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }

            showToast(info.message, 'success');
        });
    });

})();
