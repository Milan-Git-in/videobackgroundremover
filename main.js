import * as THREE from 'three';
import { removeBackground } from '@imgly/background-removal';

// ============================================================================
// V8 HIGH-PERFORMANCE BACKGROUND REMOVAL ENGINE
// WebGPU + Parallel Pipeline + Alpha Confidence Reinforcement
// ============================================================================

// --- SHADERS ---
const vertexShader = `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const fragmentShader = `
    uniform sampler2D uTexture;
    uniform sampler2D uMask;
    uniform sampler2D uRestoreMask;
    uniform sampler2D uEraseMask;
    uniform bool uShowMask;
    uniform bool uOriginalBg;
    varying vec2 vUv;

    void main() {
        vec4 color = texture2D(uTexture, vUv);
        
        if (uOriginalBg) {
            gl_FragColor = color;
            return;
        }

        float rawMaskAlpha = texture2D(uMask, vUv).a;
        float userAdd = texture2D(uRestoreMask, vUv).r;
        float userSub = texture2D(uEraseMask, vUv).r;

        // V8: Preserve weak interior regions — do NOT binarize or discard low alpha.
        // True-zero background stays zero. Any non-zero value is treated as foreground.
        // Apply a gentle smoothstep that lifts the low end without changing edges.
        float maskAlpha = rawMaskAlpha > 0.0
            ? smoothstep(0.05, 1.0, rawMaskAlpha)  // Lift weak values, preserve full interior
            : 0.0;                                   // True background — keep transparent

        float alpha = clamp(maskAlpha + userAdd - userSub, 0.0, 1.0);
        
        if (uShowMask) {
            if (alpha > 0.3) gl_FragColor = vec4(1.0, 0.0, 0.0, 0.8);
            else gl_FragColor = vec4(color.rgb * 0.3, 1.0);
        } else {
            // Straight alpha compositing — color channels unmodified, only alpha drives transparency
            gl_FragColor = vec4(color.rgb, alpha);
        }
    }
`;

// --- GLOBAL STATE ---
let scene, camera, renderer, plane, material, bgMesh;
let video, videoTexture;
let isVideo = false;
let restoreCanvas, restoreCtx, restoreMaskTexture;
let eraseCanvas, eraseCtx, eraseMaskTexture;
let brushCursorEl;
let brushModeRadios;
let brushSoftnessInput, brushStrengthInput;
let currentBrushMode = 'restore';
let isBrushAltPressed = false;
let isRendering = false;
let mediaRecorder;
let recordedChunks = [];

// Mask storage — Uint8Array alpha buffers
let frameMasks = [];       // Array of { time, alpha, width, height, coverage, source } for video
let imageMaskCanvas = null;
let totalFrames = 0;
let currentMaskTexture = null;

// V9: System readiness gate — blocks all user interaction until GPU + model ready
let isSystemReady = false;

// Worker pool
const WORKER_COUNT = 4;
let workers = [];
let workersReady = 0;
let gpuAvailable = false;
let workerInitFailed = 0;

// Performance tracking
let processingStartTime = 0;
let framesProcessed = 0;
let framesSkipped = 0;
let fastMode = false;

// V6: Debug mode
let debugMode = false;

// --- DOM REFS ---
const container = document.getElementById('canvas-container');
const videoInput = document.getElementById('video-upload');
const brushToggle = document.getElementById('brush-toggle');
const brushSizeInput = document.getElementById('brush-size');
brushSoftnessInput = document.getElementById('brush-softness');
brushStrengthInput = document.getElementById('brush-strength');
brushCursorEl = document.getElementById('brush-cursor');
brushModeRadios = document.getElementsByName('brush-mode');
const maskPreviewToggle = document.getElementById('mask-preview');
const bgToggle = document.getElementById('bg-toggle');
const clearMaskBtn = document.getElementById('clear-mask');
const playPauseBtn = document.getElementById('play-pause');
const renderBtn = document.getElementById('download');
const sampleBtn = document.getElementById('load-sample');
const progressContainer = document.getElementById('render-progress');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const statusText = document.getElementById('status');
const statusIcon = document.getElementById('status-icon');
const toastContainer = document.getElementById('toast-container');

const processingOverlay = document.getElementById('processing-overlay');
const processingTitle = document.getElementById('processing-title');
const processingStatus = document.getElementById('processing-status');
const processingFill = document.getElementById('processing-fill');
const processingPercent = document.getElementById('processing-percent');
const processingETA = document.getElementById('processing-eta');
const processingSpeed = document.getElementById('processing-speed');
const processingWorkers = document.getElementById('processing-workers');
const processingGPU = document.getElementById('processing-gpu');
const gpuBadge = document.getElementById('gpu-badge');
const debugToggle = document.getElementById('debug-toggle');
const debugPanel = document.getElementById('debug-panel');

const bgRadios = document.querySelectorAll('input[name="bg-type"]');
const bgColorPicker = document.getElementById('bg-color-picker');
const bgImagePicker = document.getElementById('bg-image-picker');
const bgColorInput = document.getElementById('bg-color');
const bgImageUpload = document.getElementById('bg-image-upload');
let customBgTex = null;
let customBgVideo = null;

// ============================================================================
// WORKER POOL MANAGEMENT
// ============================================================================

function initWorkerPool() {
  console.log(`[V8 Engine] Initializing ${WORKER_COUNT} segmentation workers...`);

  // Update init gate status live
  const initStatus = document.getElementById('init-status');

  let workersResponded = 0; // Count ready + error responses

  function onWorkerResponded() {
    workersResponded++;
    if (initStatus) {
      initStatus.textContent = `Worker ${workersResponded}/${WORKER_COUNT} initialized...`;
    }
    if (workersResponded >= WORKER_COUNT) {
      updateGPUBadge();
      const mode = gpuAvailable ? 'WebGPU' : 'CPU (WASM)';
      console.log(`[V8 Engine] All workers responded. Mode: ${mode}. GPU: ${gpuAvailable}`);
      setSystemReady();
    }
  }

  for (let i = 0; i < WORKER_COUNT; i++) {
    const worker = new Worker(
      new URL('./workers/segmentationWorker.js', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (e) => {
      if (e.data.type === 'ready') {
        workersReady++;
        if (e.data.gpu) gpuAvailable = true;
        console.log(`[V8 Engine] Worker ${i} ready (GPU: ${e.data.gpu}). ${workersReady}/${WORKER_COUNT} online.`);
        onWorkerResponded();
      } else if (e.data.type === 'error') {
        console.warn(`[V8 Engine] Worker ${i} reported error (will use degraded mode):`, e.data.message);
        onWorkerResponded();
      }
    };

    worker.onerror = (err) => {
      console.warn(`[V8 Engine] Worker ${i} crashed, counting as responded:`, err.message);
      onWorkerResponded();
    };

    worker.postMessage({ type: 'init' });
    workers.push(worker);
  }
}


function updateGPUBadge() {
  if (gpuBadge) {
    gpuBadge.textContent = gpuAvailable ? '⚡ GPU' : '🔧 CPU';
    gpuBadge.className = gpuAvailable ? 'gpu-badge gpu-active' : 'gpu-badge cpu-active';
  }
}

function terminateWorkers() {
  workers.forEach(w => w.terminate());
  workers = [];
  workersReady = 0;
}

// ============================================================================
// V9: INITIALIZATION GATE — lock/unlock UI
// ============================================================================

function lockUI() {
  const uploadInput = document.getElementById('video-upload');
  const sampleButton = document.getElementById('load-sample');
  if (uploadInput) uploadInput.disabled = true;
  if (sampleButton) sampleButton.disabled = true;
  document.getElementById('app')?.classList.add('app-locked');
}

function unlockUI() {
  const uploadInput = document.getElementById('video-upload');
  const sampleButton = document.getElementById('load-sample');
  if (uploadInput) uploadInput.disabled = false;
  if (sampleButton) sampleButton.disabled = false;
  document.getElementById('app')?.classList.remove('app-locked');
}

function setSystemReady() {
  isSystemReady = true;
  // Dismiss init gate overlay
  const gate = document.getElementById('init-gate');
  if (gate) {
    gate.classList.add('init-gate-done');
    setTimeout(() => { gate.style.display = 'none'; }, 500);
  }
  unlockUI();
  updateStatus('System Ready', 'ready');
  console.log('[V8 Engine] System ready. UI unlocked.');
}

function showInitError(title, detail) {
  // Update init gate to show error, not dismiss
  const initTitle = document.getElementById('init-title');
  const initStatus = document.getElementById('init-status');
  const gate = document.getElementById('init-gate');
  const card = gate?.querySelector('.init-card');

  if (initTitle) initTitle.textContent = '⚠️ ' + title;
  if (initStatus) initStatus.textContent = detail;

  // Add reload button
  if (card && !card.querySelector('.init-reload-btn')) {
    const spinner = card.querySelector('.init-spinner');
    if (spinner) spinner.style.display = 'none';
    const btn = document.createElement('button');
    btn.className = 'btn primary init-reload-btn';
    btn.textContent = 'Reload Page';
    btn.onclick = () => location.reload();
    card.appendChild(btn);
  }

  if (gate) gate.classList.add('init-gate-error');
  console.error(`[V8 Engine] Init error: ${title} ${detail}`);
}

// ============================================================================
// PROCESSING OVERLAY HELPERS
// ============================================================================

function showProcessing(title, status) {
  processingOverlay.style.display = 'flex';
  processingTitle.textContent = title;
  processingStatus.textContent = status;
  processingFill.style.width = '0%';
  processingPercent.textContent = '0%';
  if (processingETA) processingETA.textContent = '--';
  if (processingSpeed) processingSpeed.textContent = '0 fps';
  if (processingWorkers) processingWorkers.textContent = `${WORKER_COUNT} workers`;
  if (processingGPU) processingGPU.textContent = gpuAvailable ? 'WebGPU' : 'CPU (WASM)';
  processingStartTime = performance.now();
  framesProcessed = 0;
  framesSkipped = 0;
  fastMode = false;
}

function updateProcessing(status, percent, completed, total) {
  processingStatus.textContent = status;
  processingFill.style.width = `${percent}%`;
  processingPercent.textContent = `${Math.round(percent)}%`;

  if (completed > 0 && processingStartTime) {
    const elapsed = (performance.now() - processingStartTime) / 1000;
    const speed = completed / elapsed;
    const remaining = total - completed;
    const eta = remaining / Math.max(speed, 0.01);

    if (processingSpeed) processingSpeed.textContent = `${speed.toFixed(1)} fps`;
    if (processingETA) {
      if (eta < 60) processingETA.textContent = `${Math.ceil(eta)}s`;
      else processingETA.textContent = `${Math.floor(eta / 60)}m ${Math.ceil(eta % 60)}s`;
    }

    if (elapsed > 30 && percent < 50 && !fastMode) {
      fastMode = true;
      showToast("⚡ Switching to Fast Mode — lower resolution for speed");
      const fastIndicator = document.getElementById('fast-mode-indicator');
      if (fastIndicator) fastIndicator.style.display = 'block';
    }
  }
}

function hideProcessing() {
  processingOverlay.style.display = 'none';
  const fastIndicator = document.getElementById('fast-mode-indicator');
  if (fastIndicator) fastIndicator.style.display = 'none';
}

// ============================================================================
// V6: EDGE-BASED FRAME SIMILARITY DETECTION
// ============================================================================

/**
 * Compute a simple edge magnitude map from a canvas.
 * Uses Sobel-like horizontal+vertical gradient on grayscale.
 * Returns a Float32Array of edge magnitudes.
 */
function computeEdgeMap(imageData, w, h) {
  const gray = new Float32Array(w * h);
  const data = imageData;
  
  // Convert to grayscale
  for (let i = 0; i < w * h; i++) {
    gray[i] = (data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114);
  }

  // Compute gradient magnitude (simple Sobel approximation)
  const edges = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      // Horizontal gradient
      const gx = gray[(y - 1) * w + (x + 1)] + 2 * gray[y * w + (x + 1)] + gray[(y + 1) * w + (x + 1)]
               - gray[(y - 1) * w + (x - 1)] - 2 * gray[y * w + (x - 1)] - gray[(y + 1) * w + (x - 1)];
      // Vertical gradient
      const gy = gray[(y + 1) * w + (x - 1)] + 2 * gray[(y + 1) * w + x] + gray[(y + 1) * w + (x + 1)]
               - gray[(y - 1) * w + (x - 1)] - 2 * gray[(y - 1) * w + x] - gray[(y - 1) * w + (x + 1)];
      edges[idx] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return edges;
}

/**
 * V6: Determine if a frame is "similar" to the previous one.
 * Uses BOTH pixel-level difference AND edge/structure difference.
 * Returns { similar: bool, pixelDiff: number, edgeDiff: number }
 */
function analyzeFrameSimilarity(currentData, prevData, prevEdges, w, h, frameIndex) {
  // Rule: Never skip first 3 frames — let model establish baseline
  if (frameIndex < 3) {
    return { similar: false, pixelDiff: 999, edgeDiff: 999 };
  }

  if (!prevData) {
    return { similar: false, pixelDiff: 999, edgeDiff: 999 };
  }

  // 1. Pixel difference (existing logic but sampling more densely)
  let totalPixelDiff = 0;
  const sampleStep = 8; // More dense sampling than before (was 16)
  let sampleCount = 0;
  for (let p = 0; p < currentData.length; p += 4 * sampleStep) {
    totalPixelDiff += Math.abs(currentData[p] - prevData[p]);
    totalPixelDiff += Math.abs(currentData[p + 1] - prevData[p + 1]);
    totalPixelDiff += Math.abs(currentData[p + 2] - prevData[p + 2]);
    sampleCount++;
  }
  const avgPixelDiff = totalPixelDiff / (sampleCount * 3);

  // 2. Edge/structure difference
  const currentEdges = computeEdgeMap(currentData, w, h);
  let totalEdgeDiff = 0;
  let edgeSamples = 0;
  const edgeStep = 4;
  for (let i = 0; i < currentEdges.length; i += edgeStep) {
    totalEdgeDiff += Math.abs(currentEdges[i] - prevEdges[i]);
    edgeSamples++;
  }
  const avgEdgeDiff = totalEdgeDiff / edgeSamples;

  // V6: Frame is similar ONLY if BOTH metrics are below thresholds
  // Pixel diff < 4 AND edge diff < 8 → truly stable frame
  const similar = avgPixelDiff < 4 && avgEdgeDiff < 8;

  return { similar, pixelDiff: avgPixelDiff, edgeDiff: avgEdgeDiff, edges: currentEdges };
}

// ============================================================================
// V6: MASK INTEGRITY VALIDATION
// ============================================================================

const MIN_COVERAGE = 0.005; // Reject masks with < 0.5% coverage
const MAX_SHRINK_RATIO = 0.3; // Reject if coverage drops to < 30% of previous

/**
 * Validate a mask result. Returns true if mask is acceptable.
 */
function validateMask(coverage, prevCoverage) {
  // Reject near-empty masks
  if (coverage < MIN_COVERAGE) {
    console.warn(`[V6] Mask rejected: coverage ${(coverage * 100).toFixed(1)}% < ${(MIN_COVERAGE * 100).toFixed(1)}% threshold`);
    return false;
  }

  // Reject sudden large shrinks (object disappearing)
  if (prevCoverage > MIN_COVERAGE && coverage < prevCoverage * MAX_SHRINK_RATIO) {
    console.warn(`[V6] Mask rejected: coverage ${(coverage * 100).toFixed(1)}% is < 30% of previous ${(prevCoverage * 100).toFixed(1)}%`);
    return false;
  }

  return true;
}

// ============================================================================
// V6: FLIP ALPHA ROWS (Fix DataTexture orientation)
// ============================================================================

/**
 * Flip an alpha buffer vertically (reverse row order).
 * DataTexture uses bottom-left origin. Canvas getImageData uses top-left origin.
 * We flip rows so the DataTexture displays correctly.
 */
function flipAlphaRows(alpha, width, height) {
  const flipped = new Uint8Array(alpha.length);
  for (let y = 0; y < height; y++) {
    const srcRow = y * width;
    const dstRow = (height - 1 - y) * width;
    flipped.set(alpha.subarray(srcRow, srcRow + width), dstRow);
  }
  return flipped;
}

// ============================================================================
// INIT
// ============================================================================

async function init() {
  console.log("[V8 Engine] Initializing Three.js + Worker Pool");
  updateStatus("Initializing...", "processing");

  scene = new THREE.Scene();

  const w = container.offsetWidth || 800;
  const h = container.offsetHeight || 450;
  camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
  camera.position.z = 2;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // Checkerboard Background
  const bgSize = 128;
  const bgCanvas = document.createElement('canvas');
  bgCanvas.width = bgSize; bgCanvas.height = bgSize;
  const bgCtx = bgCanvas.getContext('2d');
  bgCtx.fillStyle = '#111'; bgCtx.fillRect(0, 0, bgSize, bgSize);
  bgCtx.fillStyle = '#222';
  bgCtx.fillRect(0, 0, bgSize / 2, bgSize / 2);
  bgCtx.fillRect(bgSize / 2, bgSize / 2, bgSize / 2, bgSize / 2);
  const bgTex = new THREE.CanvasTexture(bgCanvas);
  bgTex.wrapS = THREE.RepeatWrapping; bgTex.wrapT = THREE.RepeatWrapping;
  bgTex.repeat.set(20, 20);

  bgMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshBasicMaterial({ map: bgTex, visible: false })
  );
  bgMesh.position.z = -1;
  scene.add(bgMesh);

  // V10: Dual User Mask Canvases (For Restore and Erase tools)
  restoreCanvas = document.createElement('canvas');
  restoreCanvas.width = 1024; restoreCanvas.height = 1024;
  restoreCtx = restoreCanvas.getContext('2d');
  restoreCtx.fillStyle = 'black'; restoreCtx.fillRect(0, 0, 1024, 1024);
  restoreMaskTexture = new THREE.CanvasTexture(restoreCanvas);

  eraseCanvas = document.createElement('canvas');
  eraseCanvas.width = 1024; eraseCanvas.height = 1024;
  eraseCtx = eraseCanvas.getContext('2d');
  eraseCtx.fillStyle = 'black'; eraseCtx.fillRect(0, 0, 1024, 1024);
  eraseMaskTexture = new THREE.CanvasTexture(eraseCanvas);

  // Initial dummy mask
  const dummyMaskCanvas = document.createElement('canvas');
  dummyMaskCanvas.width = 1; dummyMaskCanvas.height = 1;
  currentMaskTexture = new THREE.CanvasTexture(dummyMaskCanvas);

  material = new THREE.ShaderMaterial({
    uniforms: {
      uTexture: { value: new THREE.Texture() },
      uMask: { value: currentMaskTexture },
      uRestoreMask: { value: restoreMaskTexture },
      uEraseMask: { value: eraseMaskTexture },
      uShowMask: { value: false },
      uOriginalBg: { value: false }
    },
    vertexShader,
    fragmentShader,
    transparent: true,
  });

  plane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(plane);

  setupListeners();
  initWorkerPool();
  animate();
  updateStatus("System Ready", "ready");
}

// ============================================================================
// LISTENERS
// ============================================================================

function setupListeners() {
  window.addEventListener('resize', onWindowResize);

  bgRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      const val = e.target.value;
      bgColorPicker.style.display = val === 'color' ? 'block' : 'none';
      bgImagePicker.style.display = (val === 'image' || val === 'video') ? 'block' : 'none';
      updateBackgroundType(val);
    });
  });
  bgColorInput.addEventListener('input', () => updateBackgroundType('color'));
  bgImageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      document.getElementById('bg-image-label').textContent = file.name.substring(0, 15) + '...';
      const url = URL.createObjectURL(file);

      if (file.type.startsWith('video/')) {
        if (customBgVideo) { customBgVideo.src = ""; customBgVideo = null; }
        customBgVideo = document.createElement('video');
        customBgVideo.src = url;
        customBgVideo.loop = true;
        customBgVideo.muted = true;
        customBgVideo.play();
        customBgTex = new THREE.VideoTexture(customBgVideo);
        document.querySelector('input[name="bg-type"][value="video"]').checked = true;
        updateBackgroundType('video');
      } else {
        new THREE.TextureLoader().load(url, tex => {
          tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.RepeatWrapping;
          customBgTex = tex;
          document.querySelector('input[name="bg-type"][value="image"]').checked = true;
          updateBackgroundType('image');
        });
      }
    }
  });

  function updateCursorSize() {
    const size = parseInt(brushSizeInput.value) * 2; // diameter roughly matches radius logic
    brushCursorEl.style.width = `${size}px`;
    brushCursorEl.style.height = `${size}px`;
  }

  function updateBrushModeUI() {
    const effectiveMode = isBrushAltPressed ? 'erase' : currentBrushMode;
    if (effectiveMode === 'erase') {
        brushCursorEl.classList.remove('restore-mode');
        brushCursorEl.classList.add('erase-mode');
    } else {
        brushCursorEl.classList.remove('erase-mode');
        brushCursorEl.classList.add('restore-mode');
    }
  }

  brushToggle.addEventListener('change', () => {
    if (brushToggle.checked) {
      container.classList.add('brushing');
      brushCursorEl.style.display = 'flex';
      updateCursorSize();
      updateBrushModeUI();
    } else {
      container.classList.remove('brushing');
      brushCursorEl.style.display = 'none';
    }
  });

  brushModeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      currentBrushMode = e.target.value;
      updateBrushModeUI();
    });
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Alt') {
      isBrushAltPressed = true;
      e.preventDefault();
      updateBrushModeUI();
    }
  });

  document.addEventListener('keyup', (e) => {
    if (e.key === 'Alt') {
      isBrushAltPressed = false;
      updateBrushModeUI();
    }
  });

  // Keep cursor size up to date
  brushSizeInput.addEventListener('input', () => {
    document.getElementById('brush-size-value').textContent = `${brushSizeInput.value}px`;
    updateCursorSize();
  });
  brushSoftnessInput.addEventListener('input', () => {
    document.getElementById('brush-softness-value').textContent = `${brushSoftnessInput.value}%`;
  });
  brushStrengthInput.addEventListener('input', () => {
    document.getElementById('brush-strength-value').textContent = `${brushStrengthInput.value}%`;
  });

  maskPreviewToggle.addEventListener('change', () => material.uniforms.uShowMask.value = maskPreviewToggle.checked);
  bgToggle.addEventListener('change', () => bgMesh.material.visible = bgToggle.checked);

  videoInput.addEventListener('change', (e) => {
    if (!isSystemReady) return;
    const file = e.target.files[0];
    if (file) {
      hideLandingHero();
      const url = URL.createObjectURL(file);
      if (file.type.startsWith('image/')) {
        loadAndProcessImage(url);
      } else {
        loadAndProcessVideo(url);
      }
    }
  });

  playPauseBtn.addEventListener('click', () => {
    if (!video || isRendering || !isSystemReady) return;
    if (video.paused) video.play(); else video.pause();
    playPauseBtn.textContent = video.paused ? 'Play' : 'Pause';
  });
  renderBtn.addEventListener('click', () => {
    if (!isSystemReady) return;
    if (isRendering) stopRendering();
    else startRender();
  });
  sampleBtn.addEventListener('click', () => {
    if (!isSystemReady) return;
    hideLandingHero();
    loadAndProcessVideo('/eyes.webm');
  });
  
  clearMaskBtn.addEventListener('click', () => {
    restoreCtx.fillStyle = 'black';
    restoreCtx.fillRect(0, 0, 1024, 1024);
    restoreMaskTexture.needsUpdate = true;

    eraseCtx.fillStyle = 'black';
    eraseCtx.fillRect(0, 0, 1024, 1024);
    eraseMaskTexture.needsUpdate = true;
  });

  // V6: Debug toggle
  if (debugToggle) {
    debugToggle.addEventListener('change', () => {
      debugMode = debugToggle.checked;
      if (debugPanel) debugPanel.style.display = debugMode ? 'block' : 'none';
      console.log(`[V6] Debug mode: ${debugMode ? 'ON' : 'OFF'}`);
    });
  }

  // Canvas Brush Listeners
  const onMove = (e) => {
    if (!brushToggle.checked) return;

    // Track cursor visuals bounds check
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    brushCursorEl.style.left = `${clientX}px`;
    brushCursorEl.style.top = `${clientY}px`;

    // Only draw if actively clicking/touching
    if (e.buttons !== 1 && e.type !== 'touchstart' && e.type !== 'touchmove') return;

    const rect = renderer.domElement.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
    const intersects = raycaster.intersectObject(plane);
    
    if (intersects.length > 0) {
      const uv = intersects[0].uv;
      
      const size = parseInt(brushSizeInput.value);
      const softness = parseInt(brushSoftnessInput.value) / 100.0;
      const strength = parseInt(brushStrengthInput.value) / 100.0;
      
      const effectiveMode = isBrushAltPressed ? 'erase' : currentBrushMode;
      const primaryCtx = effectiveMode === 'restore' ? restoreCtx : eraseCtx;
      const secondaryCtx = effectiveMode === 'restore' ? eraseCtx : restoreCtx;

      const cx = uv.x * 1024;
      const cy = (1 - uv.y) * 1024;
      
      const grad = primaryCtx.createRadialGradient(cx, cy, 0, cx, cy, size);
      const innerCore = Math.max(0, 1.0 - softness);
      
      grad.addColorStop(0, `rgba(255, 255, 255, ${strength})`);
      if (innerCore < 1) grad.addColorStop(innerCore, `rgba(255, 255, 255, ${strength})`);
      grad.addColorStop(1, 'rgba(255, 255, 255, 0.0)');

      // Add stroke to active mode buffer
      primaryCtx.globalCompositeOperation = 'source-over';
      primaryCtx.fillStyle = grad;
      primaryCtx.beginPath();
      primaryCtx.arc(cx, cy, size, 0, Math.PI * 2);
      primaryCtx.fill();

      // Erase stroke simultaneously from opposing buffer
      secondaryCtx.globalCompositeOperation = 'destination-out';
      secondaryCtx.fillStyle = grad;
      secondaryCtx.beginPath();
      secondaryCtx.arc(cx, cy, size, 0, Math.PI * 2);
      secondaryCtx.fill();

      restoreMaskTexture.needsUpdate = true;
      eraseMaskTexture.needsUpdate = true;
    }
  };
  
  renderer.domElement.addEventListener('mousedown', onMove);
  renderer.domElement.addEventListener('mousemove', onMove);
  renderer.domElement.addEventListener('touchstart', onMove, {passive: true});
  renderer.domElement.addEventListener('touchmove', onMove, {passive: true});
  renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
}

function updateBackgroundType(type) {
  material.uniforms.uOriginalBg.value = (type === 'original');

  if (type === 'transparent' || type === 'original') {
    scene.background = null;
  } else if (type === 'color') {
    scene.background = new THREE.Color(bgColorInput.value);
  } else if (type === 'image' || type === 'video') {
    if (customBgTex) scene.background = customBgTex;
  }
}

// ============================================================================
// IMAGE PROCESSING (GPU-Accelerated)
// ============================================================================

async function loadAndProcessImage(url) {
  showProcessing("Processing Image", "Loading AI engine with GPU...");
  updateStatus("Processing...", "processing");
  isVideo = false;

  try {
    const response = await fetch(url);
    const inputBlob = await response.blob();

    updateProcessing("AI analyzing image (GPU)...", 30, 0, 1);

    let resultBlob;
    try {
      resultBlob = await removeBackground(inputBlob, {
        device: 'gpu',
        progress: (key, current, total) => {
          if (key === 'compute:inference') {
            const pct = 30 + (current / total) * 60;
            updateProcessing("Removing background (GPU)...", pct, current, total);
          }
        }
      });
    } catch (gpuErr) {
      console.warn('[V6 Engine] GPU failed for image, falling back to CPU:', gpuErr.message);
      resultBlob = await removeBackground(inputBlob, {
        device: 'cpu',
        progress: (key, current, total) => {
          if (key === 'compute:inference') {
            const pct = 30 + (current / total) * 60;
            updateProcessing("Removing background (CPU)...", pct, current, total);
          }
        }
      });
    }

    updateProcessing("Finalizing...", 95, 1, 1);

    const resultUrl = URL.createObjectURL(resultBlob);
    const resultImg = new Image();
    resultImg.crossOrigin = 'anonymous';

    await new Promise((resolve, reject) => {
      resultImg.onload = resolve;
      resultImg.onerror = reject;
      resultImg.src = resultUrl;
    });

    imageMaskCanvas = document.createElement('canvas');
    imageMaskCanvas.width = resultImg.width;
    imageMaskCanvas.height = resultImg.height;
    const ctx = imageMaskCanvas.getContext('2d');
    ctx.drawImage(resultImg, 0, 0);

    const origImg = new Image();
    origImg.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      origImg.onload = resolve;
      origImg.onerror = reject;
      origImg.src = url;
    });

    const aspect = origImg.width / origImg.height;
    plane.scale.set(aspect, 1, 1);
    videoTexture = new THREE.Texture(origImg);
    videoTexture.needsUpdate = true;
    material.uniforms.uTexture.value = videoTexture;

    currentMaskTexture = new THREE.CanvasTexture(imageMaskCanvas);
    material.uniforms.uMask.value = currentMaskTexture;

    renderer.render(scene, camera);

    hideProcessing();
    updateStatus("Ready", "ready");
    showToast("✅ Image processed successfully!");
    URL.revokeObjectURL(resultUrl);

  } catch (err) {
    console.error("[V6 Engine] Image processing error:", err);
    hideProcessing();
    updateStatus("Error", "error");
    showToast("❌ Error processing image: " + err.message);
  }
}

// ============================================================================
// V6: VIDEO PROCESSING — PARALLEL WORKER PIPELINE WITH INTEGRITY CHECKS
// ============================================================================

async function loadAndProcessVideo(url) {
  showProcessing("Loading Video", "Preparing video for parallel processing...");
  updateStatus("Processing...", "processing");
  isVideo = true;
  frameMasks = [];

  if (video) { video.pause(); video.src = ""; if (video.parentNode) video.parentNode.removeChild(video); }
  video = document.createElement('video');
  video.src = url;
  video.loop = false;
  video.muted = true;
  video.crossOrigin = 'anonymous';
  video.playsInline = true;
  video.style.display = 'none';
  document.body.appendChild(video);

  await new Promise((resolve) => { video.onloadedmetadata = resolve; });

  video.width = video.videoWidth;
  video.height = video.videoHeight;
  const aspect = video.videoWidth / video.videoHeight;
  plane.scale.set(aspect, 1, 1);
  videoTexture = new THREE.VideoTexture(video);
  material.uniforms.uTexture.value = videoTexture;

  const duration = video.duration;
  const fps = 5;
  const frameTime = 1 / fps;
  totalFrames = Math.ceil(duration * fps);

  let INFER_SIZE = 512;
  const scale = Math.min(INFER_SIZE / video.videoWidth, INFER_SIZE / video.videoHeight, 1);
  let inferW = Math.round(video.videoWidth * scale);
  let inferH = Math.round(video.videoHeight * scale);

  showProcessing("Extracting Frames", `Analyzing ${totalFrames} frames...`);

  // ---- PHASE 1: Extract frames + V6 edge-based similarity analysis ----
  const extractCanvas = document.createElement('canvas');
  extractCanvas.width = inferW;
  extractCanvas.height = inferH;
  const extractCtx = extractCanvas.getContext('2d');

  const DIFF_SAMPLE_SIZE = 64;
  const diffCanvas = document.createElement('canvas');
  diffCanvas.width = DIFF_SAMPLE_SIZE;
  diffCanvas.height = DIFF_SAMPLE_SIZE;
  const diffCtx = diffCanvas.getContext('2d');

  const frameJobs = [];
  let prevDiffData = null;
  let prevEdges = null;
  let skippedTotal = 0;

  for (let i = 0; i < totalFrames; i++) {
    const targetTime = Math.min(i * frameTime, duration - 0.01);

    video.currentTime = targetTime;
    await new Promise(resolve => {
      const onSeeked = () => { video.removeEventListener('seeked', onSeeked); resolve(); };
      video.addEventListener('seeked', onSeeked);
    });

    if (fastMode && inferW > 320) {
      const newScale = Math.min(320 / video.videoWidth, 320 / video.videoHeight, 1);
      inferW = Math.round(video.videoWidth * newScale);
      inferH = Math.round(video.videoHeight * newScale);
      extractCanvas.width = inferW;
      extractCanvas.height = inferH;
    }

    // V6: Edge-based similarity check
    diffCtx.drawImage(video, 0, 0, DIFF_SAMPLE_SIZE, DIFF_SAMPLE_SIZE);
    const currentDiffData = diffCtx.getImageData(0, 0, DIFF_SAMPLE_SIZE, DIFF_SAMPLE_SIZE).data;

    const analysis = analyzeFrameSimilarity(
      currentDiffData, prevDiffData, prevEdges,
      DIFF_SAMPLE_SIZE, DIFF_SAMPLE_SIZE, i
    );

    // V6: Dynamic threshold — if too many frames skipped, stop skipping
    let isSimilar = analysis.similar;
    if (isSimilar && skippedTotal > totalFrames * 0.4) {
      isSimilar = false; // Force processing — too many skips
      console.log(`[V6] Frame ${i}: forced processing (>40% skipped already)`);
    }

    if (isSimilar) skippedTotal++;

    prevDiffData = new Uint8ClampedArray(currentDiffData);
    prevEdges = analysis.edges || computeEdgeMap(currentDiffData, DIFF_SAMPLE_SIZE, DIFF_SAMPLE_SIZE);

    // Extract frame pixels
    extractCtx.drawImage(video, 0, 0, inferW, inferH);
    const frameImageData = extractCtx.getImageData(0, 0, inferW, inferH);

    frameJobs.push({
      index: i,
      time: targetTime,
      imageData: frameImageData.data.buffer.slice(0),
      width: inferW,
      height: inferH,
      similar: isSimilar,
      pixelDiff: analysis.pixelDiff,
      edgeDiff: analysis.edgeDiff,
    });

    const extractPct = ((i + 1) / totalFrames) * 20;
    updateProcessing(`Analyzing frame ${i + 1}/${totalFrames}`, extractPct, i + 1, totalFrames);
  }

  // ---- PHASE 2: Dispatch to worker pool in parallel ----
  const uniqueFrameJobs = frameJobs.filter(f => !f.similar);
  const totalToProcess = uniqueFrameJobs.length;
  const skippedCount = frameJobs.length - totalToProcess;

  console.log(`[V6 Engine] ${totalToProcess} unique frames, ${skippedCount} similar skipped`);
  showProcessing("AI Processing", `Processing ${totalToProcess} frames with ${WORKER_COUNT} workers...`);

  const maskResults = new Array(totalFrames).fill(null);
  const coverageResults = new Array(totalFrames).fill(0);
  let completedCount = 0;
  let lastValidCoverage = 0;

  await new Promise((resolve) => {
    if (totalToProcess === 0) { resolve(); return; }

    let jobIndex = 0;

    const workerHandlers = [];
    for (let w = 0; w < WORKER_COUNT; w++) {
      const handler = (e) => {
        if (e.data.type === 'result') {
          const { frameIndex, alpha, maskWidth, maskHeight, coverage } = e.data;

          if (alpha) {
            const alphaArr = new Uint8Array(alpha);

            // V6: Mask integrity validation
            if (validateMask(coverage, lastValidCoverage)) {
              maskResults[frameIndex] = {
                alpha: alphaArr,
                width: maskWidth,
                height: maskHeight,
                source: 'new',
              };
              coverageResults[frameIndex] = coverage;
              lastValidCoverage = coverage;
            } else {
              // Rejected — will be filled in Phase 3
              console.log(`[V6] Frame ${frameIndex}: mask rejected, will use fallback`);
              maskResults[frameIndex] = null;
              coverageResults[frameIndex] = coverage;
            }
          }

          completedCount++;
          const pct = 20 + (completedCount / totalToProcess) * 75;
          updateProcessing(
            `Frame ${completedCount}/${totalToProcess} (${skippedCount} skipped)`,
            pct,
            completedCount,
            totalToProcess
          );

          if (jobIndex < uniqueFrameJobs.length) {
            dispatchJob(w, uniqueFrameJobs[jobIndex]);
            jobIndex++;
          }

          if (completedCount >= totalToProcess) {
            workerHandlers.forEach((h, idx) => {
              workers[idx].removeEventListener('message', h);
            });
            resolve();
          }
        }
      };
      workerHandlers.push(handler);
      workers[w].addEventListener('message', handler);
    }

    function dispatchJob(workerIdx, job) {
      const transferable = [job.imageData];
      workers[workerIdx].postMessage({
        type: 'process',
        frameIndex: job.index,
        imageData: job.imageData,
        width: job.width,
        height: job.height,
      }, transferable);
    }

    const initialBatch = Math.min(WORKER_COUNT, uniqueFrameJobs.length);
    for (let w = 0; w < initialBatch; w++) {
      dispatchJob(w, uniqueFrameJobs[jobIndex]);
      jobIndex++;
    }
  });

  // ---- PHASE 3: Fill in skipped/rejected frames with nearest valid mask ----
  updateProcessing("Stabilizing masks...", 96, totalToProcess, totalToProcess);

  for (let i = 0; i < totalFrames; i++) {
    if (!maskResults[i]) {
      let nearest = null;
      // Prefer backward (previous frame) over forward for temporal consistency
      for (let d = 1; d < totalFrames; d++) {
        if (i - d >= 0 && maskResults[i - d]) { nearest = maskResults[i - d]; break; }
        if (i + d < totalFrames && maskResults[i + d]) { nearest = maskResults[i + d]; break; }
      }
      if (nearest) {
        maskResults[i] = { ...nearest, source: 'reused' };
      }
    }
  }

  // Build frameMasks array with metadata
  frameMasks = frameJobs.map((job, i) => ({
    time: job.time,
    alpha: maskResults[i]?.alpha || null,
    width: maskResults[i]?.width || inferW,
    height: maskResults[i]?.height || inferH,
    coverage: coverageResults[i] || 0,
    source: maskResults[i]?.source || 'none',
    pixelDiff: job.pixelDiff || 0,
    edgeDiff: job.edgeDiff || 0,
  }));

  hideProcessing();

  const elapsed = ((performance.now() - processingStartTime) / 1000).toFixed(1);
  updateStatus("Ready", "ready");
  showToast(`✅ Done in ${elapsed}s! ${totalToProcess} frames processed, ${skippedCount} skipped.`);

  video.loop = true;
  video.currentTime = 0;
  video.play();
}

// ============================================================================
// V6: MASK TEXTURE WITH ORIENTATION FIX + CACHE
// ============================================================================

function getNearestMaskIndex(currentTime) {
  if (frameMasks.length === 0) return -1;

  let lo = 0, hi = frameMasks.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frameMasks[mid].time < currentTime) lo = mid + 1;
    else hi = mid;
  }

  if (lo > 0) {
    const distCurr = Math.abs(frameMasks[lo].time - currentTime);
    const distPrev = Math.abs(frameMasks[lo - 1].time - currentTime);
    if (distPrev < distCurr) lo--;
  }

  return lo;
}

let cachedMaskIndex = -1;
let cachedDataTexture = null;

function getMaskTexture(currentTime) {
  const idx = getNearestMaskIndex(currentTime);
  if (idx < 0) return null;

  if (idx === cachedMaskIndex && cachedDataTexture) return cachedDataTexture;

  const mask = frameMasks[idx];
  if (!mask || !mask.alpha) return cachedDataTexture;

  // V6: Flip alpha rows to fix DataTexture orientation
  // Canvas getImageData = top-left origin → DataTexture expects bottom-left origin
  const flippedAlpha = flipAlphaRows(mask.alpha, mask.width, mask.height);

  // Create RGBA DataTexture from flipped alpha buffer
  const rgba = new Uint8Array(mask.width * mask.height * 4);
  for (let i = 0; i < flippedAlpha.length; i++) {
    const a = flippedAlpha[i];
    rgba[i * 4] = 255;
    rgba[i * 4 + 1] = 255;
    rgba[i * 4 + 2] = 255;
    rgba[i * 4 + 3] = a;
  }

  if (cachedDataTexture) cachedDataTexture.dispose();
  cachedDataTexture = new THREE.DataTexture(rgba, mask.width, mask.height, THREE.RGBAFormat);
  cachedDataTexture.flipY = false; // V6: Prevent Three.js from applying its own flip
  cachedDataTexture.needsUpdate = true;
  cachedMaskIndex = idx;

  // V6: Update debug panel
  updateDebugInfo(idx, mask);

  return cachedDataTexture;
}

// ============================================================================
// V6: DEBUG OVERLAY
// ============================================================================

function updateDebugInfo(frameIdx, mask) {
  if (!debugMode || !debugPanel) return;

  debugPanel.innerHTML = `
    <div><strong>Frame:</strong> ${frameIdx}/${frameMasks.length - 1}</div>
    <div><strong>Time:</strong> ${mask.time.toFixed(2)}s</div>
    <div><strong>Source:</strong> <span class="dbg-${mask.source}">${mask.source}</span></div>
    <div><strong>Coverage:</strong> ${(mask.coverage * 100).toFixed(1)}%</div>
    <div><strong>Pixel Δ:</strong> ${mask.pixelDiff?.toFixed(1) || '--'}</div>
    <div><strong>Edge Δ:</strong> ${mask.edgeDiff?.toFixed(1) || '--'}</div>
    <div><strong>Size:</strong> ${mask.width}×${mask.height}</div>
  `;
}

// ============================================================================
// RENDER / EXPORT
// ============================================================================

// V9: Hide landing hero when media work begins
function hideLandingHero() {
  const hero = document.getElementById('landing-hero');
  if (hero) {
    hero.classList.add('hero-hidden');
    setTimeout(() => { hero.style.display = 'none'; }, 400);
  }
}

async function startRender() {
  if (!videoTexture) return showToast("Please load media first");

  if (!isVideo) {
    renderer.render(scene, camera);
    const a = document.createElement('a');
    a.href = renderer.domElement.toDataURL('image/png');
    a.download = `micro-bgremover-${Date.now()}.png`;
    a.click();
    showToast("🖼️ Image Rendered and Downloaded!");
    return;
  }

  if (!video) return showToast("Please load a video first");

  isRendering = true;
  renderBtn.textContent = "Cancel Render";
  renderBtn.classList.add('recording');
  progressContainer.style.display = 'block';
  updateStatus("Rendering...", "processing");

  video.pause();
  video.currentTime = 0;
  video.loop = false;

  recordedChunks = [];
  const stream = renderer.domElement.captureStream(60);
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';

  mediaRecorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 10000000 });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: mime });
    const url = URL.createObjectURL(blob);
    const filename = `micro-bgremover-${Date.now()}.webm`;

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();

    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);

    finishRender();
    showToast("🎬 Video Rendered and Downloaded!");
  };

  mediaRecorder.start();

  const videoDuration = video.duration;
  const renderFps = 30;
  const renderFrameTime = 1 / renderFps;
  let currentTime = 0;

  while (currentTime < videoDuration && isRendering) {
    video.currentTime = currentTime;

    await new Promise(resolve => {
      const onSeeked = () => { video.removeEventListener('seeked', onSeeked); resolve(); };
      video.addEventListener('seeked', onSeeked);
    });

    const tex = getMaskTexture(currentTime);
    if (tex) {
      material.uniforms.uMask.value = tex;
    }

    renderer.render(scene, camera);
    await new Promise(r => setTimeout(r, 20));

    currentTime += renderFrameTime;
    const progress = Math.min((currentTime / videoDuration) * 100, 100);
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `Rendering: ${Math.round(progress)}%`;
  }

  if (isRendering) {
    mediaRecorder.stop();
  }
}

function stopRendering() {
  isRendering = false;
  finishRender();
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
}

function finishRender() {
  isRendering = false;
  renderBtn.textContent = "Render Final Video";
  renderBtn.classList.remove('recording');
  progressContainer.style.display = 'none';
  updateStatus("Ready", "ready");
  if (video) { video.loop = true; video.play(); }
}

// ============================================================================
// UTILITIES
// ============================================================================

function clearMask() {
  maskCtx.fillStyle = 'black'; maskCtx.fillRect(0, 0, 1024, 1024);
  userMaskTexture.needsUpdate = true;
  showToast("Mask cleared");
}

function updateStatus(text, type) { statusText.textContent = text; statusIcon.className = type; }

function showToast(m) {
  const t = document.createElement('div'); t.className = 'toast'; t.textContent = m;
  toastContainer.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 3000);
}

function onWindowResize() {
  const w = container.clientWidth; const h = container.clientHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
}

// ============================================================================
// ANIMATION LOOP
// ============================================================================

let lastMaskTime = -1;
function animate() {
  requestAnimationFrame(animate);
  if (!isRendering) {
    if (isVideo && video && video.readyState >= 2 && frameMasks.length > 0) {
      const ct = video.currentTime;
      if (Math.abs(ct - lastMaskTime) > 0.03) {
        lastMaskTime = ct;
        const tex = getMaskTexture(ct);
        if (tex) {
          material.uniforms.uMask.value = tex;
        }
      }
    }
    renderer.render(scene, camera);
  }
}

// ============================================================================
// BOOT
// ============================================================================
setTimeout(init, 100);
