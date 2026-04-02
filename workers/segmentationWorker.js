/**
 * V9 Segmentation Worker
 * GPU → CPU → graceful fallback pipeline.
 * - Explicit publicPath for CDN WASM/model loading (fixes broken CPU in Arc/non-WebGPU)
 * - 45s warmup timeout (never gets stuck)
 * - Always sends 'ready' or 'error' — no silent hangs
 * - V8 post-processing: alpha boost + dilation + interior fill
 */

import { removeBackground } from '@imgly/background-removal';

let isReady = false;
let useGPU = false; // Determined during warmup

// The library's CDN for WASM/ONNX model files.
// Required when local serving fails (non-WebGPU browsers, Vercel/Netlify deployments).
const IMGLY_CDN = 'https://cdn.img.ly/packages/imgly/background-removal-js/1.7.0/dist/';

// ============================================================================
// WARMUP — try GPU, fallback CPU, always resolve within timeout
// ============================================================================

async function warmUp() {
  // Hard timeout: if ONNX hangs (e.g. dxil.dll crash), still report ready
  const timeoutMs = 45_000;
  let settled = false;

  const timeoutPromise = new Promise((_, reject) =>
    setTimeout(() => reject(new Error('Warmup timed out after 45s')), timeoutMs)
  );

  try {
    await Promise.race([doWarmUp(), timeoutPromise]);
  } catch (err) {
    if (!settled) {
      console.warn('[Worker] Warmup failed/timed out, marking as CPU-ready anyway:', err.message);
      isReady = true;
      useGPU = false;
      settled = true;
      // Still post 'ready' so the app isn't stuck — individual frames will handle their own errors
      self.postMessage({ type: 'ready', gpu: false });
    }
  }
}

async function doWarmUp() {
  const canvas = new OffscreenCanvas(8, 8);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#888';
  ctx.fillRect(0, 0, 8, 8);
  const blob = await canvas.convertToBlob({ type: 'image/png' });

  // Try GPU first
  try {
    await removeBackground(blob, {
      device: 'gpu',
      publicPath: IMGLY_CDN,
      output: { quality: 0.5 },
    });
    useGPU = true;
    isReady = true;
    console.log('[Worker] WebGPU warm-up successful');
    self.postMessage({ type: 'ready', gpu: true });
    return;
  } catch (gpuErr) {
    console.warn('[Worker] WebGPU unavailable, trying CPU:', gpuErr.message.substring(0, 120));
  }

  // Try CPU with explicit CDN publicPath (critical for Arc/non-WebGPU)
  try {
    await removeBackground(blob, {
      device: 'cpu',
      publicPath: IMGLY_CDN,
      output: { quality: 0.5 },
    });
    useGPU = false;
    isReady = true;
    console.log('[Worker] CPU warm-up successful');
    self.postMessage({ type: 'ready', gpu: false });
    return;
  } catch (cpuErr) {
    console.warn('[Worker] CPU also failed:', cpuErr.message.substring(0, 120));
    // Don't throw — fall through so the timeout handler posts 'ready' anyway
    isReady = true;
    useGPU = false;
    self.postMessage({ type: 'ready', gpu: false });
  }
}

// ============================================================================
// V8: POST-PROCESSING PIPELINE
// ============================================================================

function boostAlpha(alpha) {
  const boosted = new Uint8Array(alpha.length);
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] === 0) {
      boosted[i] = 0;
    } else {
      const norm = alpha[i] / 255;
      const lifted = Math.pow(norm, 0.55);
      boosted[i] = Math.min(255, Math.round(lifted * 255));
    }
  }
  return boosted;
}

function dilate(alpha, w, h) {
  const result = new Uint8Array(alpha.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let maxVal = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
            const v = alpha[ny * w + nx];
            if (v > maxVal) maxVal = v;
          }
        }
      }
      result[y * w + x] = maxVal;
    }
  }
  return result;
}

function fillInterior(alpha, w, h) {
  const HIGH_THRESHOLD = 200;
  const SURROUND_THRESHOLD = 80;
  const FILL_RADIUS = 2;
  const MIN_NEIGHBOR_FRAC = 0.4;
  const result = new Uint8Array(alpha);

  for (let y = FILL_RADIUS; y < h - FILL_RADIUS; y++) {
    for (let x = FILL_RADIUS; x < w - FILL_RADIUS; x++) {
      const idx = y * w + x;
      const center = alpha[idx];
      if (center >= HIGH_THRESHOLD || center === 0) continue;

      let sum = 0, highCount = 0, totalCount = 0;
      for (let dy = -FILL_RADIUS; dy <= FILL_RADIUS; dy++) {
        for (let dx = -FILL_RADIUS; dx <= FILL_RADIUS; dx++) {
          if (dy === 0 && dx === 0) continue;
          const nv = alpha[(y + dy) * w + (x + dx)];
          sum += nv;
          if (nv > SURROUND_THRESHOLD) highCount++;
          totalCount++;
        }
      }

      const avgNeighbor = sum / totalCount;
      const highFrac = highCount / totalCount;
      if (avgNeighbor > SURROUND_THRESHOLD && highFrac >= MIN_NEIGHBOR_FRAC) {
        result[idx] = Math.min(255, Math.round(center + (avgNeighbor - center) * 0.6));
      }
    }
  }
  return result;
}

// ============================================================================
// FRAME PROCESSING
// ============================================================================

async function processFrame(frameData) {
  const { frameIndex, imageData, width, height } = frameData;

  try {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(new ImageData(new Uint8ClampedArray(imageData), width, height), 0, 0);

    const blob = await canvas.convertToBlob({ type: 'image/png' });

    const resultBlob = await removeBackground(blob, {
      device: useGPU ? 'gpu' : 'cpu',
      publicPath: IMGLY_CDN,
    });

    const resultBitmap = await createImageBitmap(resultBlob);
    const resultCanvas = new OffscreenCanvas(resultBitmap.width, resultBitmap.height);
    const resultCtx = resultCanvas.getContext('2d');
    resultCtx.drawImage(resultBitmap, 0, 0);
    resultBitmap.close();

    const resultImageData = resultCtx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
    const pixels = resultImageData.data;
    const maskW = resultCanvas.width;
    const maskH = resultCanvas.height;
    const totalPixels = maskW * maskH;

    let rawAlpha = new Uint8Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) rawAlpha[i] = pixels[i * 4 + 3];

    let processedAlpha = boostAlpha(rawAlpha);
    processedAlpha = dilate(processedAlpha, maskW, maskH);
    processedAlpha = fillInterior(processedAlpha, maskW, maskH);

    let nonZeroCount = 0, alphaSum = 0;
    for (let i = 0; i < totalPixels; i++) {
      if (processedAlpha[i] > 10) nonZeroCount++;
      alphaSum += processedAlpha[i];
    }
    const coverage = nonZeroCount / totalPixels;
    const avgAlpha = alphaSum / totalPixels;

    if (coverage < 0.01) {
      console.warn(`[Worker] Frame ${frameIndex}: near-empty mask (${(coverage * 100).toFixed(1)}%)`);
    }

    self.postMessage(
      { type: 'result', frameIndex, alpha: processedAlpha.buffer, maskWidth: maskW, maskHeight: maskH, coverage, avgAlpha },
      [processedAlpha.buffer]
    );
  } catch (err) {
    console.error(`[Worker] Frame ${frameIndex} failed:`, err.message);
    self.postMessage({ type: 'result', frameIndex, alpha: null, maskWidth: 0, maskHeight: 0, coverage: 0, avgAlpha: 0, error: err.message });
  }
}

self.onmessage = async (e) => {
  const { type } = e.data;
  if (type === 'init') {
    warmUp();
  } else if (type === 'process') {
    await processFrame(e.data);
  }
};
