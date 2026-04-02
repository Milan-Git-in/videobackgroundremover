/**
 * V8 Segmentation Worker
 * GPU-accelerated background removal with:
 * - Mask coverage validation (V6)
 * - Alpha confidence reinforcement (V8): boost + dilation + interior fill
 */

import { removeBackground } from '@imgly/background-removal';

let isReady = false;
let useGPU = true;

async function warmUp() {
  try {
    const canvas = new OffscreenCanvas(8, 8);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#888';
    ctx.fillRect(0, 0, 8, 8);
    const blob = await canvas.convertToBlob({ type: 'image/png' });

    try {
      await removeBackground(blob, { device: 'gpu' });
      useGPU = true;
      console.log('[Worker] WebGPU warm-up successful');
    } catch (gpuErr) {
      console.warn('[Worker] WebGPU unavailable, falling back to CPU:', gpuErr.message);
      useGPU = false;
      await removeBackground(blob, { device: 'cpu' });
      console.log('[Worker] CPU warm-up successful');
    }

    isReady = true;
    self.postMessage({ type: 'ready', gpu: useGPU });
  } catch (err) {
    console.error('[Worker] Warm-up failed:', err);
    self.postMessage({ type: 'error', message: 'Worker warm-up failed: ' + err.message });
  }
}

// ============================================================================
// V8: POST-PROCESSING PIPELINE
// ============================================================================

/**
 * Step 1: Alpha Boost
 * Nonlinear curve lifts weak interior values without blowing out edges.
 * Uses a blended pow curve:  alpha = pow(alpha, 0.55)
 * Pixels with alpha > 0 are boosted; alpha == 0 stays 0 (true background).
 */
function boostAlpha(alpha) {
  const boosted = new Uint8Array(alpha.length);
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] === 0) {
      boosted[i] = 0; // True background — never boost
    } else {
      // Normalize to [0,1], apply power curve, re-normalize to [0,255]
      const norm = alpha[i] / 255;
      const lifted = Math.pow(norm, 0.55); // Boosts mid-low values, preserves peaks
      boosted[i] = Math.min(255, Math.round(lifted * 255));
    }
  }
  return boosted;
}

/**
 * Step 2: Morphological Dilation (1px radius)
 * Expands the mask outward by 1 pixel, filling small gaps and holes at subject boundary.
 * Uses a 3x3 max-filter kernel.
 */
function dilate(alpha, w, h) {
  const result = new Uint8Array(alpha.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let maxVal = 0;
      // 3x3 neighborhood max
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

/**
 * Step 3: Context-Aware Interior Fill
 * For each pixel where alpha < HIGH_THRESHOLD:
 *   Sample its 5x5 neighborhood.
 *   If the average of neighbors is > SURROUND_THRESHOLD, pull the pixel toward that value.
 * This rescues isolated low-confidence interior pixels surrounded by confident foreground.
 *
 * NOTE: Only strengthens interior pixels — never assigns foreground to true background (alpha==0).
 */
function fillInterior(alpha, w, h) {
  const HIGH_THRESHOLD = 200;   // Pixels below this can be a candidate for boosting
  const SURROUND_THRESHOLD = 80; // If neighborhood avg > this, boost the center pixel
  const FILL_RADIUS = 2;         // 5x5 kernel (radius 2)
  const MIN_NEIGHBOR_FRAC = 0.4; // At least 40% of neighbors must be high-alpha

  const result = new Uint8Array(alpha); // copy

  for (let y = FILL_RADIUS; y < h - FILL_RADIUS; y++) {
    for (let x = FILL_RADIUS; x < w - FILL_RADIUS; x++) {
      const idx = y * w + x;
      const center = alpha[idx];

      // Skip if already high confidence or true background
      if (center >= HIGH_THRESHOLD || center === 0) continue;

      // Sample neighborhood
      let sum = 0;
      let highCount = 0;
      let totalCount = 0;
      for (let dy = -FILL_RADIUS; dy <= FILL_RADIUS; dy++) {
        for (let dx = -FILL_RADIUS; dx <= FILL_RADIUS; dx++) {
          if (dy === 0 && dx === 0) continue; // Exclude self
          const nv = alpha[(y + dy) * w + (x + dx)];
          sum += nv;
          if (nv > SURROUND_THRESHOLD) highCount++;
          totalCount++;
        }
      }

      const avgNeighbor = sum / totalCount;
      const highFrac = highCount / totalCount;

      // If surrounded by confident foreground, pull center toward neighborhood avg
      if (avgNeighbor > SURROUND_THRESHOLD && highFrac >= MIN_NEIGHBOR_FRAC) {
        // Blend: move center 60% toward neighborhood average
        const blended = center + (avgNeighbor - center) * 0.6;
        result[idx] = Math.min(255, Math.round(blended));
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
    const imgData = new ImageData(new Uint8ClampedArray(imageData), width, height);
    ctx.putImageData(imgData, 0, 0);

    const blob = await canvas.convertToBlob({ type: 'image/png' });

    const resultBlob = await removeBackground(blob, {
      device: useGPU ? 'gpu' : 'cpu',
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

    // Extract raw alpha from model output
    let rawAlpha = new Uint8Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      rawAlpha[i] = pixels[i * 4 + 3];
    }

    // ---- V8: POST-PROCESSING PIPELINE ----
    // Step 1: Boost weak interior alpha values
    let processedAlpha = boostAlpha(rawAlpha);

    // Step 2: Dilate to close small boundary gaps (1px)
    processedAlpha = dilate(processedAlpha, maskW, maskH);

    // Step 3: Context-aware interior fill (rescues isolated weak pixels)
    processedAlpha = fillInterior(processedAlpha, maskW, maskH);
    // ----------------------------------------

    // Coverage metric (on final processed alpha)
    let nonZeroCount = 0;
    let alphaSum = 0;
    for (let i = 0; i < totalPixels; i++) {
      const a = processedAlpha[i];
      if (a > 10) nonZeroCount++;
      alphaSum += a;
    }
    const coverage = nonZeroCount / totalPixels;
    const avgAlpha = alphaSum / totalPixels;

    if (coverage < 0.01) {
      console.warn(`[Worker] Frame ${frameIndex}: near-empty mask (coverage=${(coverage * 100).toFixed(1)}%)`);
    }

    self.postMessage(
      {
        type: 'result',
        frameIndex,
        alpha: processedAlpha.buffer,
        maskWidth: maskW,
        maskHeight: maskH,
        coverage,
        avgAlpha,
      },
      [processedAlpha.buffer]
    );
  } catch (err) {
    console.error(`[Worker] Frame ${frameIndex} failed:`, err);
    self.postMessage({
      type: 'result',
      frameIndex,
      alpha: null,
      maskWidth: 0,
      maskHeight: 0,
      coverage: 0,
      avgAlpha: 0,
      error: err.message,
    });
  }
}

self.onmessage = async (e) => {
  const { type } = e.data;
  if (type === 'init') {
    warmUp();
  } else if (type === 'process') {
    if (!isReady) console.warn('[Worker] Received frame before ready, processing anyway...');
    await processFrame(e.data);
  }
};
