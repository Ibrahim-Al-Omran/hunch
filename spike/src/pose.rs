//! BlazePose two-stage pipeline on ONNX Runtime, mirroring
//! `prototype/hunch_cv/pose.py::OnnxBackend` so outputs can be compared.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use image::RgbImage;
use ort::environment::Environment;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

const DET_SIZE: usize = 224;
const LM_SIZE: usize = 256;
pub const NUM_LANDMARKS: usize = 33;

#[derive(Clone, Copy, Debug)]
pub struct Landmark {
    /// Pixel coordinates in the source image.
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub visibility: f32,
    pub presence: f32,
}

pub struct PoseModel {
    _env: Environment,
    det: Session,
    lm: Session,
    anchors: Vec<[f32; 2]>,
    pub det_threshold: f32,
    pub presence_threshold: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-80.0, 80.0)).exp())
}

fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow!("onnxruntime: {e}")
}

fn session(env: &Environment, path: &Path, threads: usize) -> Result<Session> {
    Session::builder(env)
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(threads)
        .map_err(ort_err)?
        .with_inter_threads(1)
        .map_err(ort_err)?
        .commit_from_file(path)
        .map_err(ort_err)
        .with_context(|| format!("loading {}", path.display()))
}

pub fn ssd_anchors() -> Vec<[f32; 2]> {
    let strides = [8usize, 16, 32, 32, 32];
    let mut out = Vec::with_capacity(2254);
    let mut layer = 0;
    while layer < strides.len() {
        let mut same = 0;
        let mut last = layer;
        while last < strides.len() && strides[last] == strides[layer] {
            same += 2;
            last += 1;
        }
        let fm = DET_SIZE.div_ceil(strides[layer]);
        for y in 0..fm {
            for x in 0..fm {
                for _ in 0..same {
                    out.push([(x as f32 + 0.5) / fm as f32, (y as f32 + 0.5) / fm as f32]);
                }
            }
        }
        layer = last;
    }
    out
}

/// Bilinear sample with edge replication; (x, y) in source pixel coordinates.
#[inline]
fn sample(img: &RgbImage, x: f32, y: f32, out: &mut [f32]) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x0f = x.floor();
    let y0f = y.floor();
    let fx = x - x0f;
    let fy = y - y0f;
    let cx = |v: i64| v.clamp(0, w - 1) as u32;
    let cy = |v: i64| v.clamp(0, h - 1) as u32;
    let (x0, y0) = (x0f as i64, y0f as i64);
    let p00 = img.get_pixel(cx(x0), cy(y0)).0;
    let p10 = img.get_pixel(cx(x0 + 1), cy(y0)).0;
    let p01 = img.get_pixel(cx(x0), cy(y0 + 1)).0;
    let p11 = img.get_pixel(cx(x0 + 1), cy(y0 + 1)).0;
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
}

struct Detection {
    kps: [[f32; 2]; 4],
}

impl PoseModel {
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self> {
        let env = ort::init().with_name("hunch-spike").with_telemetry(false).build().map_err(ort_err)?;
        Ok(Self {
            det: session(&env, &model_dir.join("pose_detector.onnx"), threads)?,
            lm: session(&env, &model_dir.join("pose_landmarks_detector.onnx"), threads)?,
            _env: env,
            anchors: ssd_anchors(),
            det_threshold: 0.5,
            presence_threshold: 0.5,
        })
    }

    fn detect_person(&mut self, img: &RgbImage) -> Result<Option<Detection>> {
        let (w, h) = (img.width() as f32, img.height() as f32);
        let s = DET_SIZE as f32;
        let scale = s / w.max(h);
        let (nw, nh) = ((w * scale).round() as usize, (h * scale).round() as usize);
        let (pad_x, pad_y) = ((DET_SIZE - nw) / 2, (DET_SIZE - nh) / 2);
        let mut input = vec![-1.0f32; DET_SIZE * DET_SIZE * 3]; // zero padding after normalization
        let mut px = [0f32; 3];
        for y in 0..nh {
            for x in 0..nw {
                let sx = (x as f32 + 0.5) / scale - 0.5;
                let sy = (y as f32 + 0.5) / scale - 0.5;
                sample(img, sx, sy, &mut px);
                let o = ((y + pad_y) * DET_SIZE + x + pad_x) * 3;
                for c in 0..3 {
                    input[o + c] = px[c] / 127.5 - 1.0;
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, DET_SIZE, DET_SIZE, 3], input.into_boxed_slice()))?;
        let outputs = self.det.run(ort::inputs![tensor])?;
        let (_, boxes) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, scores) = outputs[1].try_extract_tensor::<f32>()?;

        let mut cands: Vec<(f32, [f32; 4], [[f32; 2]; 4])> = Vec::new();
        for (i, a) in self.anchors.iter().enumerate() {
            let score = sigmoid(scores[i].clamp(-100.0, 100.0));
            if score < self.det_threshold {
                continue;
            }
            let r = &boxes[i * 12..i * 12 + 12];
            let (cx, cy) = (r[0] / s + a[0], r[1] / s + a[1]);
            let (bw, bh) = (r[2] / s, r[3] / s);
            let mut kps = [[0f32; 2]; 4];
            for (k, kp) in kps.iter_mut().enumerate() {
                *kp = [r[4 + 2 * k] / s + a[0], r[5 + 2 * k] / s + a[1]];
            }
            cands.push((score, [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], kps));
        }
        let Some(best) = cands.iter().cloned().max_by(|a, b| a.0.total_cmp(&b.0)) else {
            return Ok(None);
        };
        let iou = |a: &[f32; 4], b: &[f32; 4]| {
            let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
            let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            let inter = ix * iy;
            let area = |r: &[f32; 4]| (r[2] - r[0]) * (r[3] - r[1]);
            inter / (area(a) + area(b) - inter).max(1e-9)
        };
        let mut acc = [[0f32; 2]; 4];
        let mut total = 0f32;
        for (score, bx, kps) in &cands {
            if iou(&best.1, bx) > 0.3 {
                total += score;
                for k in 0..4 {
                    acc[k][0] += kps[k][0] * score;
                    acc[k][1] += kps[k][1] * score;
                }
            }
        }
        let mut kps = [[0f32; 2]; 4];
        for k in 0..4 {
            kps[k] = [
                (acc[k][0] / total * s - pad_x as f32) / scale,
                (acc[k][1] / total * s - pad_y as f32) / scale,
            ];
        }
        Ok(Some(Detection { kps }))
    }

    pub fn detect(&mut self, img: &RgbImage) -> Result<Option<Vec<Landmark>>> {
        let Some(det) = self.detect_person(img)? else {
            return Ok(None);
        };
        let ([x0, y0], [x1, y1]) = (det.kps[0], det.kps[1]);
        let mut rot = std::f32::consts::FRAC_PI_2 - (-(y1 - y0)).atan2(x1 - x0);
        rot = (rot + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI;
        let side = 2.0 * (x1 - x0).hypot(y1 - y0) * 1.25;
        let n = LM_SIZE as f32;
        let (c, sn) = (rot.cos(), rot.sin());
        let k = side / n;
        // crop (u, v) -> image, identical to the Python port
        let m = [
            [c * k, -sn * k, x0 - c * k * n / 2.0 + sn * k * n / 2.0],
            [sn * k, c * k, y0 - sn * k * n / 2.0 - c * k * n / 2.0],
        ];
        let mut input = vec![0f32; LM_SIZE * LM_SIZE * 3];
        let mut px = [0f32; 3];
        for v in 0..LM_SIZE {
            for u in 0..LM_SIZE {
                let (uf, vf) = (u as f32, v as f32);
                let sx = m[0][0] * uf + m[0][1] * vf + m[0][2];
                let sy = m[1][0] * uf + m[1][1] * vf + m[1][2];
                sample(img, sx, sy, &mut px);
                let o = (v * LM_SIZE + u) * 3;
                for ch in 0..3 {
                    input[o + ch] = px[ch] / 255.0;
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, LM_SIZE, LM_SIZE, 3], input.into_boxed_slice()))?;
        let outputs = self.lm.run(ort::inputs![tensor])?;
        let (_, raw) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, flag) = outputs[1].try_extract_tensor::<f32>()?;
        let (hm_shape, heatmap) = outputs[3].try_extract_tensor::<f32>()?;
        if flag[0] < self.presence_threshold {
            return Ok(None);
        }
        let (rows, cols, nl) = (hm_shape[1] as usize, hm_shape[2] as usize, hm_shape[3] as usize);
        let mut out = Vec::with_capacity(NUM_LANDMARKS);
        for i in 0..NUM_LANDMARKS {
            let r = &raw[i * 5..i * 5 + 5];
            let (mut u, mut v) = (r[0] / n, r[1] / n);
            refine(&mut u, &mut v, heatmap, rows, cols, nl, i);
            let (uf, vf) = (u * n, v * n);
            out.push(Landmark {
                x: m[0][0] * uf + m[0][1] * vf + m[0][2],
                y: m[1][0] * uf + m[1][1] * vf + m[1][2],
                z: r[2] / n * side,
                visibility: sigmoid(r[3]),
                presence: sigmoid(r[4]),
            });
        }
        Ok(Some(out))
    }
}

/// MediaPipe's heatmap refinement: confidence-weighted centroid in a 7x7 window.
fn refine(u: &mut f32, v: &mut f32, hm: &[f32], rows: usize, cols: usize, nl: usize, idx: usize) {
    let (c, r) = ((*u * cols as f32) as i64, (*v * rows as f32) as i64);
    if c < 0 || r < 0 || c >= cols as i64 || r >= rows as i64 {
        return;
    }
    let off = 3i64;
    let (r0, r1) = ((r - off).max(0), (r + off + 1).min(rows as i64));
    let (c0, c1) = ((c - off).max(0), (c + off + 1).min(cols as i64));
    let (mut total, mut wc, mut wr, mut maxv) = (0f32, 0f32, 0f32, 0f32);
    for rr in r0..r1 {
        for cc in c0..c1 {
            let conf = sigmoid(hm[(rr as usize * cols + cc as usize) * nl + idx]);
            total += conf;
            wc += cc as f32 * conf;
            wr += rr as f32 * conf;
            maxv = maxv.max(conf);
        }
    }
    if maxv >= 0.5 && total > 0.0 {
        *u = wc / total / cols as f32;
        *v = wr / total / rows as f32;
    }
}
