//! Phase 1 spike: what does native capture + ONNX pose inference cost?
//!
//!   hunch-spike cameras                                  devices and formats
//!   hunch-spike image <file> [--variant lite]            landmarks as JSON (parity check)
//!   hunch-spike camera [--index 0] [--interval 2] [--duration 60]
//!                      [--mode keep|reopen] [--width 320 --height 240] [--fps 30]
//!                      [--variant lite] [--threads 1] [--flush 1] [--no-infer]
//!
//! Frames are processed in memory and never written to disk.

mod pose;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use cpu_time::ProcessTime;
use image::RgbImage;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use pose::PoseModel;

struct Args {
    cmd: String,
    positional: Vec<String>,
    opts: std::collections::HashMap<String, String>,
}

impl Args {
    fn parse() -> Self {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_default();
        let (mut positional, mut opts) = (Vec::new(), std::collections::HashMap::new());
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            if let Some(k) = rest[i].strip_prefix("--") {
                let flag = i + 1 >= rest.len() || rest[i + 1].starts_with("--");
                opts.insert(k.to_string(), if flag { "true".into() } else { rest[i + 1].clone() });
                i += if flag { 1 } else { 2 };
            } else {
                positional.push(rest[i].clone());
                i += 1;
            }
        }
        Self { cmd, positional, opts }
    }
    fn get<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        self.opts.get(k).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
    fn flag(&self, k: &str) -> bool {
        self.opts.contains_key(k)
    }
}

fn model_dir(variant: &str) -> PathBuf {
    let root = std::env::var("HUNCH_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../prototype/models"));
    root.join(variant)
}

fn rss_mb() -> f64 {
    memory_stats::memory_stats().map(|m| m.physical_mem as f64 / 1048576.0).unwrap_or(f64::NAN)
}

fn landmarks_json(lms: &[pose::Landmark]) -> serde_json::Value {
    serde_json::Value::Array(
        lms.iter()
            .map(|l| serde_json::json!([l.x, l.y, l.z, l.visibility, l.presence]))
            .collect(),
    )
}

fn run_image(args: &Args) -> Result<()> {
    let path = args.positional.first().context("image path required")?;
    let img = image::open(path)?.to_rgb8();
    let mut model = PoseModel::new(&model_dir(&args.get("variant", "lite".to_string())), args.get("threads", 1))?;
    let t = Instant::now();
    let lms = model.detect(&img)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let repeat: usize = args.get("repeat", 0);
    let (mut walls, cpu0) = (Vec::with_capacity(repeat), ProcessTime::now());
    for _ in 0..repeat {
        let t = Instant::now();
        model.detect(&img)?;
        walls.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let cpu_ms = if repeat > 0 { cpu0.elapsed().as_secs_f64() * 1e3 / repeat as f64 } else { f64::NAN };
    walls.sort_by(|a, b| a.total_cmp(b));
    let out = serde_json::json!({
        "width": img.width(), "height": img.height(), "ms": ms,
        "steady_ms_median": walls.get(walls.len() / 2), "steady_cpu_ms_mean": cpu_ms,
        "rss_mb": rss_mb(),
        "landmarks": lms.as_deref().map(landmarks_json),
    });
    println!("{out}");
    Ok(())
}

fn open_camera(index: u32, w: u32, h: u32, fps: u32) -> Result<Camera> {
    let mut last_err = None;
    for ff in [FrameFormat::MJPEG, FrameFormat::YUYV, FrameFormat::NV12] {
        let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
            Resolution::new(w, h),
            ff,
            fps,
        )));
        match Camera::new(CameraIndex::Index(index), fmt) {
            Ok(mut cam) => {
                cam.open_stream()?;
                return Ok(cam);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap().into())
}

fn list_cameras() -> Result<()> {
    for info in nokhwa::query(nokhwa::utils::ApiBackend::Auto)? {
        println!("[{}] {} ({})", info.index(), info.human_name(), info.description());
        if let CameraIndex::Index(i) = info.index() {
            let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
            match Camera::new(CameraIndex::Index(*i), fmt) {
                Ok(mut cam) => {
                    let mut formats = cam.compatible_camera_formats().unwrap_or_default();
                    formats.sort_by_key(|f| (f.format().to_string(), f.resolution().width(), f.frame_rate()));
                    for f in formats.iter().filter(|f| f.resolution().width() <= 640) {
                        println!("    {} {}x{} @{}", f.format(), f.resolution().width(), f.resolution().height(), f.frame_rate());
                    }
                }
                Err(e) => println!("    (could not query formats: {e})"),
            }
        }
    }
    Ok(())
}

fn grab(cam: &mut Camera) -> Result<RgbImage> {
    Ok(cam.frame()?.decode_image::<RgbFormat>()?)
}

fn run_camera(args: &Args) -> Result<()> {
    let index: u32 = args.get("index", 0);
    let interval = Duration::from_secs_f64(args.get("interval", 2.0));
    let duration = Duration::from_secs_f64(args.get("duration", 60.0));
    let (w, h): (u32, u32) = (args.get("width", 320), args.get("height", 240));
    let mode = args.get("mode", "keep".to_string());
    let infer = !args.flag("no-infer");
    let flush: usize = args.get("flush", 1);
    let fps: u32 = args.get("fps", 30);

    let rss_start = rss_mb();
    let t_load = Instant::now();
    let mut model = if infer {
        Some(PoseModel::new(&model_dir(&args.get("variant", "lite".to_string())), args.get("threads", 1))?)
    } else {
        None
    };
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let rss_model = rss_mb();

    let mut cam = if mode == "keep" { Some(open_camera(index, w, h, fps)?) } else { None };
    if let Some(c) = cam.as_mut() {
        grab(c)?; // warm-up
    }

    let cpu0 = ProcessTime::now();
    let t0 = Instant::now();
    let (mut n, mut detected, mut rss_peak) = (0usize, 0usize, rss_mb());
    let (mut capture_ms, mut infer_ms) = (Vec::new(), Vec::new());
    let mut actual = (0, 0);
    while t0.elapsed() < duration {
        let cycle = Instant::now();
        let tc = Instant::now();
        let frame = match mode.as_str() {
            "keep" => {
                let c = cam.as_mut().unwrap();
                // frames queued while idle are stale; drop them before using one
                for _ in 0..flush {
                    c.frame()?;
                }
                grab(c)?
            }
            "reopen" => {
                let mut c = open_camera(index, w, h, fps)?;
                for _ in 0..flush {
                    c.frame()?;
                }
                let f = grab(&mut c)?;
                c.stop_stream()?;
                f
            }
            m => bail!("unknown mode {m}"),
        };
        capture_ms.push(tc.elapsed().as_secs_f64() * 1e3);
        actual = (frame.width(), frame.height());
        if let Some(m) = model.as_mut() {
            let ti = Instant::now();
            if m.detect(&frame)?.is_some() {
                detected += 1;
            }
            infer_ms.push(ti.elapsed().as_secs_f64() * 1e3);
        }
        n += 1;
        rss_peak = rss_peak.max(rss_mb());
        if let Some(rest) = interval.checked_sub(cycle.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let cpu = cpu0.elapsed().as_secs_f64();
    let med = |v: &mut Vec<f64>| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let summary = serde_json::json!({
        "mode": mode, "requested": [w, h], "actual": [actual.0, actual.1],
        "interval_s": interval.as_secs_f64(), "samples": n, "detected": detected,
        "inference": infer, "flush_frames": flush, "fps_requested": fps,
        "model_load_ms": load_ms,
        "capture_ms_median": med(&mut capture_ms), "inference_ms_median": med(&mut infer_ms),
        "process_cpu_pct_one_core": 100.0 * cpu / wall,
        "process_cpu_pct_all_cores": 100.0 * cpu / wall / std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1) as f64,
        "rss_mb_start": rss_start, "rss_mb_after_model": rss_model, "rss_mb_peak": rss_peak,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd.as_str() {
        "image" => run_image(&args),
        "camera" => run_camera(&args),
        "cameras" => list_cameras(),
        _ => {
            eprintln!("usage: hunch-spike image <file> | camera [--index N] [--mode keep|reopen] ...");
            std::process::exit(2);
        }
    }
}
