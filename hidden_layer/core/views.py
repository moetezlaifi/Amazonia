import json, time, random
from pathlib import Path
from threading import Thread

from django.urls import reverse
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_protect

from shapely.geometry import shape as shapely_shape
from django.utils.timezone import now
from core.pipeline.structure import run_dino_structure_anomaly
from core.pipeline.structure import run_dino_structure_anomaly
from core.pipeline.dinov2 import run_dinov2_hotspots


def index(request):
    return render(request, "core/index.html")


def map_view(request):
    job_id = request.GET.get("job_id")
    context = {"job_id": job_id, "has_result": False, "status": None, "meta": None, "meta_json": "{}"}

    if job_id:
        run_dir = Path(settings.MEDIA_ROOT) / job_id
        status_path = run_dir / "status.json"
        meta_path = run_dir / "meta.json"

        if status_path.exists():
            try:
                context["status"] = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                context["status"] = {"state": "error", "message": "Bad status.json"}

        if meta_path.exists():
            try:
                context["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
                context["meta_json"] = json.dumps(context["meta"])
                context["has_result"] = True
            except Exception:
                context["has_result"] = False

    return render(request, "core/map.html", context)


@require_POST
@csrf_protect
def analyze(request):
    aoi_geojson = json.loads(request.POST["aoi_geojson"])
    mode = request.POST.get("mode", "original")  # Get the analysis mode

    job_id = time.strftime("%Y%m%d_%H%M%S_") + hex(int(time.time() * 1e6))[-4:]
    run_dir = Path(settings.MEDIA_ROOT) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "aoi": aoi_geojson,
        "cloud_pct": 20,
        "days_back": 30,
        "mode": mode,
    }
    (run_dir / "payload.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    _write_status(run_dir, "queued", "Queued")

    def worker():
        try:
            _write_status(run_dir, "running", f"Running {mode} analysis...")

            # Simulated outputs for demo
            bbox, original_hotspots = _demo_hotspots_from_aoi(aoi_geojson, n=6)

            final_hotspots = []

            if mode == "original":
                final_hotspots = original_hotspots
            elif mode == "dinov2":
                final_hotspots = _demo_dinov2_hotspots(aoi_geojson)
            elif mode == "merge":
                dino_hotspots = _demo_dinov2_hotspots(aoi_geojson)
                final_hotspots = _merge_hotspots(original_hotspots, dino_hotspots)

            meta = {
                "bbox": bbox,
                "mode": mode,  # Store the mode in meta for frontend display
                "cloud_pct": 20,
                "days_back": 30,
                "data_source": {"source": "offline_demo"},
                "hotspots": final_hotspots,
                "disclaimer": "Decision support only. Validate with additional sources.",
            }

            (run_dir / "meta.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            _write_status(run_dir, "done", "Done")

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            _write_status(run_dir, "error", "Error", error=str(e))

    Thread(target=worker, daemon=True).start()

    return redirect(f"{reverse('map')}?job_id={job_id}")


def _write_status(run_dir: Path, state: str, message: str = "", error: str = ""):
    payload = {"state": state, "message": message, "error": error, "ts": now().isoformat()}
    (run_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _demo_hotspots_from_aoi(aoi_geojson: dict, n: int = 6):
    geom = shapely_shape(aoi_geojson["geometry"] if "geometry" in aoi_geojson else aoi_geojson)
    minx, miny, maxx, maxy = geom.bounds

    hotspots = []
    for _ in range(n):
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        score = random.uniform(0.65, 0.95)
        priority = "High" if score >= 0.85 else ("Medium" if score >= 0.70 else "Low")
        hotspots.append({
            "lon": lon,
            "lat": lat,
            "score": score,
            "priority": priority,
            "reasons": ["spectral anomaly"],
        })

    hotspots.sort(key=lambda h: h["score"], reverse=True)
    return [minx, miny, maxx, maxy], hotspots
@require_GET
def job_status(request, job_id: str):
    run_dir = Path(settings.MEDIA_ROOT) / job_id
    if not run_dir.exists():
        return JsonResponse({"state": "missing"}, status=404)

    status_path = run_dir / "status.json"
    meta_path = run_dir / "meta.json"

    if status_path.exists():
        st = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        st = {"state": "queued", "message": "Queued"}

    # if meta exists, treat as done
    if st.get("state") not in ("done", "error") and meta_path.exists():
        st = {"state": "done", "message": "Done"}

    return JsonResponse(st)

def _demo_dinov2_hotspots(aoi_geojson: dict, n: int = 4):
    geom = shapely_shape(
        aoi_geojson["geometry"] if "geometry" in aoi_geojson else aoi_geojson
    )
    minx, miny, maxx, maxy = geom.bounds

    hotspots = []

    for _ in range(n):
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        score = random.uniform(0.7, 0.99)
        priority = "High" if score >= 0.9 else (
            "Medium" if score >= 0.8 else "Low"
        )

        hotspots.append({
            "lon": lon,
            "lat": lat,
            "score": score,
            "priority": priority,
            "reasons": ["DINOv2 structural score"],
            "source": "dinov2",
        })

    return hotspots


def _merge_hotspots(original, dino):
    merged = []

    for h in original:
        h["reasons"].append("Found by Original")
        merged.append(h)

    for d in dino:
        # Simple spatial proximity check (very basic for demo)
        is_duplicate = False

        for m in merged:
            dist = (
                (m["lon"] - d["lon"]) ** 2 +
                (m["lat"] - d["lat"]) ** 2
            ) ** 0.5

            if dist < 0.001:  # Roughly 100m
                m["reasons"].append("Confirmed by DINOv2")
                m["score"] = max(m["score"], d["score"])
                m["source"] = "merged"
                is_duplicate = True
                break

        if not is_duplicate:
            d["reasons"].append("Found by DINOv2")
            merged.append(d)

    merged.sort(key=lambda x: x["score"], reverse=True)

    return merged
