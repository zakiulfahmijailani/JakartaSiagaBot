import os, re, io, math, time, json, zipfile, asyncio, unicodedata
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats

import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx

from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Telegram
from telegram import Update, InputFile
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# --------------------------
# ENV & KONFIGURASI
# --------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
JAKARTA_SIM_THRESHOLD = float(os.getenv("JAKARTA_SIM_THRESHOLD", "0.55"))
INSIDE_RATIO_THRESHOLD = float(os.getenv("INSIDE_RATIO_THRESHOLD", "0.80"))

DATA_DIR = Path("./data"); DATA_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR  = Path("./out"); OUT_DIR.mkdir(exist_ok=True, parents=True)

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_IDN_shp.zip"
WORLDPOP_URLS = [
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IDN/idn_ppp_2020_1km_Aggregated.tif",
    "https://data.worldpop.org/GIS/Population/Global_2020_1km/IDN/idn_ppp_2020_1km_Aggregated.tif",
]

SERVICE_TAGS = {
    "ambulance": [{"emergency": "ambulance_station"}, {"amenity": "ambulance_station"}],
    "hospital":  [{"amenity": "hospital"}],
    "fire":      [{"amenity": "fire_station"}, {"building": "fire_station"}, {"emergency": "fire_station"}],
    "police":    [{"amenity": "police"}],
}
ADEQUACY_STD = {"ambulance": 50000, "hospital": 100000, "fire": 30000, "police": 90000}
SERVICE_ALIASES = {
    "ambulance": ["ambulans", "ambulance", "pos ambulan", "pos ambulans"],
    "hospital":  ["rs", "rumah sakit", "hospital", "rsud", "rsu"],
    "fire":      ["pemadam", "damkar", "fire", "fire station", "pos pemadam", "dinas pemadam"],
    "police":    ["polisi", "police", "pos polisi", "kantor polisi"],
}

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 180
ox.settings.overpass_rate_limit = True

# Executor untuk offload komputasi berat
EXECUTOR = ThreadPoolExecutor(max_workers=2)

# --------------------------
# UTIL
# --------------------------
def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return re.sub(r"\s+"," ", s.strip().lower())

def ensure_latlon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(4326) if (gdf.crs is None or gdf.crs.to_epsg()!=4326) else gdf

def download(url: str, dst: Path) -> Path:
    dst = Path(dst)
    if dst.exists(): return dst
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
    return dst

def try_download_any(urls, dst_dir) -> Path:
    last_err = None
    for u in urls:
        try:
            p = download(u, Path(dst_dir) / Path(u).name)
            return p
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Gagal unduh semua URL. Terakhir: {last_err}")

# --------------------------
# DATA: GADM L3 (DKI) + WorldPop
# --------------------------
def read_gdf(path: Path) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, engine="pyogrio")
    except Exception:
        return gpd.read_file(path)

def load_admin_and_population():
    # GADM
    gadm_zip = download(GADM_URL, DATA_DIR/"gadm41_IDN_shp.zip")
    with zipfile.ZipFile(gadm_zip,"r") as z: z.extractall(DATA_DIR)
    shp = list(DATA_DIR.glob("**/gadm41_IDN_3.shp"))
    if not shp: raise FileNotFoundError("gadm41_IDN_3.shp tidak ditemukan")
    adm3 = ensure_latlon(read_gdf(shp[0]))
    adm3_jkt = adm3[adm3["NAME_1"].str.contains("jakarta", case=False, na=False)].copy().reset_index(drop=True)
    adm3_jkt["district"] = adm3_jkt["NAME_3"].astype(str)

    # WorldPop 1km COUNTS 2020
    wp_tif = try_download_any(WORLDPOP_URLS, DATA_DIR)
    dki_union = unary_union(adm3_jkt.geometry)
    with rasterio.open(wp_tif) as src:
        out_img, out_transform = mask(src, [dki_union.__geo_interface__], crop=True, filled=True)
        out_meta = src.meta.copy()
        out_meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform})
    clipped = DATA_DIR / "worldpop_dki_clip.tif"
    with rasterio.open(clipped,"w",**out_meta) as dst: dst.write(out_img)
    zs = zonal_stats(adm3_jkt, clipped, stats=["sum"], all_touched=True, nodata=None)
    adm3_jkt["population"] = [z["sum"] if z["sum"] is not None else 0.0 for z in zs]
    pop_lut = {normalize_name(d): float(p) for d,p in zip(adm3_jkt["district"], adm3_jkt["population"])}
    return adm3_jkt, pop_lut, dki_union

ADM3_JKT, POP_LUT, DKI_UNION = load_admin_and_population()

# --------------------------
# RAG-Lite FAISS (district matcher + score)
# --------------------------
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
DISTRICT_NAMES = ADM3_JKT["district"].tolist()
EMB = MODEL.encode(DISTRICT_NAMES, normalize_embeddings=True).astype("float32")
INDEX = faiss.IndexFlatIP(EMB.shape[1]); INDEX.add(EMB)

def match_district_with_scores(text, k=8):
    q = MODEL.encode([text], normalize_embeddings=True).astype("float32")
    D, I = INDEX.search(q, k)
    cands = [DISTRICT_NAMES[i] for i in I[0]]
    sims  = D[0].tolist()
    best_sim = float(sims[0]) if sims else 0.0
    return cands, sims, best_sim

# --------------------------
# OSM Retrieval
# --------------------------
def _filters_for(service):
    fs = []
    for tag in SERVICE_TAGS[service]:
        for k, v in tag.items():
            fs.append({k: v if isinstance(v, list) else [v]})
    return fs

def _normalize_osmnx_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    import pandas as pd
    gdf = gdf.copy()
    if isinstance(gdf.index, pd.MultiIndex):
        idx_names = list(gdf.index.names)
        gdf = gdf.reset_index()
        if "osmid" not in gdf.columns:
            last = idx_names[-1] if idx_names and idx_names[-1] in gdf.columns else "index"
            gdf = gdf.rename(columns={last: "osmid"})
    else:
        if gdf.index.name and gdf.index.name != "osmid":
            gdf = gdf.reset_index().rename(columns={gdf.index.name: "osmid"})
        else:
            gdf = gdf.reset_index().rename(columns={"index":"osmid"})
    if "name" not in gdf.columns:
        for alt in ["official_name","alt_name","brand","operator","ref"]:
            if alt in gdf.columns:
                gdf["name"] = gdf[alt].astype(str); break
        if "name" not in gdf.columns: gdf["name"] = None
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.drop_duplicates(subset=["osmid"])
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.centroid if (geom is not None and not isinstance(geom, Point)) else geom)
    return gdf[["osmid","name","geometry"]]

def fetch_osm_facilities(polygon, service):
    gdfs = []
    for f in _filters_for(service):
        try:
            raw = ox.features_from_polygon(polygon, tags=f)
            if len(raw)==0: continue
            gdfs.append(_normalize_osmnx_gdf(raw))
        except Exception as e:
            print("OSM fail:", f, "->", e)
    if not gdfs:
        return gpd.GeoDataFrame(columns=["osmid","name","geometry"], geometry="geometry", crs="EPSG:4326")
    g = pd.concat(gdfs, ignore_index=True).drop_duplicates(subset=["osmid"])
    return gpd.GeoDataFrame(g, geometry="geometry", crs="EPSG:4326")

# --------------------------
# Adequacy & helpers
# --------------------------
def adequacy_metrics(population, service, actual):
    std = ADEQUACY_STD[service]
    need = population/std if std>0 else np.nan
    adequacy = (actual/need*100.0) if need>0 else 0.0
    status = "Surplus" if adequacy>=100 else ("Moderate" if adequacy>=50 else "Critical")
    return need, adequacy, status

def is_within_dki(poly, ratio_threshold=INSIDE_RATIO_THRESHOLD):
    try:
        g_poly = gpd.GeoSeries([poly], crs=4326).to_crs(3857)
        g_dki  = gpd.GeoSeries([DKI_UNION], crs=4326).to_crs(3857)
        inter  = g_poly.intersection(g_dki.iloc[0]).area.values[0]
        area   = g_poly.area.values[0]
        return (inter/area) >= ratio_threshold if area>0 else False
    except Exception:
        return False

# --------------------------
# LLM via OpenRouter (planner+report)
# --------------------------
def call_openrouter(messages, model=OPENROUTER_MODEL, temperature=0, max_tokens=256, timeout=45):
    if not OPENROUTER_API_KEY: raise RuntimeError("OPENROUTER_API_KEY belum di-set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your.app", "X-Title": "SiagaJakarta-TelegramBot"
    }
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def extract_json(text: str):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m: return None
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        s = re.sub(r",\s*}", "}", s); s = re.sub(r",\s*]", "]", s)
        try: return json.loads(s)
        except Exception: return None

def llm_parse_query_gate(query_text, candidate_districts):
    svc_allowed = ["ambulance","hospital","fire","police"]
    sys_prompt = (
        "Anda Orchestrator GeoAI DKI. Keluarkan JSON {service,in_jakarta,district,oos_place}.\n"
        "Jika di DKI: in_jakarta=true dan district HARUS salah satu kandidat.\n"
        "Jika di luar: in_jakarta=false dan isi oos_place (nama tempat lengkap)."
    )
    user_prompt = f"Pertanyaan: {query_text}\nKandidat district: {candidate_districts}\nService valid: {svc_allowed}\n"
    out = call_openrouter(
        [{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
        temperature=0, max_tokens=220
    )
    return extract_json(out)

def geocode_polygon(place_query, buffer_m_if_point=1500):
    gdf = ox.geocode_to_gdf(place_query)
    gdf = ensure_latlon(gdf)
    geom = gdf.geometry.iloc[0]
    if not str(geom.geom_type).lower().endswith("polygon"):
        g3857 = gpd.GeoSeries([geom], crs=4326).to_crs(3857).buffer(buffer_m_if_point)
        geom = g3857.to_crs(4326).iloc[0]
    return geom

def llm_generate_report(ctx):
    d = ctx["district"]; s = ctx["service"]; pop = int(ctx["population"])
    need = ctx["need"]; actual = ctx["actual"]; adequacy = ctx["adequacy"]; status = ctx["status"]
    sys_prompt = ("Anda asisten teknis. Tulis 2–4 kalimat Bahasa Indonesia, ringkas, sertakan angka. "
                  "Jika kecukupan <100%, beri rekomendasi singkat.")
    user_prompt = (f"Layanan: {s}\nKecamatan/Area: {d}\nPopulasi: {pop}\n"
                   f"Kebutuhan: {need:.2f}\nAktual: {actual}\nKecukupan: {adequacy:.1f}\nStatus: {status}")
    out = call_openrouter(
        [{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
        temperature=0.2, max_tokens=220
    )
    return out.strip()

# --------------------------
# Orchestrator dengan Jakarta Gate
# --------------------------
@dataclass
class RunResult:
    ok: bool
    png_path: Path
    caption: str

def run_pipeline(query_text: str) -> RunResult:
    # 1) kandidat DKI via FAISS + skor
    cand, sims, best_sim = match_district_with_scores(query_text, k=8)
    # 2) planner LLM
    try:
        parsed = llm_parse_query_gate(query_text, cand) or {}
    except Exception:
        parsed = {}
    if not parsed or ("service" not in parsed):
        # fallback heuristik minimal
        svc = next((k for k,als in SERVICE_ALIASES.items() if k in query_text.lower() or any(a in query_text.lower() for a in als)), None)
        district = cand[0]
        parsed = {"service": svc or "hospital", "in_jakarta": True, "district": district, "oos_place": None}

    service = parsed.get("service","hospital")
    in_jkt  = bool(parsed.get("in_jakarta", True))
    district= parsed.get("district")
    oos_place = parsed.get("oos_place")

    # 3) Gating keputusan akhir
    use_oos = (not in_jkt) or (best_sim < JAKARTA_SIM_THRESHOLD)
    if not use_oos:
        if district not in cand: district = cand[0]
        row = ADM3_JKT[ADM3_JKT["district"]==district].iloc[0]
        polygon = row.geometry
        pop = POP_LUT[normalize_name(district)]
        gdf = fetch_osm_facilities(polygon, service)
        need, adequacy, status = adequacy_metrics(pop, service, len(gdf))
        label = district
    else:
        place_query = oos_place if (oos_place and isinstance(oos_place,str) and oos_place.strip()) else query_text
        try:
            polygon = geocode_polygon(place_query)
            gdf = fetch_osm_facilities(polygon, service)
        except Exception:
            # fallback ke DKI terbaik
            row = ADM3_JKT[ADM3_JKT["district"]==cand[0]].iloc[0]
            polygon = row.geometry
            gdf = fetch_osm_facilities(polygon, service)
            place_query = cand[0]
        # Untuk OOS, demo: populasi tidak dihitung (0), kecukupan N/A (opsional bisa dihitung jika ada raster global).
        pop = 0
        need, adequacy, status = (np.nan, np.nan, "N/A")
        label = place_query

    # 4) Narasi LLM
    ctx = {"district": label, "service": service, "facilities_gdf": gdf, "polygon": polygon,
           "population": pop, "actual": int(len(gdf)), "need": need, "adequacy": adequacy, "status": status}
    try:
        insight = llm_generate_report(ctx)
    except Exception:
        insight = (f"{service.capitalize()} di {label}: aktual={len(gdf)}. (Narasi LLM gagal, gunakan ringkasan singkat.)")

    # 5) Peta statis + basemap OSM → PNG
    png_path = OUT_DIR / f"map_{int(time.time())}_{service}_{re.sub('[^A-Za-z0-9_]+','_',label)}.png"
    plot_static_map_osm(ctx, insight, png_path)

    # 6) Caption singkat (Telegram batas caption 1024; sisanya kirim pesan terpisah)
    caption = f"{service.capitalize()} — {label}\nAktual: {len(gdf)}; Kecukupan: {('N/A' if (adequacy is np.nan) else f'{adequacy:.1f}%')}"
    return RunResult(ok=True, png_path=png_path, caption=caption)

# --------------------------
# Plot statis + basemap OSM
# --------------------------
def plot_static_map_osm(ctx_obj, insight_text, out_png_path: Path, dpi=200):
    poly = ctx_obj["polygon"]; pts = ctx_obj["facilities_gdf"]
    district = ctx_obj["district"]; service = ctx_obj["service"]
    pop = int(ctx_obj["population"]); need = ctx_obj["need"]
    actual = ctx_obj["actual"]; adequacy = ctx_obj["adequacy"]; status = ctx_obj["status"]

    gpoly = gpd.GeoSeries([poly], crs=4326).to_crs(3857)
    pts3857 = pts.to_crs(3857) if len(pts)>0 else pts

    fig, ax = plt.subplots(figsize=(7.5,7.5), dpi=dpi)
    xmin,ymin,xmax,ymax = gpoly.total_bounds
    padx=(xmax-xmin)*0.10; pady=(ymax-ymin)*0.10
    ax.set_xlim(xmin-padx, xmax+padx); ax.set_ylim(ymin-pady, ymax+pady)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=6)
    gpoly.boundary.plot(ax=ax, linewidth=1.3, zorder=5)
    gpoly.plot(ax=ax, color="#DDEAF6", alpha=0.35, edgecolor="black", zorder=4)
    if len(pts)>0:
        pts3857.plot(ax=ax, markersize=18, marker="o", zorder=6)

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    title = f"{service.capitalize()} — {district}"
    subtitle = f"Populasi: {pop:,} | Kebutuhan: {('N/A' if (need is np.nan) else f'{need:.2f}')} | Aktual: {actual} | Kecukupan: {('N/A' if (adequacy is np.nan) else f'{adequacy:.1f}%')} ({status})"
    ax.set_title(title+"\n"+subtitle, loc="left", fontsize=12, pad=10)

    plt.subplots_adjust(bottom=0.20)
    foot = ("Insight LLM: " + (insight_text or "")).strip()
    plt.figtext(0.01, 0.05, foot, ha="left", va="bottom", fontsize=9, wrap=True)

    out_png_path = Path(out_png_path); out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight"); plt.close(fig)

# --------------------------
# TELEGRAM HANDLERS
# --------------------------
WELCOME = (
    "Halo! Saya *SiagaJakarta Bot*.\n"
    "Ketik pertanyaan seperti:\n"
    "• `pemadam Jatinegara`\n"
    "• `polisi Setiabudi`\n"
    "• `rumah sakit Tebet`\n\n"
    "Bot akan mengembalikan peta statis + ringkasan.\n"
    "_Catatan: proses bisa 10–30 dtk tergantung beban & jaringan._"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    if not text:
        return

    # Tampilkan typing sementara pipeline jalan di thread pool
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    loop = asyncio.get_running_loop()
    try:
        result: RunResult = await loop.run_in_executor(EXECUTOR, run_pipeline, text)
        # Kirim foto + caption
        with open(result.png_path, "rb") as f:
            await context.bot.send_photo(chat_id=chat_id, photo=InputFile(f), caption=result.caption)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Maaf, terjadi kesalahan: {e}")

# --------------------------
# MAIN (long-polling)
# --------------------------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN belum di-set di .env")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_text))
    print("Bot siap. Tekan Ctrl+C untuk berhenti.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
