# main.py (axis fix + 4-section layout + raw tabs)
# -*- coding: utf-8 -*-
import io, math
import streamlit as st
import pandas as pd
import altair as alt

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="극지식물 실험", layout="wide")
st.title("🌱 극지식물 최적 EC 농도 실험 대시보드")
st.subheader("4개교 공동 실험 결과 분석")

alt.data_transformers.disable_max_rows()

SCHOOL_KEYS = ["송도고", "하늘고", "아라고", "동산고"]
EC_MAP = {"송도고":1, "하늘고":2, "아라고":4, "동산고":8}
COLOR_MAP = {"송도고":"#8bb8ff", "하늘고":"#88d4a9", "아라고":"#ffd66b", "동산고":"#ff9b9b"}

# --------------------------
# Sidebar: upload & mapping
# --------------------------
with st.sidebar:
    st.header("📁 파일 업로드")
    env_files = st.file_uploader("환경 CSV 4개", type=["csv"], accept_multiple_files=True,
                                 help="각 학교별 CSV 1개 (timestamp, temperature, humid, ec, ph, co2)")
    growth_file = st.file_uploader("생육 결과 엑셀(.xlsx)", type=["xlsx"],
                                   help="시트명: 송도고/하늘고/아라고/동산고")

def infer_school(name: str):
    low = name.lower()
    if "송도" in low: return "송도고"
    if "하늘" in low: return "하늘고"
    if "아라"  in low: return "아라고"
    if "동산" in low: return "동산고"
    return None

if env_files:
    st.sidebar.divider()
    st.sidebar.caption("🔗 CSV ↔ 학교 매핑 (자동 실패 시 직접 지정)")
    if "env_sel" not in st.session_state: st.session_state.env_sel = {}
    for f in env_files:
        guess = infer_school(f.name) or ""
        st.session_state.env_sel[f.name] = st.sidebar.selectbox(
            f"파일: {f.name}", [""]+SCHOOL_KEYS,
            index=([""]+SCHOOL_KEYS).index(guess) if guess in SCHOOL_KEYS else 0,
            key=f"sel_{f.name}"
        )

# --------------------------
# Loader (returns combined + RAW dicts)
# --------------------------
@st.cache_data(show_spinner=True)
def load_from_uploads(env_meta, xlsx_bytes):
    """
    env_meta: [(school, file_bytes), ...]
    return:
      combined_df: 요약(학교별 평균)
      raw_env:  dict[school] = 환경 DataFrame(원본)
      raw_growth: dict[school] = 생육 DataFrame(원본)
    """
    raw_env, raw_growth = {}, {}

    # ENV
    env_rows = []
    for school, fb in env_meta:
        bio = io.BytesIO(fb)
        try: df = pd.read_csv(bio, encoding="utf-8")
        except Exception:
            bio.seek(0); df = pd.read_csv(bio, encoding="cp949")
        raw_env[school] = df.copy()

        cols = {c.lower(): c for c in df.columns}
        need = ["temperature","humid","ec","ph"]
        for n in need:
            if n not in cols: raise ValueError(f"[{school}] 환경 CSV 칼럼 누락: {n}")

        t = pd.to_numeric(df[cols["temperature"]], errors="coerce").dropna()
        h = pd.to_numeric(df[cols["humid"]], errors="coerce").dropna()
        e = pd.to_numeric(df[cols["ec"]], errors="coerce").dropna()
        p = pd.to_numeric(df[cols["ph"]], errors="coerce").dropna()
        if len(p) and p.mean() > 100: p = p/100.0  # pH /100 보정

        env_rows.append({
            "학교": school,
            "평균 온도": t.mean() if len(t) else math.nan,
            "평균 습도": h.mean() if len(h) else math.nan,
            "평균 EC(측정)": e.mean() if len(e) else math.nan,
            "평균 pH": p.mean() if len(p) else math.nan
        })
    env_df = pd.DataFrame(env_rows)

    # GROWTH
    if xlsx_bytes is None: raise ValueError("생육 엑셀(.xlsx) 업로드 필요")
    bio = io.BytesIO(xlsx_bytes)

    g_rows = []
    for s in SCHOOL_KEYS:
        gdf = pd.read_excel(bio, sheet_name=s); bio.seek(0)
        raw_growth[s] = gdf.copy()

        if "생중량(g)" in gdf.columns:  # 아라고
            w = pd.to_numeric(gdf["생중량(g)"], errors="coerce").mean()
            l = math.nan; leaf = math.nan
        else:
            for c in ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]:
                if c not in gdf.columns: raise ValueError(f"[{s}] 생육 칼럼 누락: {c}")
            w = pd.to_numeric(gdf["지상부 생중량(g)"], errors="coerce").mean()
            l = pd.to_numeric(gdf["지상부 길이(cm)"], errors="coerce").mean()
            leaf = pd.to_numeric(gdf["잎 수(장)"], errors="coerce").mean()

        g_rows.append({"학교": s, "평균 생중량(g)": w, "평균 길이(cm)": l, "평균 잎 수": leaf})

    g = pd.DataFrame(g_rows)
    g["EC(설정)"] = g["학교"].map(EC_MAP)

    combined = pd.merge(g, env_df, on="학교", how="left")
    combined["color"] = combined["학교"].map(COLOR_MAP)
    return combined, raw_env, raw_growth

# Build env_meta
env_meta = []
if env_files:
    used = set()
    for f in env_files:
        sch = st.session_state.env_sel.get(f.name) or infer_school(f.name)
        if sch and sch not in used:
            env_meta.append((sch, f.getvalue())); used.add(sch)

data = raw_env = raw_growth = None
err = None
if env_meta and growth_file is not None:
    try:
        data, raw_env, raw_growth = load_from_uploads(env_meta, growth_file.getvalue())
    except Exception as e:
        err = str(e)

if not env_files or growth_file is None:
    st.info("좌측 사이드바에서 **환경 CSV(최대 4개)** 와 **생육 엑셀(.xlsx)** 을 업로드하세요. 파일명에 ‘송도/하늘/아라/동산’이 있으면 자동 매핑합니다.")
elif err:
    st.error(f"데이터 로드 오류: {err}")
elif data is None:
    st.warning("CSV ↔ 학교 매핑을 완료해 주세요.")
else:
    # --------------------------
    # Sidebar filters
    # --------------------------
    with st.sidebar:
        st.divider()
        st.header("🔬 데이터 필터")
        school_opts = ["전체","송도고(EC1)","하늘고(EC2)","아라고(EC4)","동산고(EC8)"]
        sel_schools = st.multiselect("학교 선택(복수 가능)", school_opts, default=["전체"])
        env_opts = ["온도","습도","EC","pH"]
        sel_env = st.multiselect("환경 변수", env_opts, default=["온도","습도","EC"])
        metric_opts = ["지상부 생중량","잎 수","지상부 길이"]
        sel_metric = st.selectbox("생육 지표", metric_opts, ind
