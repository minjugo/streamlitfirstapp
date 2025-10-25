# main.py  (UTF-8)
# Streamlit 대시보드 - 4개교 극지식물 EC 실험 (하이브리드 로더: 레포 파일 자동탐색 + 업로드 대체)
import os, glob, io, math
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="극지식물 실험", layout="wide")
st.title("🌱 극지식물 최적 EC 농도 실험 대시보드")
st.subheader("4개교 공동 실험 결과 분석")
alt.data_transformers.disable_max_rows()

SCHOOLS = ["송도고","하늘고","아라고","동산고"]
EC_MAP = {"송도고":1,"하늘고":2,"아라고":4,"동산고":8}
COLOR = {"송도고":"#8bb8ff","하늘고":"#88d4a9","아라고":"#ffd66b","동산고":"#ff9b9b"}

# ---------- 파일 자동탐색 ----------
def find_first(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

def guess_school_from_name(name:str):
    key = name.lower()
    if "송도" in key: return "송도고"
    if "하늘" in key: return "하늘고"
    if "아라" in key:  return "아라고"
    if "동산" in key: return "동산고"
    return None

def read_env_csv(path_or_bytes, school_hint):
    # CSV 읽기 (파일경로 또는 업로드 바이트)
    if isinstance(path_or_bytes, (bytes,bytearray)):
        bio = io.BytesIO(path_or_bytes)
        try: df = pd.read_csv(bio, encoding="utf-8")
        except: 
            bio.seek(0); df = pd.read_csv(bio, encoding="cp949")
    else:
        try: df = pd.read_csv(path_or_bytes, encoding="utf-8")
        except: df = pd.read_csv(path_or_bytes, encoding="cp949")
    cols = {c.lower():c for c in df.columns}
    need = ["temperature","humid","ec","ph"]
    for n in need:
        if n not in cols: raise ValueError(f"[{school_hint}] 환경 CSV 칼럼 누락: {n}")
    t = pd.to_numeric(df[cols["temperature"]], errors="coerce").dropna()
    h = pd.to_numeric(df[cols["humid"]], errors="coerce").dropna()
    e = pd.to_numeric(df[cols["ec"]], errors="coerce").dropna()
    p = pd.to_numeric(df[cols["ph"]], errors="coerce").dropna()
    if len(p) and p.mean()>100: p = p/100.0
    return {
        "학교": school_hint,
        "평균 온도": t.mean() if len(t) else math.nan,
        "평균 습도": h.mean() if len(h) else math.nan,
        "평균 EC(측정)": e.mean() if len(e) else math.nan,
        "평균 pH": p.mean() if len(p) else math.nan
    }

def read_growth_excel(bytes_or_path, school):
    # 엑셀 시트 4개 중 하나 읽음
    if isinstance(bytes_or_path, (bytes,bytearray)):
        bio = io.BytesIO(bytes_or_path)
        gdf = pd.read_excel(bio, sheet_name=school)
    else:
        gdf = pd.read_excel(bytes_or_path, sheet_name=school)
    if "생중량(g)" in gdf.columns:  # 아라고 형태
        w = pd.to_numeric(gdf["생중량(g)"], errors="coerce").mean()
        L = math.nan; leaf = math.nan
    else:
        need = ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]
        for c in need:
            if c not in gdf.columns: raise ValueError(f"[{school}] 생육 칼럼 누락: {c}")
        w = pd.to_numeric(gdf["지상부 생중량(g)"], errors="coerce").mean()
        L = pd.to_numeric(gdf["지상부 길이(cm)"], errors="coerce").mean()
        leaf = pd.to_numeric(gdf["잎 수(장)"], errors="coerce").mean()
    return {"학교":school,"평균 잎 수":leaf,"평균 길이(cm)":L,"평균 생중량(g)":w}

def read_growth_csv(path_or_bytes, school):
    # 학교별 생육 CSV 지원 (칼럼명 동일 가정)
    if isinstance(path_or_bytes, (bytes,bytearray)):
        bio = io.BytesIO(path_or_bytes)
        try: df = pd.read_csv(bio, encoding="utf-8")
        except: 
            bio.seek(0); df = pd.read_csv(bio, encoding="cp949")
    else:
        try: df = pd.read_csv(path_or_bytes, encoding="utf-8")
        except: df = pd.read_csv(path_or_bytes, encoding="cp949")
    if "생중량(g)" in df.columns:  # 아라고 형태
        w = pd.to_numeric(df["생중량(g)"], errors="coerce").mean()
        L = math.nan; leaf = math.nan
    else:
        need = ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]
        for c in need:
            if c not in df.columns: raise ValueError(f"[{school}] 생육 CSV 칼럼 누락: {c}")
        w = pd.to_numeric(df["지상부 생중량(g)"], errors="coerce").mean()
        L = pd.to_numeric(df["지상부 길이(cm)"], errors="coerce").mean()
        leaf = pd.to_numeric(df["잎 수(장)"], errors="coerce").mean()
    return {"학교":school,"평균 잎 수":leaf,"평균 길이(cm)":L,"평균 생중량(g)":w}

# ---------- 1) 레포에서 자동 로드 시도 ----------
def auto_load_from_repo():
    # 환경 CSV 자동 수집
    env_rows = []
    for path in sorted(glob.glob("*.csv")):
        school = guess_school_from_name(os.path.basename(path))
        # 환경 CSV 추정: '환경' 단어 포함
        if school and ("환경" in path or "env" in path.lower()):
            try:
                env_rows.append(read_env_csv(path, school))
            except Exception:
                pass
    env_df = pd.DataFrame(env_rows)

    # 생육: 엑셀 우선, 없으면 생육 CSV 사용
    growth_df = pd.DataFrame(columns=["학교","평균 잎 수","평균 길이(cm)","평균 생중량(g)"])
    xlsx = find_first(["*생육*데이터*.xlsx","*4개교*생육*.xlsx","*.xlsx"])
    if xlsx:
        rows=[]
        for s in SCHOOLS:
            try: rows.append(read_growth_excel(xlsx, s))
            except Exception: pass
        growth_df = pd.DataFrame(rows)
    else:
        # 학교별 생육 CSV 탐색(파일명에 '생육' 포함)
        rows=[]
        for path in sorted(glob.glob("*.csv")):
            school = guess_school_from_name(os.path.basename(path))
            if school and ("생육" in path or "growth" in path.lower()):
                try: rows.append(read_growth_csv(path, school))
                except Exception: pass
        growth_df = pd.DataFrame(rows)

    if env_df.empty and growth_df.empty:
        return None  # 레포에 유의미 파일 없음
    # EC/색상 부여 및 병합(있 는 학교만)
    if not growth_df.empty: growth_df["EC(설정)"] = growth_df["학교"].map(EC_MAP)
    combined = pd.merge(growth_df, env_df, on="학교", how="left")
    if not combined.empty: combined["color"] = combined["학교"].map(COLOR)
    return combined

# ---------- 2) 업로드 대체 루트 ----------
def ui_upload_and_build():
    with st.sidebar:
        st.header("📁 파일 업로드 (대체)")
        env_files = st.file_uploader("환경 CSV (여러 개 가능)", type=["csv"], accept_multiple_files=True)
        growth_xlsx = st.file_uploader("생육 엑셀(.xlsx, 선택)", type=["xlsx"])
        growth_csvs = st.file_uploader("또는 생육 CSV(학교별 여러 개)", type=["csv"], accept_multiple_files=True)

    if not env_files and not growth_xlsx and not growth_csvs:
        st.info("레포에서 파일을 못 찾았고, 업로드도 아직 없습니다. 좌측에서 파일을 업로드하세요.")
        return None

    # 환경
    env_rows=[]
    if env_files:
        for f in env_files:
            school = guess_school_from_name(f.name) or st.selectbox(f"환경 CSV 매핑: {f.name}", [""]+SCHOOLS, key=f"env_{f.name}")
            if not school: continue
            try: env_rows.append(read_env_csv(f.getvalue(), school))
            except Exception as e:
                st.warning(f"환경 CSV 로드 실패({f.name}): {e}")
    env_df = pd.DataFrame(env_rows)

    # 생육
    growth_rows=[]
    if growth_xlsx is not None:
        for s in SCHOOLS:
            try: growth_rows.append(read_growth_excel(growth_xlsx.getvalue(), s))
            except Exception: pass
    if growth_csvs:
        for f in growth_csvs:
            school = guess_school_from_name(f.name) or st.selectbox(f"생육 CSV 매핑: {f.name}", [""]+SCHOOLS, key=f"growth_{f.name}")
            if not school: continue
            try: growth_rows.append(read_growth_csv(f.getvalue(), school))
            except Exception as e:
                st.warning(f"생육 CSV 로드 실패({f.name}): {e}")
    growth_df = pd.DataFrame(growth_rows)

    if growth_df.empty and env_df.empty:
        st.warning("업로드한 파일에서 유효 데이터를 찾지 못했습니다.")
        return None

    if not growth_df.empty: growth_df["EC(설정)"] = growth_df["학교"].map(EC_MAP)
    combined = pd.merge(growth_df, env_df, on="학교", how="left")
    if not combined.empty: combined["color"] = combined["학교"].map(COLOR)
    return combined

# ---------- 데이터 준비 ----------
data = auto_load_from_repo()  # 1) 레포 자동탐색
if data is None:
    data = ui_upload_and_build()  # 2) 업로드 대체
if data is None or data.empty:
    st.stop()

# ---------- 사이드바 필터 ----------
with st.sidebar:
    st.header("🔬 데이터 필터")
    school_opts = ["전체"] + [f"{s}(EC{EC_MAP[s]})" for s in SCHOOLS]
    sel_sch = st.multiselect("학교 선택(복수)", school_opts, default=["전체"])
    env_opts = ["온도","습도","EC","pH"]
    sel_env = st.multiselect("환경 변수", env_opts, default=["온도","습도","EC"])
    metric_opts = ["지상부 생중량","잎 수","지상부 길이"]
    sel_metric = st.selectbox("생육 지표", metric_opts, index=0)

def normalize_school_filter(selected):
    if ("전체" in selected) or (not selected): return [s for s in SCHOOLS if s in set(data["학교"])]
    mp = {f"{s}(EC{EC_MAP[s]})":s for s in SCHOOLS}
    return [mp[x] for x in selected if x in mp]

use_schools = normalize_school_filter(sel_sch)
filtered = data[data["학교"].isin(use_schools)].copy()

# ---------- KPI ----------
col1,col2,col3,col4 = st.columns(4)
col1.metric("총 학교 수", f"{len(filtered):,}")
if "평균 생중량(g)" in filtered:
    avg_w = filtered["평균 생중량(g)"].mean()
    col2.metric("평균 생중량", f"{avg_w:.2f} g")
    best = filtered.loc[filtered["평균 생중량(g)"].idxmax()]
    col3.metric("최고 EC 농도 (생중량 기준)", f"EC {int(best['EC(설정)'])}")
    col4.metric("최고 생중량", f"{best['평균 생중량(g)']:.2f} g")
else:
    col2.metric("평균 생중량","-"); col3.metric("최고 EC","-"); col4.metric("최고 생중량","-")

st.markdown("---")

# ----------
