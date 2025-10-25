# main.py (final, no-duplicate-id + 4-section + axis fix)
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
COLOR_MAP = {"송도고":"#8bb8ff", "하늘고":"#88d4a9", "아라고":"#ffd66b", "동산고":"#ff9b9b"}  # 파스텔

# --------------------------
# Sidebar: upload & mapping
# --------------------------
with st.sidebar:
    st.header("📁 파일 업로드")
    # ① 업로더에 고유 key 지정
    env_files = st.file_uploader(
        "환경 CSV 4개",
        type=["csv"],
        accept_multiple_files=True,
        help="각 학교별 CSV 1개 (timestamp, temperature, humid, ec, ph, co2)",
        key="uploader_env_csvs"
    )
    growth_file = st.file_uploader(
        "생육 결과 엑셀(.xlsx)",
        type=["xlsx"],
        help="시트명: 송도고/하늘고/아라고/동산고",
        key="uploader_growth_xlsx"
    )

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
    if "env_sel" not in st.session_state:
        st.session_state.env_sel = {}
    # ② 매핑용 selectbox 키를 파일명 대신 인덱스 기반으로
    for i, f in enumerate(env_files):
        guess = infer_school(f.name) or ""
        st.session_state.env_sel[i] = st.sidebar.selectbox(
            f"파일: {f.name}",
            [""] + SCHOOL_KEYS,
            index=([""] + SCHOOL_KEYS).index(guess) if guess in SCHOOL_KEYS else 0,
            key=f"sel_env_{i}"   # 인덱스 기반 고정 키
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
        try:
            df = pd.read_csv(bio, encoding="utf-8")
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, encoding="cp949")
        raw_env[school] = df.copy()

        cols = {c.lower(): c for c in df.columns}
        need = ["temperature","humid","ec","ph"]
        for n in need:
            if n not in cols:
                raise ValueError(f"[{school}] 환경 CSV 칼럼 누락: {n}")

        t = pd.to_numeric(df[cols["temperature"]], errors="coerce").dropna()
        h = pd.to_numeric(df[cols["humid"]], errors="coerce").dropna()
        e = pd.to_numeric(df[cols["ec"]], errors="coerce").dropna()
        p = pd.to_numeric(df[cols["ph"]], errors="coerce").dropna()
        # pH 스케일 보정 (/100)
        if len(p) and p.mean() > 100:
            p = p / 100.0

        env_rows.append({
            "학교": school,
            "평균 온도": t.mean() if len(t) else math.nan,
            "평균 습도": h.mean() if len(h) else math.nan,
            "평균 EC(측정)": e.mean() if len(e) else math.nan,
            "평균 pH": p.mean() if len(p) else math.nan
        })
    env_df = pd.DataFrame(env_rows)

    # GROWTH
    if xlsx_bytes is None:
        raise ValueError("생육 엑셀(.xlsx) 업로드 필요")
    bio = io.BytesIO(xlsx_bytes)

    g_rows = []
    for s in SCHOOL_KEYS:
        gdf = pd.read_excel(bio, sheet_name=s)
        bio.seek(0)
        raw_growth[s] = gdf.copy()

        if "생중량(g)" in gdf.columns:  # 아라고: 총 생중량만
            w = pd.to_numeric(gdf["생중량(g)"], errors="coerce").mean()
            l = math.nan
            leaf = math.nan
        else:
            for c in ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]:
                if c not in gdf.columns:
                    raise ValueError(f"[{s}] 생육 칼럼 누락: {c}")
            w = pd.to_numeric(gdf["지상부 생중량(g)"], errors="coerce").mean()
            l = pd.to_numeric(gdf["지상부 길이(cm)"], errors="coerce").mean()
            leaf = pd.to_numeric(gdf["잎 수(장)"], errors="coerce").mean()

        g_rows.append({"학교": s, "평균 생중량(g)": w, "평균 길이(cm)": l, "평균 잎 수": leaf})

    g = pd.DataFrame(g_rows)
    g["EC(설정)"] = g["학교"].map(EC_MAP)

    combined = pd.merge(g, env_df, on="학교", how="left")
    combined["color"] = combined["학교"].map(COLOR_MAP)
    return combined, raw_env, raw_growth

# --------------------------
# Build env_meta (index-based)
# --------------------------
env_meta = []
if env_files:
    used = set()
    # ③ env_meta 구성도 인덱스 기반으로
    for i, f in enumerate(env_files):
        sch = st.session_state.env_sel.get(i) or infer_school(f.name)
        if sch and sch not in used:
            env_meta.append((sch, f.getvalue()))
            used.add(sch)

# --------------------------
# Load data
# --------------------------
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
        sel_metric = st.selectbox("생육 지표", metric_opts, index=0)

    def norm_school_filter(selected):
        if ("전체" in selected) or (not selected):
            return SCHOOL_KEYS
        mp = {"송도고(EC1)":"송도고","하늘고(EC2)":"하늘고","아라고(EC4)":"아라고","동산고(EC8)":"동산고"}
        return [mp[s] for s in selected if s in mp]

    use_schools = norm_school_filter(sel_schools)
    filtered = data[data["학교"].isin(use_schools)].copy()

    # --------------------------
    # KPI
    # --------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 학교 수", f"{len(filtered):,}")
    c2.metric("평균 생중량", f"{filtered['평균 생중량(g)'].mean():.2f} g")
    best = filtered.loc[filtered["평균 생중량(g)"].idxmax()]
    c3.metric("최고 EC 농도 (생중량 기준)", f"EC {int(best['EC(설정)'])}")
    c4.metric("최고 생중량", f"{best['평균 생중량(g)']:.2f} g")

    st.markdown("---")

    # --------------------------
    # Tabs (2 tabs) — 각 탭 2×2 섹션(총 4섹션)
    # --------------------------
    tab1, tab2 = st.tabs(["📊 생육 결과", "🌡️ 환경 분석"])

    # 공통 tidy
    tidy = filtered[["학교","EC(설정)","평균 생중량(g)","평균 잎 수","평균 길이(cm)","color"]].copy()

    # ---------- TAB 1: 생육 결과 (4 섹션) ----------
    with tab1:
        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        # 섹션 1: 차트 1 — EC vs 선택 지표 (꺾은선 + ★)
        metric_map = {"지상부 생중량":"평균 생중량(g)","잎 수":"평균 잎 수","지상부 길이":"평균 길이(cm)"}
        ycol = metric_map[sel_metric]
        ln_df = tidy.sort_values("EC(설정)").dropna(subset=[ycol])
        ln_df["is_max"] = (ln_df[ycol] == ln_df[ycol].max()) if not ln_df.empty else False

        axis_x = alt.Axis(title="EC (설정)", labelAngle=0, labelOverlap=False, labelPadding=8, labelLimit=200)
        axis_y = alt.Axis(title=sel_metric)

        line = alt.Chart(ln_df).mark_line(point=True, strokeWidth=2, color="#5b8def").encode(
            x=alt.X("EC(설정):O", sort=[1,2,4,8], axis=axis_x, scale=alt.Scale(padding=0.5)),
            y=alt.Y(f"{ycol}:Q", axis=axis_y),
            tooltip=[alt.Tooltip("학교:N"), alt.Tooltip("EC(설정):O"), alt.Tooltip(f"{ycol}:Q", format=".2f")]
        ).properties(title="차트 1 · EC vs 선택 지표", height=330)

        star = alt.Chart(ln_df[ln_df["is_max"]]).mark_point(shape="star", size=220, color="#ef4444").encode(
            x="EC(설정):O", y=f"{ycol}:Q"
        )

        with g1:
            st.altair_chart(line + star, use_container_width=True)

        # 섹션 2: 차트 2 — 학교별 TOP 4 (가로 막대)
        bar_df = tidy[["학교","color",ycol]].dropna().sort_values(ycol, ascending=False)
        bar = alt.Chart(bar_df).mark_bar(cornerRadius=6).encode(
            x=alt.X(f"{ycol}:Q", title=sel_metric),
            y=alt.Y("학교:N", sort="-x", title=None,
                    axis=alt.Axis(labelAngle=0, labelLimit=200)),
            color=alt.Color("학교:N",
                scale=alt.Scale(range=[data[data['학교']=='송도고']['color'].iloc[0],
                                       data[data['학교']=='하늘고']['color'].iloc[0],
                                       data[data['학교']=='아라고']['color'].iloc[0],
                                       data[data['학교']=='동산고']['color'].iloc[0]]),
                legend=None),
            tooltip=[alt.Tooltip("학교:N"), alt.Tooltip(f"{ycol}:Q", format=".2f")]
        )
        text = bar.mark_text(align='left', dx=6, color="#3a4762").encode(
            text=alt.Text(f"{ycol}:Q", format=".2f")
        )
        with g2:
            st.altair_chart((bar + text).properties(title="차트 2 · 학교별 TOP 4", height=330),
                            use_container_width=True)

        # 섹션 3: 차트 3 — 3지표 정규화 (0-100), x축 학교명 세로 회전 방지
        norm_cols = ["평균 생중량(g)","평균 잎 수","평균 길이(cm)"]
        ndf = tidy[["학교","color"] + norm_cols].copy()
        for c in norm_cols:
            cmax = ndf[c].max(skipna=True)
            ndf[c+"_점수"] = (ndf[c] / cmax * 100).where(pd.notna(ndf[c]), None)

        tnorm = ndf.melt(id_vars=["학교","color"], value_vars=[c+"_점수" for c in norm_cols],
                         var_name="지표", value_name="점수").dropna()

        chart3 = alt.Chart(tnorm).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("학교:N", title=None, sort=SCHOOL_KEYS,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=200)),
            y=alt.Y("점수:Q", title="정규화 점수(0-100)"),
            color=alt.Color("지표:N", scale=alt.Scale(range=["#9ec5fe","#a7f3d0","#fde68a"]))
        ).properties(title="차트 3 · 3가지 지표 종합", height=330)

        with g3:
            st.altair_chart(chart3, use_container_width=True)

        # 섹션 4: RAW · 생육(학교별 10행)
        with g4:
            st.markdown("**RAW 데이터 · 생육(학교별 10행)**")
            tabs = st.tabs(SCHOOL_KEYS)
            for i, sch in enumerate(SCHOOL_KEYS):
                with tabs[i]:
                    if sch in raw_growth:
                        st.dataframe(raw_growth[sch].head(10), use_container_width=True)
                    else:
                        st.info("해당 학교 생육 데이터가 없습니다.")

    # ---------- TAB 2: 환경 분석 (4 섹션) ----------
    with tab2:
        e1, e2 = st.columns(2)
        e3, e4 = st.columns(2)

        col_temp, col_humid, col_ec = "평균 온도", "평균 습도", "평균 EC(측정)"
        c_axis = alt.Axis(labelAngle=0, labelOverlap=False, labelPadding=8, labelLimit=200)

        # 섹션 1: 평균 EC(측정)
        df_ec = filtered[["학교", col_ec]].copy()
        chart_ec = alt.Chart(df_ec).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#9ec5fe").encode(
            x=alt.X("학교:N", sort=SCHOOL_KEYS, axis=c_axis, title=None),
            y=alt.Y(f"{col_ec}:Q", title="평균 EC(측정)")
        ).properties(title="평균 EC(측정)", height=330)
        with e1: st.altair_chart(chart_ec, use_container_width=True)

        # 섹션 2: 평균 습도
        df_h = filtered[["학교", col_humid]].copy()
        chart_h = alt.Chart(df_h).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#a7f3d0").encode(
            x=alt.X("학교:N", sort=SCHOOL_KEYS, axis=c_axis, title=None),
            y=alt.Y(f"{col_humid}:Q", title="평균 습도(%)")
        ).properties(title="평균 습도", height=330)
        with e2: st.altair_chart(chart_h, use_container_width=True)

        # 섹션 3: 평균 온도
        df_t = filtered[["학교", col_temp]].copy()
        chart_t = alt.Chart(df_t).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#fde68a").encode(
            x=alt.X("학교:N", sort=SCHOOL_KEYS, axis=c_axis, title=None),
            y=alt.Y(f"{col_temp}:Q", title="평균 온도(°C)")
        ).properties(title="평균 온도", height=330)
        with e3: st.altair_chart(chart_t, use_container_width=True)

        # 섹션 4: RAW · 환경(학교별 10행)
        with e4:
            st.markdown("**RAW 데이터 · 환경(학교별 10행)**")
            tabs = st.tabs(SCHOOL_KEYS)
            for i, sch in enumerate(SCHOOL_KEYS):
                with tabs[i]:
                    if sch in raw_env:
                        st.dataframe(raw_env[sch].head(10), use_container_width=True)
                    else:
                        st.info("해당 학교 환경 데이터가 없습니다.")
