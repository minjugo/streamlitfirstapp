# main.py (Design Upgraded)
# -*- coding: utf-8 -*-
import io, math
import streamlit as st
import pandas as pd
import altair as alt

# --------------------------
# Page Config & Global Style
# --------------------------
st.set_page_config(page_title="극지식물 실험", layout="wide")

# Altair light theme (inspired by Streamlit Gallery aesthetic)
def soft_theme():
    return {
        "config": {
            "view": {"continuousWidth": 500, "continuousHeight": 320},
            "axis": {
                "labelFont": "Inter, Pretendard, Apple SD Gothic Neo, Segoe UI, Roboto, Arial",
                "titleFont": "Inter, Pretendard, Apple SD Gothic Neo, Segoe UI, Roboto, Arial",
                "gridColor": "#e9edf5",
                "tickColor": "#cfd3dc",
                "labelColor": "#44506b",
                "titleColor": "#2b344a"
            },
            "legend": {
                "labelFont": "Inter, Pretendard, Apple SD Gothic Neo, Segoe UI, Roboto, Arial",
                "titleFont": "Inter, Pretendard, Apple SD Gothic Neo, Segoe UI, Roboto, Arial",
                "labelColor": "#3a4762",
                "titleColor": "#2b344a",
                "symbolStrokeColor": "#8ea5ff"
            },
            "title": {"font": "Inter, Pretendard, Apple SD Gothic Neo, Segoe UI, Roboto, Arial", "color": "#1f2a44"},
            "range": {
                "category": ["#8bb8ff","#88d4a9","#ffd66b","#ff9b9b","#a78bfa","#60a5fa"]
            }
        }
    }

alt.themes.register("soft_theme", soft_theme)
alt.themes.enable("soft_theme")
alt.data_transformers.disable_max_rows()

# Global CSS (glass cards, spacing)
st.markdown("""
<style>
:root{
  --bg:#f7f9fd; --panel:#ffffffee; --border:#e9eef7; --text:#1f2a44; --muted:#6b7a99;
}
body { background: var(--bg); }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.hero {
  background: radial-gradient(1200px 500px at 20% -10%, #e9f2ff88, transparent 60%),
              radial-gradient(900px 500px at 120% 0%, #ffe7b988, transparent 50%);
  border: 1px solid var(--border); border-radius: 16px;
  padding: 24px 28px; box-shadow: 0 12px 40px rgba(31,59,140,0.06);
}
.hero h1 { margin: 0 0 6px 0; color: var(--text); font-weight: 800; letter-spacing:.2px }
.hero p { color: var(--muted); margin: 0; }
.card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 16px; padding: 16px 18px; box-shadow: 0 8px 28px rgba(13,40,70,.06);
}
.kpi { display:flex; flex-direction:column; gap:6px }
.kpi .label { font-size:12px; color:var(--muted) }
.kpi .value { font-size:28px; font-weight:800; color:var(--text) }
.kpi .hint { font-size:12px; color:#8a97b6 }
.section-title { margin: 6px 0 10px 2px; color:#2b344a; font-weight:700 }
hr { border: none; height:1px; background: var(--border); margin: 10px 0 18px 0;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Constants
# --------------------------
SCHOOL_KEYS = ["송도고", "하늘고", "아라고", "동산고"]
EC_MAP = {"송도고": 1, "하늘고": 2, "아라고": 4, "동산고": 8}
COLOR_MAP = {"송도고":"#8bb8ff", "하늘고":"#88d4a9", "아라고":"#ffd66b", "동산고":"#ff9b9b"}

# --------------------------
# Hero
# --------------------------
st.markdown(
    '<div class="hero"><h1>🌱 극지식물 최적 EC 농도 실험 대시보드</h1>'
    '<p>4개교 공동 실험 결과를 업로드하여 환경·생육 지표를 비교 분석하고, 깔끔한 카드/차트로 요약합니다.</p></div>',
    unsafe_allow_html=True
)
st.write("")

# --------------------------
# Sidebar: Upload & mapping
# --------------------------
with st.sidebar:
    st.header("📁 파일 업로드")
    env_files = st.file_uploader("환경 CSV 4개", type=["csv"], accept_multiple_files=True,
                                 help="각 학교별 CSV 1개 (timestamp, temperature, humid, ec, ph, co2)")
    growth_file = st.file_uploader("생육 결과 엑셀(.xlsx)", type=["xlsx"],
                                   help="시트명: 송도고/하늘고/아라고/동산고")

def infer_school(name: str):
    key = name.lower()
    if "송도" in key: return "송도고"
    if "하늘" in key: return "하늘고"
    if "아라"  in key: return "아라고"
    if "동산" in key: return "동산고"
    return None

if env_files:
    st.sidebar.divider()
    st.sidebar.caption("🔗 CSV ↔ 학교 매핑 (자동 실패 시 직접 지정)")
    if "env_sel" not in st.session_state: st.session_state.env_sel = {}
    for f in env_files:
        guess = infer_school(f.name) or ""
        st.session_state.env_sel[f.name] = st.sidebar.selectbox(
            f"파일: {f.name}",
            [""] + SCHOOL_KEYS,
            index=([""]+SCHOOL_KEYS).index(guess) if guess in SCHOOL_KEYS else 0,
            key=f"sel_{f.name}"
        )

# --------------------------
# Data loader (from uploads)
# --------------------------
@st.cache_data(show_spinner=True)
def load_from_uploads(env_meta, xlsx_bytes):
    # env_meta: [(school, file_bytes), ...]
    # ----- ENV AVG -----
    env_rows = []
    for school, fb in env_meta:
        bio = io.BytesIO(fb)
        try: df = pd.read_csv(bio, encoding="utf-8")
        except Exception:
            bio.seek(0); df = pd.read_csv(bio, encoding="cp949")
        cols = {c.lower(): c for c in df.columns}
        need = ["temperature","humid","ec","ph"]
        for n in need:
            if n not in cols: raise ValueError(f"[{school}] 환경 CSV 칼럼 누락: {n}")
        t = pd.to_numeric(df[cols["temperature"]], errors="coerce").dropna()
        h = pd.to_numeric(df[cols["humid"]], errors="coerce").dropna()
        e = pd.to_numeric(df[cols["ec"]], errors="coerce").dropna()
        p = pd.to_numeric(df[cols["ph"]], errors="coerce").dropna()
        if len(p) and p.mean() > 100: p = p/100.0
        env_rows.append({
            "학교": school, "평균 온도": t.mean() if len(t) else math.nan,
            "평균 습도": h.mean() if len(h) else math.nan,
            "평균 EC(측정)": e.mean() if len(e) else math.nan,
            "평균 pH": p.mean() if len(p) else math.nan
        })
    env_df = pd.DataFrame(env_rows)

    # ----- GROWTH AVG -----
    if xlsx_bytes is None: raise ValueError("생육 엑셀(.xlsx) 업로드 필요")
    bio = io.BytesIO(xlsx_bytes)
    g_rows = []
    for s in SCHOOL_KEYS:
        gdf = pd.read_excel(bio, sheet_name=s); bio.seek(0)
        if "생중량(g)" in gdf.columns:
            w = pd.to_numeric(gdf["생중량(g)"], errors="coerce").mean()
            l = math.nan; leaf = math.nan
        else:
            for c in ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]:
                if c not in gdf.columns: raise ValueError(f"[{s}] 생육 칼럼 누락: {c}")
            w = pd.to_numeric(gdf["지상부 생중량(g)"], errors="coerce").mean()
            l = pd.to_numeric(gdf["지상부 길이(cm)"], errors="coerce").mean()
            leaf = pd.to_numeric(gdf["잎 수(장)"], errors="coerce").mean()
        g_rows.append({"학교": s, "평균 생중량(g)": w, "평균 길이(cm)": l, "평균 잎 수": leaf})
    g = pd.DataFrame(g_rows); g["EC(설정)"] = g["학교"].map(EC_MAP)

    combined = pd.merge(g, env_df, on="학교", how="left")
    combined["color"] = combined["학교"].map(COLOR_MAP)
    return combined

# Build env_meta from mapping
env_meta = []
if env_files:
    used = set()
    for f in env_files:
        sch = st.session_state.env_sel.get(f.name) or infer_school(f.name)
        if sch and sch not in used:
            env_meta.append((sch, f.getvalue())); used.add(sch)

data, load_err = None, None
if env_meta and growth_file is not None:
    try:
        data = load_from_uploads(env_meta, growth_file.getvalue())
    except Exception as e:
        load_err = str(e)

if not env_files or growth_file is None:
    st.info("좌측 사이드바에서 **환경 CSV(최대 4개)** 와 **생육 엑셀(.xlsx)** 을 업로드하세요. 파일명에 ‘송도/하늘/아라/동산’이 있으면 자동 매핑합니다.")
elif load_err:
    st.error(f"데이터 로드 오류: {load_err}")
elif data is None:
    st.warning("CSV ↔ 학교 매핑을 완료해 주세요.")
else:
    # --------------------------
    # Sidebar filters (after data)
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
        if ("전체" in selected) or (not selected): return SCHOOL_KEYS
        mp = {"송도고(EC1)":"송도고","하늘고(EC2)":"하늘고","아라고(EC4)":"아라고","동산고(EC8)":"동산고"}
        return [mp[s] for s in selected if s in mp]

    use_schools = norm_school_filter(sel_schools)
    filtered = data[data["학교"].isin(use_schools)].copy()

    # --------------------------
    # KPI Cards (styled)
    # --------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="card kpi"><div class="label">총 학교 수</div><div class="value">{len(filtered):,}</div><div class="hint">선택된 범위</div></div>', unsafe_allow_html=True)
    avg_w = filtered["평균 생중량(g)"].mean()
    with c2: st.markdown(f'<div class="card kpi"><div class="label">평균 생중량</div><div class="value">{avg_w:.2f} g</div><div class="hint">소수점 2자리</div></div>', unsafe_allow_html=True)
    best = filtered.loc[filtered["평균 생중량(g)"].idxmax()]
    with c3: st.markdown(f'<div class="card kpi"><div class="label">최고 EC 농도 (생중량 기준)</div><div class="value">EC {int(best["EC(설정)"])}</div><div class="hint">{best["학교"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="card kpi"><div class="label">최고 생중량</div><div class="value">{best["평균 생중량(g)"]:.2f} g</div><div class="hint">{best["학교"]}</div></div>', unsafe_allow_html=True)

    st.write("")  # spacing

    # --------------------------
    # Tabs
    # --------------------------
    tab1, tab2 = st.tabs(["📊 생육 결과", "🌡️ 환경 분석"])

    # Common tidy
    tidy = filtered[["학교","EC(설정)","평균 생중량(g)","평균 잎 수","평균 길이(cm)","color"]].copy()

    # ===== TAB 1 =====
    with tab1:
        # Chart 1: EC vs 선택 지표 (line + star)
        metric_map = {"지상부 생중량":"평균 생중량(g)","잎 수":"평균 잎 수","지상부 길이":"평균 길이(cm)"}
        ycol = metric_map[sel_metric]
        ln_df = tidy.sort_values("EC(설정)").dropna(subset=[ycol])

        if not ln_df.empty:
            max_idx = ln_df[ycol].idxmax()
            ln_df["is_max"] = ln_df.index == max_idx
        else:
            ln_df["is_max"] = False

        base = alt.Chart(ln_df).mark_line(point=True, strokeWidth=2, color="#5b8def").encode(
            x=alt.X("EC(설정):O", sort=[1,2,4,8], title="EC (설정)"),
            y=alt.Y(f"{ycol}:Q", title=sel_metric),
            tooltip=[alt.Tooltip("학교:N"), alt.Tooltip("EC(설정):O"), alt.Tooltip(f"{ycol}:Q", format=".2f")]
        )
        star = alt.Chart(ln_df[ln_df["is_max"]]).mark_point(shape="star", size=220, color="#ef4444").encode(
            x="EC(설정):O", y=f"{ycol}:Q"
        )
        st.markdown('<div class="section-title">차트 1 · EC vs 선택 지표</div>', unsafe_allow_html=True)
        st.altair_chart((base + star).properties(height=340), use_container_width=True)

        # Chart 2: TOP 4 horizontal bars
        bar_df = tidy[["학교","color",ycol]].dropna().sort_values(ycol, ascending=False)
        bar = alt.Chart(bar_df).mark_bar(cornerRadius=6).encode(
            x=alt.X(f"{ycol}:Q", title=sel_metric),
            y=alt.Y("학교:N", sort="-x", title=None),
            color=alt.Color("학교:N", scale=alt.Scale(range=[data[data['학교']=='송도고']['color'].iloc[0],
                                                             data[data['학교']=='하늘고']['color'].iloc[0],
                                                             data[data['학교']=='아라고']['color'].iloc[0],
                                                             data[data['학교']=='동산고']['color'].iloc[0]]),
                           legend=None),
            tooltip=[alt.Tooltip("학교:N"), alt.Tooltip(f"{ycol}:Q", format=".2f")]
        )
        text = bar.mark_text(align="left", dx=6, color="#3a4762").encode(text=alt.Text(f"{ycol}:Q", format=".2f"))
        st.markdown('<div class="section-title">차트 2 · 학교별 TOP 4</div>', unsafe_allow_html=True)
        st.altair_chart((bar + text).properties(height=340), use_container_width=True)

        # Chart 3: 3지표 정규화 (0-100)
        norm_cols = ["평균 생중량(g)","평균 잎 수","평균 길이(cm)"]
        ndf = tidy[["학교","color"] + norm_cols].copy()
        for c in norm_cols:
            cmax = ndf[c].max(skipna=True)
            ndf[c+"_점수"] = (ndf[c] / cmax * 100).where(pd.notna(ndf[c]), None)
        tnorm = ndf.melt(id_vars=["학교","color"], value_vars=[c+"_점수" for c in norm_cols],
                         var_name="지표", value_name="점수")
        tnorm["지표"] = tnorm["지표"].replace({
            "평균 생중량(g)_점수":"생중량 점수","평균 잎 수_점수":"잎 수 점수","평균 길이(cm)_점수":"길이 점수"
        })
        grouped = alt.Chart(tnorm.dropna()).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("학교:N", title=None),
            y=alt.Y("점수:Q", title="정규화 점수(0-100)"),
            color=alt.Color("지표:N", scale=alt.Scale(range=["#9ec5fe","#a7f3d0","#fde68a"])),
            column=alt.Column("지표:N", header=alt.Header(labelOrient="bottom"))
        ).resolve_scale(y='independent')
        st.markdown('<div class="section-title">차트 3 · 3가지 지표 종합</div>', unsafe_allow_html=True)
        st.altair_chart(grouped.properties(height=320), use_container_width=True)

    # ===== TAB 2 =====
    with tab2:
        env_map = {"온도":"평균 온도","습도":"평균 습도","EC":"평균 EC(측정)","pH":"평균 pH"}
        env_cols = [env_map[e] for e in sel_env] if sel_env else []
        if not env_cols:
            st.info("환경 변수를 1개 이상 선택하세요.")
        else:
            e_df = filtered[["학교","color"] + env_cols].copy()
            tidy_env = e_df.melt(id_vars=["학교","color"], var_name="변수", value_name="값")
            chart4 = alt.Chart(tidy_env.dropna()).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("학교:N", title=None),
                y=alt.Y("값:Q", title="환경 값"),
                color=alt.Color("학교:N",
                    scale=alt.Scale(range=[data[data['학교']=='송도고']['color'].iloc[0],
                                           data[data['학교']=='하늘고']['color'].iloc[0],
                                           data[data['학교']=='아라고']['color'].iloc[0],
                                           data[data['학교']=='동산고']['color'].iloc[0]]),
                    legend=None),
                column=alt.Column("변수:N", header=alt.Header(labelOrient="bottom"))
            ).resolve_scale(y='independent')
            st.markdown('<div class="section-title">차트 4 · 학교별 환경 조건</div>', unsafe_allow_html=True)
            st.altair_chart(chart4.properties(height=320), use_container_width=True)

        # Spearman |r| (참고용)
        def spearman_abs(x, y):
            sx, sy = pd.Series(x).rank(), pd.Series(y).rank()
            return abs(sx.corr(sy))

        rows = []
        if len(filtered) >= 2:
            y = filtered["평균 생중량(g)"]
            for lab, col in env_map.items():
                if col in filtered.columns:
                    r = spearman_abs(filtered[col], y)
                    rows.append([lab, r])
        r_df = pd.DataFrame(rows, columns=["환경 요인","|Spearman r|"]).sort_values("|Spearman r|", ascending=False)
        if r_df.empty:
            st.info("환경 영향력 순위를 계산하려면 2개 이상의 학교를 선택하세요.")
        else:
            r_df["영향력 점수(0-100)"] = (r_df["|Spearman r|"] * 100).round(0).astype(int)
            base = alt.Chart(r_df).mark_bar(cornerRadius=6).encode(
                x=alt.X("영향력 점수(0-100):Q", title="영향력 점수(0-100)", scale=alt.Scale(domain=[0,100])),
                y=alt.Y("환경 요인:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum["영향력 점수(0-100)"] == r_df["영향력 점수(0-100)"].max(),
                    alt.value("#a78bfa"), alt.value("#9fb3c8")
                ),
                tooltip=[alt.Tooltip("환경 요인:N"), alt.Tooltip("|Spearman r|:Q", format=".3f"),
                         alt.Tooltip("영향력 점수(0-100):Q", format=".0f")]
            )
            text = base.mark_text(align="left", dx=6, color="#334155").encode(
                text=alt.Text("영향력 점수(0-100):Q", format=".0f")
            )
            st.markdown('<div class="section-title">차트 5 · 환경 요인 영향력 순위 (n=4 참고용)</div>', unsafe_allow_html=True)
            st.altair_chart((base + text).properties(height=320), use_container_width=True)

# Footer spacing
st.write("")
