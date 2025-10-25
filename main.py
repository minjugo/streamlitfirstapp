# main.py (UTF-8)
# Streamlit 대시보드 — 4개교 극지식물 EC 실험 (차트 보강판)
import os, glob, io, math
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="극지식물 실험", layout="wide")
st.title("🌱 극지식물 최적 EC 농도 실험 대시보드")
st.subheader("4개교 공동 실험 결과 분석")
alt.data_transformers.disable_max_rows()

SCHOOLS = ["송도고","하늘고","아라고","동산고"]
EC_MAP  = {"송도고":1,"하늘고":2,"아라고":4,"동산고":8}
COLOR   = {"송도고":"#1f77b4","하늘고":"#2ca02c","아라고":"#ffbf00","동산고":"#d62728"}  # HTML과 동일 팔레트

# ---------------- 공통 유틸 ----------------
def guess_school(name:str):
    key = name.lower()
    if "송도" in key: return "송도고"
    if "하늘" in key: return "하늘고"
    if "아라"  in key: return "아라고"
    if "동산" in key: return "동산고"
    return None

def read_csv(bytes_or_path, encs=("utf-8","cp949")):
    for enc in encs:
        try:
            if isinstance(bytes_or_path, (bytes,bytearray)):
                bio = io.BytesIO(bytes_or_path)
                return pd.read_csv(bio, encoding=enc)
            return pd.read_csv(bytes_or_path, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV 인코딩 실패")

def read_env_csv(src, school:str):
    df = read_csv(src)
    cols = {c.lower():c for c in df.columns}
    for k in ["temperature","humid","ec","ph"]:
        if k not in cols: raise ValueError(f"[{school}] 환경 CSV 칼럼 누락: {k}")
    t = pd.to_numeric(df[cols["temperature"]], errors="coerce")
    h = pd.to_numeric(df[cols["humid"]], errors="coerce")
    e = pd.to_numeric(df[cols["ec"]], errors="coerce")
    p = pd.to_numeric(df[cols["ph"]], errors="coerce")
    if p.dropna().mean() > 100: p = p/100.0  # pH 보정
    return {
        "학교": school,
        "평균 온도": t.mean(skipna=True),
        "평균 습도": h.mean(skipna=True),
        "평균 EC(측정)": e.mean(skipna=True),
        "평균 pH": p.mean(skipna=True),
    }, df  # df는 박스플롯용 원자료 반환

def read_growth_excel(bytes_or_path, sheet:str):
    if isinstance(bytes_or_path, (bytes,bytearray)):
        bio = io.BytesIO(bytes_or_path)
        gdf = pd.read_excel(bio, sheet_name=sheet)
    else:
        gdf = pd.read_excel(bytes_or_path, sheet_name=sheet)
    return _summarize_growth_df(gdf, sheet), gdf

def read_growth_csv(src, school:str):
    gdf = read_csv(src)
    return _summarize_growth_df(gdf, school), gdf

def _summarize_growth_df(gdf:pd.DataFrame, school:str):
    if "생중량(g)" in gdf.columns:  # 아라고
        w = pd.to_numeric(gdf["생중량(g)"], errors="coerce").mean()
        L = math.nan; leaf = math.nan
    else:
        for c in ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]:
            if c not in gdf.columns:
                raise ValueError(f"[{school}] 생육 칼럼 누락: {c}")
        w = pd.to_numeric(gdf["지상부 생중량(g)"], errors="coerce").mean()
        L = pd.to_numeric(gdf["지상부 길이(cm)"], errors="coerce").mean()
        leaf = pd.to_numeric(gdf["잎 수(장)"], errors="coerce").mean()
    return {"학교":school,"평균 잎 수":leaf,"평균 길이(cm)":L,"평균 생중량(g)":w}

# ---------------- 업로드 영역 ----------------
with st.sidebar:
    st.header("📁 파일 업로드")
    env_files = st.file_uploader("환경 CSV (여러 개)", type=["csv"], accept_multiple_files=True)
    growth_xlsx = st.file_uploader("생육 엑셀(.xlsx, 선택)", type=["xlsx"])
    growth_csvs = st.file_uploader("또는 생육 CSV(학교별 여러 개)", type=["csv"], accept_multiple_files=True)

# ---------------- 로딩(업로드 우선) ----------------
env_rows, env_raw_by_school = [], {}
if env_files:
    for f in env_files:
        school = guess_school(f.name) or st.selectbox(f"환경 CSV 매핑: {f.name}", [""]+SCHOOLS, key=f"emap_{f.name}")
        if not school: continue
        try:
            row, raw = read_env_csv(f.getvalue(), school)
            env_rows.append(row)
            env_raw_by_school[school] = raw  # 박스플롯용
        except Exception as e:
            st.warning(f"환경 CSV 로드 실패({f.name}): {e}")
env_df = pd.DataFrame(env_rows)

growth_rows, growth_raw_by_school = [], {}
if growth_xlsx is not None:
    for s in SCHOOLS:
        try:
            row, raw = read_growth_excel(growth_xlsx.getvalue(), s)
            growth_rows.append(row)
            growth_raw_by_school[s] = raw
        except Exception:
            pass
if growth_csvs:
    for f in growth_csvs:
        school = guess_school(f.name) or st.selectbox(f"생육 CSV 매핑: {f.name}", [""]+SCHOOLS, key=f"gmap_{f.name}")
        if not school: continue
        try:
            row, raw = read_growth_csv(f.getvalue(), school)
            growth_rows.append(row)
            growth_raw_by_school[school] = raw
        except Exception as e:
            st.warning(f"생육 CSV 로드 실패({f.name}): {e}")
growth_df = pd.DataFrame(growth_rows)

if env_df.empty or growth_df.empty:
    st.info("좌측에서 **환경 CSV(각 학교)** 와 **생육 엑셀 또는 CSV** 를 업로드하면 아래 차트가 나타납니다.")
    st.stop()

# 병합/색상/EC
growth_df["EC(설정)"] = growth_df["학교"].map(EC_MAP)
data = pd.merge(growth_df, env_df, on="학교", how="left")
data["color"] = data["학교"].map(COLOR)

# ---------------- 사이드바 필터 ----------------
with st.sidebar:
    st.header("🔬 데이터 필터")
    school_opts = ["전체"] + [f"{s}(EC{EC_MAP[s]})" for s in SCHOOLS if s in set(data["학교"])]
    sel_sch = st.multiselect("학교 선택(복수)", school_opts, default=["전체"])
    env_opts = ["온도","습도","EC","pH"]
    sel_env = st.multiselect("환경 변수", env_opts, default=["온도","습도","EC"])
    metric_opts = ["지상부 생중량","잎 수","지상부 길이"]
    sel_metric = st.selectbox("생육 지표", metric_opts, index=0)

def norm_sch(selected):
    if ("전체" in selected) or (not selected):
        return list(data["학교"].unique())
    mp = {f"{s}(EC{EC_MAP[s]})":s for s in SCHOOLS}
    return [mp[x] for x in selected if x in mp]

use_schools = norm_sch(sel_sch)
filtered = data[data["학교"].isin(use_schools)].copy()

# ---------------- KPI ----------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("총 학교 수", f"{len(filtered):,}")
avg_w = filtered["평균 생중량(g)"].mean()
c2.metric("평균 생중량", f"{avg_w:.2f} g")
best_row = filtered.loc[filtered["평균 생중량(g)"].idxmax()]
c3.metric("최고 EC 농도 (생중량 기준)", f"EC {int(best_row['EC(설정)'])}")
c4.metric("최고 생중량", f"{best_row['평균 생중량(g)']:.2f} g")

st.markdown("---")

# ---------------- 탭 ----------------
tab1, tab2 = st.tabs(["📊 생육 결과 (HTML 대시보드 구성 반영)", "🌡️ 환경 분석"])

# ========== 탭1: 생육 결과 (3종 그래프) ==========
with tab1:
    metric_map = {"지상부 생중량":"평균 생중량(g)","잎 수":"평균 잎 수","지상부 길이":"평균 길이(cm)"}
    ycol = metric_map[sel_metric]

    # (그래프 A) EC vs 평균 생중량 (선 + 최대값 별)
    line_df = filtered[["학교","EC(설정)","평균 생중량(g)","color"]].dropna(subset=["평균 생중량(g)"]).sort_values("EC(설정)")
    if not line_df.empty:
        max_idx = line_df["평균 생중량(g)"].idxmax()
        line_df["is_max"] = line_df.index==max_idx
        base = alt.Chart(line_df).mark_line(point=True, strokeWidth=2, color="#60a5fa").encode(
            x=alt.X("EC(설정):O", sort=[1,2,4,8], title="EC (설정)"),
            y=alt.Y("평균 생중량(g):Q", title="평균 생중량(g)")
        )
        star = alt.Chart(line_df[line_df["is_max"]]).mark_point(shape="star", size=220, color="#ef4444").encode(
            x="EC(설정):O", y="평균 생중량(g):Q"
        )
        st.caption("**그래프 A. EC vs 평균 생중량 (최고값 ★)**")
        st.altair_chart((base+star).properties(height=340), use_container_width=True)
    else:
        st.info("생중량 평균이 비어 있어 선 그래프를 표시할 수 없습니다.")

    # (그래프 B) 학교별 생중량 비율 (도넛)
    pie_df = filtered[["학교","평균 생중량(g)"]].dropna()
    if not pie_df.empty:
        total = pie_df["평균 생중량(g)"].sum()
        pie_df["비율"] = pie_df["평균 생중량(g)"]/total
        pie_df["color"] = pie_df["학교"].map(COLOR)
        pie = alt.Chart(pie_df).mark_arc(outerRadius=120, innerRadius=70).encode(
            theta=alt.Theta("평균 생중량(g):Q"),
            color=alt.Color("학교:N", scale=alt.Scale(range=[COLOR[s] for s in pie_df["학교"]]), legend=None),
            tooltip=[alt.Tooltip("학교:N"), alt.Tooltip("평균 생중량(g):Q", format=".2f"), alt.Tooltip("비율:Q", format=".1%")]
        )
        st.caption("**그래프 B. 학교별 평균 생중량 비율 (도넛)**")
        st.altair_chart(pie.properties(height=340), use_container_width=True)
    else:
        st.info("도넛 차트를 만들 데이터가 부족합니다.")

    # (그래프 C) 선택 지표 TOP 막대
    bar_df = filtered[["학교","color", ycol]].dropna().sort_values(ycol, ascending=False)
    if not bar_df.empty:
        bar = alt.Chart(bar_df).mark_bar().encode(
            x=alt.X(f"{ycol}:Q", title=sel_metric),
            y=alt.Y("학교:N", sort="-x", title=None),
            color=alt.Color("학교:N", scale=alt.Scale(range=[COLOR[s] for s in bar_df['학교']]), legend=None)
        )
        txt = bar.mark_text(align="left", dx=5, color="#334155").encode(text=alt.Text(f"{ycol}:Q", format=".2f"))
        st.caption("**그래프 C. 학교별 TOP (선택 지표, 값 표시)**")
        st.altair_chart((bar+txt).properties(height=340), use_container_width=True)
    else:
        st.info("선택 지표 데이터가 부족합니다.")

    # (옵션) 박스플롯 — 개체 원자료가 있을 때만
    # CSV/엑셀 원자료에서 ‘지상부 생중량(g)’ 또는 ‘생중량(g)’을 찾아 EC레벨로 박스플롯
    box_rows=[]
    for s, raw in growth_raw_by_school.items():
        if s not in filtered["학교"].values: continue
        if "지상부 생중량(g)" in raw:
            vals = pd.to_numeric(raw["지상부 생중량(g)"], errors="coerce").dropna()
        elif "생중량(g)" in raw:
            vals = pd.to_numeric(raw["생중량(g)"], errors="coerce").dropna()
        else:
            continue
        for v in vals:
            box_rows.append({"EC": EC_MAP[s], "학교": s, "생중량": v, "색": COLOR[s]})
    if box_rows:
        box_df = pd.DataFrame(box_rows)
        box = alt.Chart(box_df).mark_boxplot(size=40).encode(
            x=alt.X("EC:O", sort=[1,2,4,8]),
            y=alt.Y("생중량:Q", title="개체 생중량 (g)"),
            color=alt.Color("학교:N", scale=alt.Scale(range=[COLOR[k] for k in box_df["학교"].unique()]), legend=None)
        )
        st.caption("**그래프 D. EC별 개체 생중량 분포 (박스플롯)**")
        st.altair_chart(box.properties(height=340), use_container_width=True)

# ========== 탭2: 환경 분석 (영향력 순위 + 그룹막대) ==========
with tab2:
    env_map = {"온도":"평균 온도","습도":"평균 습도","EC":"평균 EC(측정)","pH":"평균 pH"}
    use_cols = [env_map[k] for k in sel_env if env_map[k] in filtered.columns]
    if use_cols:
        tidy_env = filtered[["학교"]+use_cols].melt(id_vars=["학교"], var_name="변수", value_name="값")
        chart4 = alt.Chart(tidy_env.dropna()).mark_bar().encode(
            x=alt.X("학교:N", title=None),
            y=alt.Y("값:Q", title="환경 값"),
            color=alt.Color("학교:N", scale=alt.Scale(range=[COLOR[s] for s in tidy_env['학교'].unique()]), legend=None),
            column=alt.Column("변수:N", header=alt.Header(labelOrient="bottom"))
        ).resolve_scale(y='independent')
        st.caption("**그래프 E. 학교별 환경 조건 (선택 변수, 그룹 막대)**")
        st.altair_chart(chart4.properties(height=320), use_container_width=True)

    # 영향력 순위 (스피어만 |r| → 0~100 환산)
    def spearman_abs(x,y):
        return abs(pd.Series(x).rank().corr(pd.Series(y).rank()))
    rank = []
    if len(filtered) >= 2:
        Y = filtered["평균 생중량(g)"]
        for lab, col in env_map.items():
            if col in filtered.columns:
                r = spearman_abs(filtered[col], Y)
                rank.append([lab, r, int(round(r*100))])
    rank_df = pd.DataFrame(rank, columns=["환경 요인","|Spearman r|","영향력 점수(0-100)"]).sort_values("영향력 점수(0-100)", ascending=False)
    if not rank_df.empty:
        base = alt.Chart(rank_df).mark_bar().encode(
            x=alt.X("영향력 점수(0-100):Q", scale=alt.Scale(domain=[0,100])),
            y=alt.Y("환경 요인:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum["영향력 점수(0-100)"]==rank_df["영향력 점수(0-100)"].max(),
                alt.value("#8b5cf6"), alt.value("#475569")
            )
        )
        txt = base.mark_text(align="left", dx=6, color="#cbd5e1").encode(
            text=alt.Text("영향력 점수(0-100):Q", format=".0f")
        )
        st.caption("**그래프 F. 환경 요인 영향력 순위 (n=4, 참고용)**")
        st.altair_chart((base+txt).properties(height=320), use_container_width=True)
    else:
        st.info("영향력 순위를 계산하려면 2개 이상 학교 데이터가 필요합니다.")
