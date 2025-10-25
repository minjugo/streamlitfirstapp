# main.py
# -*- coding: utf-8 -*-
"""
Streamlit 대시보드 - 극지식물 4개교 EC 실험 (업로드 버전)
- 환경 CSV 4개(각 학교별): timestamp, temperature, humid, ec, ph, co2
- 생육 엑셀 1개(4시트: 송도고/하늘고/아라고/동산고)
  * 칼럼: 개체번호, 잎 수(장), 지상부 길이(cm), 지상부 생중량(g), 지하부 생중량(g)
  * 아라고만: 생중량(g) (지상/지하 구분 없음)
- EC 매핑: 송도=1, 하늘=2, 아라=4, 동산=8
"""

import io
import math
import streamlit as st
import pandas as pd
import altair as alt

# --------------------------
# 0) 페이지 설정
# --------------------------
st.set_page_config(page_title="극지식물 실험", layout="wide")
st.title("🌱 극지식물 최적 EC 농도 실험 대시보드")
st.subheader("4개교 공동 실험 결과 분석")

alt.data_transformers.disable_max_rows()  # Altair 행 제한 해제(교육용)

SCHOOL_KEYS = ["송도고", "하늘고", "아라고", "동산고"]
EC_MAP = {"송도고": 1, "하늘고": 2, "아라고": 4, "동산고": 8}
COLOR_MAP = {"송도고":"#8bb8ff", "하늘고":"#88d4a9", "아라고":"#ffd66b", "동산고":"#ff9b9b"}  # 파스텔

# --------------------------
# 1) 업로드 위젯 & 매핑 보조
# --------------------------
with st.sidebar:
    st.header("📁 파일 업로드")
    env_files = st.file_uploader("환경 CSV 4개 업로드", type=["csv"], accept_multiple_files=True, help="각 학교별 1개씩 업로드하세요.")
    growth_file = st.file_uploader("생육 결과 엑셀(.xlsx)", type=["xlsx"], help="시트명: 송도고/하늘고/아라고/동산고")

def infer_school_from_name(name: str) -> str | None:
    """파일명에서 학교명을 추론(간단 키워드 매칭)"""
    key = name.lower()
    if "송도" in key: return "송도고"
    if "하늘" in key: return "하늘고"
    if "아라"  in key: return "아라고"
    if "동산" in key: return "동산고"
    return None

# 업로드된 CSV 파일을 학교에 매핑(자동+수동)
def build_env_mapping(files):
    mapping_rows = []
    used = set()
    for i, f in enumerate(files or []):
        guess = infer_school_from_name(f.name)
        mapping_rows.append({"파일명": f.name, "학교(자동추론)": guess, "선택": guess if guess else ""})
        if guess: used.add(guess)

    # 부족/중복 안내
    return pd.DataFrame(mapping_rows)

env_map_df = build_env_mapping(env_files)

with st.sidebar:
    if env_files:
        st.divider()
        st.caption("🔗 CSV → 학교 매핑 (자동 추론 실패 시 직접 선택)")
        # 각 파일별로 매핑 드롭다운 제공
        if "env_selections" not in st.session_state:
            st.session_state.env_selections = {}

        for i, f in enumerate(env_files):
            default = env_map_df.loc[i, "선택"]
            st.session_state.env_selections[f.name] = st.selectbox(
                f"파일: {f.name}",
                [""] + SCHOOL_KEYS,
                index=([""] + SCHOOL_KEYS).index(default) if default in SCHOOL_KEYS else 0,
                key=f"sel_{f.name}"
            )

# --------------------------
# 2) 데이터 로드(업로드에서 읽기) - 캐시
# --------------------------
@st.cache_data(show_spinner=True)
def load_data_from_uploads(_env_files_meta, _growth_bytes):
    """
    업로드 파일들로부터 환경/생육 평균을 계산해 병합 데이터프레임 반환.
    - _env_files_meta: [(school, bytes), ...] 4건 기대
    - _growth_bytes: 엑셀 파일의 bytes (None 허용)
    """
    # ===== 환경 평균 =====
    env_rows = []
    for school, fbytes in _env_files_meta:
        # CSV를 파일객체로 변환
        bio = io.BytesIO(fbytes)
        try:
            df = pd.read_csv(bio, encoding="utf-8")
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, encoding="cp949")

        # 칼럼 정규화 (소문자 키 → 원래 칼럼명)
        cols = {c.lower(): c for c in df.columns}
        need = ["temperature", "humid", "ec", "ph"]
        for n in need:
            if n not in cols:
                raise ValueError(f"[{school}] 환경 CSV 칼럼 누락: {n}")

        temperature = pd.to_numeric(df[cols["temperature"]], errors="coerce").dropna()
        humid       = pd.to_numeric(df[cols["humid"]], errors="coerce").dropna()
        ec_meas     = pd.to_numeric(df[cols["ec"]], errors="coerce").dropna()
        ph          = pd.to_numeric(df[cols["ph"]], errors="coerce").dropna()

        # pH 스케일 보정: 평균이 100↑면 /100
        if len(ph) and ph.mean() > 100:
            ph = ph / 100.0

        env_rows.append({
            "학교": school,
            "평균 온도": temperature.mean() if len(temperature) else math.nan,
            "평균 습도": humid.mean() if len(humid) else math.nan,
            "평균 EC(측정)": ec_meas.mean() if len(ec_meas) else math.nan,
            "평균 pH": ph.mean() if len(ph) else math.nan,
        })
    env_df = pd.DataFrame(env_rows)

    # ===== 생육 평균 =====
    if _growth_bytes is None:
        raise ValueError("생육 엑셀(.xlsx)이 업로드되지 않았습니다.")
    g_bio = io.BytesIO(_growth_bytes)

    growth_rows = []
    for school in SCHOOL_KEYS:
        gdf = pd.read_excel(g_bio, sheet_name=school)
        g_bio.seek(0)  # 다음 시트 읽기를 위해 포인터 리셋

        if "생중량(g)" in gdf.columns:
            # 아라고 전용(총 생중량)
            avg_weight = pd.to_numeric(gdf["생중량(g)"], errors="coerce").mean()
            avg_leaf   = math.nan
            avg_len    = math.nan
        else:
            # 다른 학교: 지상부 중심 3지표
            need_cols = ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]
            for c in need_cols:
                if c not in gdf.columns:
                    raise ValueError(f"생육 시트[{school}] 칼럼 누락: {c}")
            avg_weight = pd.to_numeric(gdf["지상부 생중량(g)"], errors="coerce").mean()
            avg_len    = pd.to_numeric(gdf["지상부 길이(cm)"], errors="coerce").mean()
            avg_leaf   = pd.to_numeric(gdf["잎 수(장)"], errors="coerce").mean()

        growth_rows.append({
            "학교": school,
            "평균 잎 수": avg_leaf,
            "평균 길이(cm)": avg_len,
            "평균 생중량(g)": avg_weight
        })
    growth_df = pd.DataFrame(growth_rows)
    growth_df["EC(설정)"] = growth_df["학교"].map(EC_MAP)

    # ===== 병합 =====
    combined = pd.merge(growth_df, env_df, on="학교", how="left")
    combined["color"] = combined["학교"].map(COLOR_MAP)
    return combined

# 업로드가 완료되면 바이트와 매핑을 준비
env_meta = []
if env_files:
    # 사용자 선택값에서 학교 매핑
    assigned = set()
    for f in env_files:
        school = st.session_state.env_selections.get(f.name) or infer_school_from_name(f.name)
        if not school:
            continue
        if school in assigned:
            st.warning(f"⚠️ {school}에 중복 매핑된 CSV가 있습니다. 하나만 사용됩니다.")
            continue
        env_meta.append((school, f.getvalue()))
        assigned.add(school)

# 로드 시도
combined_df = None
load_error = None
if env_meta and len(env_meta) >= 2 and growth_file is not None:
    try:
        combined_df = load_data_from_uploads(env_meta, growth_file.getvalue())
    except Exception as e:
        load_error = str(e)

# 상태 안내
if not env_files or growth_file is None:
    st.info("⬅️ 좌측 사이드바에서 **환경 CSV(4개)** 와 **생육 엑셀(.xlsx)** 을 업로드하세요. "
            "파일명이 '송도/하늘/아라/동산'을 포함하면 자동 매핑됩니다.")
elif load_error:
    st.error(f"데이터 로드 중 오류: {load_error}")
elif combined_df is None:
    st.warning("파일은 업로드되었지만 학교 매핑이 충분하지 않습니다. 각 CSV에 대해 학교를 선택하세요.")
else:
    # ====================== 대시보드 본체 (데이터 존재) ======================
    # 3) 사이드바 필터
    with st.sidebar:
        st.divider()
        st.header("🔬 데이터 필터")
        school_options = ["전체","송도고(EC1)","하늘고(EC2)","아라고(EC4)","동산고(EC8)"]
        sel_schools = st.multiselect("학교 선택(복수 선택 가능)", school_options, default=["전체"])

        env_options = ["온도","습도","EC","pH"]
        sel_env = st.multiselect("환경 변수 선택", env_options, default=["온도","습도","EC"])

        metric_options = ["지상부 생중량","잎 수","지상부 길이"]
        sel_metric = st.selectbox("생육 지표", metric_options, index=0)

    # 학교 필터 적용
    def normalize_school_filter(selected):
        if ("전체" in selected) or (not selected):
            return SCHOOL_KEYS
        mp = {"송도고(EC1)":"송도고","하늘고(EC2)":"하늘고","아라고(EC4)":"아라고","동산고(EC8)":"동산고"}
        return [mp[s] for s in selected if s in mp]

    use_schools = normalize_school_filter(sel_schools)
    filtered = combined_df[combined_df["학교"].isin(use_schools)].copy()

    # 4) KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 학교 수", f"{len(filtered):,}")

    overall_avg_w = filtered["평균 생중량(g)"].mean()
    col2.metric("평균 생중량", f"{overall_avg_w:.2f} g")

    best_row = filtered.loc[filtered["평균 생중량(g)"].idxmax()]
    col3.metric("최고 EC 농도 (생중량 기준)", f"EC {int(best_row['EC(설정)'])}")

    col4.metric("최고 생중량", f"{best_row['평균 생중량(g)']:.2f} g")

    st.markdown("---")

    # 5) 탭
    tab1, tab2 = st.tabs(["📊 생육 결과", "🌡️ 환경 분석"])

    # 공통 tidy
    def tidy_growth(df):
        return df[["학교","EC(설정)","평균 생중량(g)","평균 잎 수","평균 길이(cm)","color"]].copy()

    growth_tidy = tidy_growth(filtered)

    # --------------------------
    # 탭1: 생육 결과
    # --------------------------
    with tab1:
        metric_col_map = {
            "지상부 생중량": "평균 생중량(g)",
            "잎 수": "평균 잎 수",
            "지상부 길이": "평균 길이(cm)"
        }
        y_col = metric_col_map[sel_metric]

        ec_order = [1,2,4,8]
        line_df = growth_tidy.sort_values("EC(설정)").dropna(subset=[y_col])

        # 최고점 표시 플래그
        if not line_df.empty:
            max_idx = line_df[y_col].idxmax()
            line_df["is_max"] = line_df.index == max_idx
        else:
            line_df["is_max"] = False

        line = alt.Chart(line_df).mark_line(point=True, strokeWidth=2, color="#6aa9ff").encode(
            x=alt.X("EC(설정):O", sort=ec_order, title="EC (설정)"),
            y=alt.Y(f"{y_col}:Q", title=sel_metric)
        )
        star = alt.Chart(line_df[line_df["is_max"]]).mark_point(shape="star", size=200, color="#ef4444").encode(
            x=alt.X("EC(설정):O", sort=ec_order),
            y=alt.Y(f"{y_col}:Q")
        )

        st.caption("**차트 1. EC vs 선택 지표 (꺾은선, 최고값 ★ 표시)**")
        st.altair_chart((line + star).properties(height=340), use_container_width=True)

        # 학교별 TOP 4 (가로 막대)
        bar_df = growth_tidy[["학교","color",y_col]].dropna().sort_values(y_col, ascending=False)
        bar = alt.Chart(bar_df).mark_bar().encode(
            x=alt.X(f"{y_col}:Q", title=sel_metric),
            y=alt.Y("학교:N", sort="-x", title=None),
            color=alt.Color("학교:N",
                            scale=alt.Scale(range=[combined_df[combined_df['학교']=='송도고']['color'].iloc[0],
                                                   combined_df[combined_df['학교']=='하늘고']['color'].iloc[0],
                                                   combined_df[combined_df['학교']=='아라고']['color'].iloc[0],
                                                   combined_df[combined_df['학교']=='동산고']['color'].iloc[0]]),
                            legend=None)
        )
        text = bar.mark_text(align='left', dx=5, color="#334155").encode(
            text=alt.Text(f"{y_col}:Q", format=".2f")
        )

        st.caption("**차트 2. 학교별 TOP 4 (선택 지표 내림차순, 막대 끝 값 표시)**")
        st.altair_chart((bar + text).properties(height=340), use_container_width=True)

        # 3지표 정규화(0-100)
        norm_cols = ["평균 생중량(g)","평균 잎 수","평균 길이(cm)"]
        norm_df = growth_tidy[["학교","color"] + norm_cols].copy()
        for c in norm_cols:
            cmax = norm_df[c].max(skipna=True)
            norm_df[c+"_점수"] = (norm_df[c] / cmax * 100).where(pd.notna(norm_df[c]), None)

        tidy_norm = norm_df.melt(id_vars=["학교","color"], value_vars=[c+"_점수" for c in norm_cols],
                                 var_name="지표", value_name="점수")
        tidy_norm["지표"] = tidy_norm["지표"].replace({
            "평균 생중량(g)_점수":"생중량 점수",
            "평균 잎 수_점수":"잎 수 점수",
            "평균 길이(cm)_점수":"길이 점수",
        })

        grouped = alt.Chart(tidy_norm.dropna()).mark_bar().encode(
            x=alt.X("학교:N", title=None),
            y=alt.Y("점수:Q", title="정규화 점수(0-100)"),
            color=alt.Color("지표:N", scale=alt.Scale(range=["#9ec5fe","#a7f3d0","#fde68a"])),
            column=alt.Column("지표:N", header=alt.Header(labelOrient="bottom"))
        ).resolve_scale(y='independent')

        st.caption("**차트 3. 3가지 지표 종합 (0-100 정규화, 그룹 막대)**")
        st.altair_chart(grouped.properties(height=320), use_container_width=True)

    # --------------------------
    # 탭2: 환경 분석
    # --------------------------
    with tab2:
        env_map = {"온도":"평균 온도","습도":"평균 습도","EC":"평균 EC(측정)","pH":"평균 pH"}
        env_use_cols = [env_map[e] for e in ["온도","습도","EC","pH"] if e in (sel_env or [])]
        if not env_use_cols:
            st.info("환경 변수를 1개 이상 선택하세요.")
        else:
            env_df = filtered[["학교","color"] + env_use_cols].copy()
            tidy_env = env_df.melt(id_vars=["학교","color"], var_name="변수", value_name="값")

            chart4 = alt.Chart(tidy_env.dropna()).mark_bar().encode(
                x=alt.X("학교:N", title=None),
                y=alt.Y("값:Q", title="환경 값"),
                color=alt.Color("학교:N",
                                scale=alt.Scale(range=[combined_df[combined_df['학교']=='송도고']['color'].iloc[0],
                                                       combined_df[combined_df['학교']=='하늘고']['color'].iloc[0],
                                                       combined_df[combined_df['학교']=='아라고']['color'].iloc[0],
                                                       combined_df[combined_df['학교']=='동산고']['color'].iloc[0]]),
                                legend=None),
                column=alt.Column("변수:N", header=alt.Header(labelOrient="bottom"))
            ).resolve_scale(y='independent')

            st.caption("**차트 4. 학교별 환경 조건 (선택 변수, 그룹 막대)**")
            st.altair_chart(chart4.properties(height=320), use_container_width=True)

        # 스피어만 |r| (학교 수 2개 미만이면 불가)
        def spearman_abs(x, y):
            sx = pd.Series(x).rank()
            sy = pd.Series(y).rank()
            return abs(sx.corr(sy))

        rank_rows = []
        if len(filtered) >= 2:
            y = filtered["평균 생중량(g)"]
            for label, col in env_map.items():
                if col in filtered.columns:
                    r = spearman_abs(filtered[col], y)
                    rank_rows.append([label, r])

        rank_df = pd.DataFrame(rank_rows, columns=["환경 요인","|Spearman r|"]).sort_values("|Spearman r|", ascending=False)
        if rank_df.empty:
            st.info("환경 요인 영향력 순위를 계산하려면 2개 이상의 학교를 선택하세요.")
        else:
            rank_df["영향력 점수(0-100)"] = (rank_df["|Spearman r|"] * 100).round(0).astype(int)
            base = alt.Chart(rank_df).mark_bar().encode(
                x=alt.X("영향력 점수(0-100):Q", title="영향력 점수(0-100)", scale=alt.Scale(domain=[0,100])),
                y=alt.Y("환경 요인:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum["영향력 점수(0-100)"] == rank_df["영향력 점수(0-100)"].max(),
                    alt.value("#a78bfa"),  # 1위 강조
                    alt.value("#475569")
                )
            )
            txt = base.mark_text(align="left", dx=6, color="#cbd5e1").encode(
                text=alt.Text("영향력 점수(0-100):Q", format=".0f")
            )
            st.caption("**차트 5. 환경 요인 영향력 순위 (스피어만 상관 절댓값, n=4 참고용)**")
            st.altair_chart((base + txt).properties(height=320), use_container_width=True)

    # 사용 안내
    with st.expander("ℹ️ 사용법 / 업로드 팁"):
        st.markdown(
            """
            **업로드 팁**
            - 환경 CSV는 파일명에 `송도/하늘/아라/동산`이 들어가면 자동 매핑됩니다. 아니라면 사이드바에서 직접 매핑하세요.
            - pH가 500대처럼 비정상 스케일이면 자동으로 `/100` 보정됩니다.

            **주의**
            - 환경 영향력 순위는 표본(n=4)이 작아 **참고용**입니다.
            - 아라고는 ‘생중량(g)’ 단일 칼럼이므로 잎 수/길이 평균은 표시되지 않습니다.
            """
        )
