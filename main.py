# main.py
# -*- coding: utf-8 -*-
"""
Streamlit 대시보드 - 극지식물 4개교 EC 실험
요구사항:
- 라이브러리: streamlit, pandas, altair 만 사용
- UTF-8, 파일 경로는 실행 폴더 기준 (상대경로)
- @st.cache_data 로 캐시
- 오류 처리: 파일 없으면 st.error()
- 숫자 형식: 생중량 .2f, 상관계수 .3f, 천단위 쉼표
- 반응형: use_container_width=True
"""

import os
import glob
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

alt.data_transformers.disable_max_rows()

# --------------------------
# 1) 유틸: 안전한 파일 찾기
# --------------------------
def find_first(patterns):
    """여러 패턴 후보 중 먼저 존재하는 경로를 반환"""
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

# --------------------------
# 2) 데이터 로더 (캐시)
# --------------------------
@st.cache_data
def load_data():
    """
    - 4개 환경 CSV 로드 -> 학교별 평균(temperature, humid, ec, ph)
    - 1개 생육 엑셀(4시트) 로드 -> 학교별 평균(잎 수, 지상부 길이, 지상부 생중량)
    - EC 매핑(송도=1, 하늘=2, 아라=4, 동산=8) 부여
    - 병합(combined_df) 반환: 한 행 = 한 학교
    """
    # 파일 탐색(한글 파일명/스페이스/정규화 차이를 감안해 와일드카드 사용)
    env_paths = {
        "송도고": find_first(["*송도*환경*데이터*.csv","*송도*환경*데이터.csv","*송도고*환경*.csv"]),
        "하늘고": find_first(["*하늘*환경*데이터*.csv","*하늘*환경*데이터.csv","*하늘고*환경*.csv"]),
        "아라고": find_first(["*아라*환경*데이터*.csv","*아라*환경*데이터.csv","*아라고*환경*.csv"]),
        "동산고": find_first(["*동산*환경*데이터*.csv","*동산*환경*데이터.csv","*동산고*환경*.csv"]),
    }
    growth_xlsx = find_first(["*4개교*생육*데이터*.xlsx","*생육*결과*데이터*.xlsx","*4개교*생육*.xlsx"])

    # 파일 존재 검사
    missing = [s for s,p in env_paths.items() if not p] + (["생육엑셀"] if not growth_xlsx else [])
    if missing:
        st.error(f"❌ 데이터 파일을 찾지 못했습니다: {', '.join(missing)}\n\n"
                 f"실행 폴더에 CSV 4개 + 엑셀 1개를 넣어주세요.")
        return None

    # 환경 평균 계산
    env_rows = []
    for school, path in env_paths.items():
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            # 인코딩 예외 처리
            df = pd.read_csv(path, encoding="cp949")
        # 기대 칼럼: timestamp, temperature, humid, ec, ph, co2
        cols_needed = {"temperature","humid","ec","ph"}
        if not cols_needed.issubset(set(map(str.lower, df.columns))):
            st.error(f"❌ 환경 CSV 칼럼 확인 필요: {os.path.basename(path)}")
            return None

        # 대소문자/한영 혼합 대비 - 소문자 접근
        _lc = {c.lower(): c for c in df.columns}
        temperature = df[_lc["temperature"]].astype("float").dropna()
        humid       = df[_lc["humid"]].astype("float").dropna()
        ec_meas     = df[_lc["ec"]].astype("float").dropna()
        ph          = df[_lc["ph"]].astype("float").dropna()

        # pH 센서 스케일 이슈 보정 (평균이 100 초과면 /100)
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

    # 생육 평균 계산 (엑셀 4시트)
    growth_rows = []
    sheet_map = {"송도고":"송도고","하늘고":"하늘고","아라고":"아라고","동산고":"동산고"}
    for school, sheet in sheet_map.items():
        try:
            gdf = pd.read_excel(growth_xlsx, sheet_name=sheet)
        except Exception:
            st.error(f"❌ 생육 엑셀 시트 로드 실패: {sheet}")
            return None

        # 아라고만 '생중량(g)' 단일 칼럼
        if "생중량(g)" in gdf.columns:
            avg_weight = gdf["생중량(g)"].astype("float").mean()
            avg_leaf   = math.nan
            avg_len    = math.nan
        else:
            # 칼럼명 존재 확인
            needed = ["지상부 생중량(g)","지상부 길이(cm)","잎 수(장)"]
            for c in needed:
                if c not in gdf.columns:
                    st.error(f"❌ 생육 엑셀 칼럼 확인 필요: 시트[{sheet}] - 누락: {c}")
                    return None
            avg_weight = gdf["지상부 생중량(g)"].astype("float").mean()
            avg_len    = gdf["지상부 길이(cm)"].astype("float").mean()
            avg_leaf   = gdf["잎 수(장)"].astype("float").mean()

        growth_rows.append({
            "학교": school,
            "평균 잎 수": avg_leaf,
            "평균 길이(cm)": avg_len,
            "평균 생중량(g)": avg_weight
        })
    growth_df = pd.DataFrame(growth_rows)

    # EC 매핑
    ec_map = {"송도고":1, "하늘고":2, "아라고":4, "동산고":8}
    growth_df["EC(설정)"] = growth_df["학교"].map(ec_map)

    # 병합
    combined = pd.merge(growth_df, env_df, on="학교", how="left")

    # 색상(파스텔)
    color_map = {"송도고":"#8bb8ff", "하늘고":"#88d4a9", "아라고":"#ffd66b", "동산고":"#ff9b9b"}
    combined["color"] = combined["학교"].map(color_map)

    return combined


data = load_data()
if data is None:
    st.stop()

# --------------------------
# 3) 사이드바 필터
# --------------------------
with st.sidebar:
    st.header("🔬 데이터 필터")
    school_options = ["전체","송도고(EC1)","하늘고(EC2)","아라고(EC4)","동산고(EC8)"]
    sel_schools = st.multiselect("학교 선택(복수 선택 가능)", school_options, default=["전체"])

    env_options = ["온도","습도","EC","pH"]
    sel_env = st.multiselect("환경 변수 선택", env_options, default=["온도","습도","EC"])

    metric_options = ["지상부 생중량","잎 수","지상부 길이"]
    sel_metric = st.selectbox("생육 지표", metric_options, index=0)

# 학교 필터 해석
def normalize_school_filter(selected):
    # "전체" 포함 시 4개교 모두
    if ("전체" in selected) or (not selected):
        return ["송도고","하늘고","아라고","동산고"]
    mapping = {
        "송도고(EC1)":"송도고", "하늘고(EC2)":"하늘고",
        "아라고(EC4)":"아라고", "동산고(EC8)":"동산고"
    }
    return [mapping[s] for s in selected if s in mapping]

use_schools = normalize_school_filter(sel_schools)

# --------------------------
# 4) 필터 적용 데이터
# --------------------------
filtered = data[data["학교"].isin(use_schools)].copy()

# --------------------------
# 5) KPI 메트릭 4개
# --------------------------
col1, col2, col3, col4 = st.columns(4)

# 총 학교 수
col1.metric("총 학교 수", f"{len(filtered):,}")

# 평균 생중량
overall_avg_w = filtered["평균 생중량(g)"].mean()
col2.metric("평균 생중량", f"{overall_avg_w:.2f} g")

# 최고 EC (생중량 기준)
best_row = filtered.loc[filtered["평균 생중량(g)"].idxmax()]
col3.metric("최고 EC 농도 (생중량 기준)", f"EC {int(best_row['EC(설정)'])}")

# 최고 생중량
col4.metric("최고 생중량", f"{best_row['평균 생중량(g)']:.2f} g")

st.markdown("---")

# --------------------------
# 6) 탭 구성
# --------------------------
tab1, tab2 = st.tabs(["📊 생육 결과", "🌡️ 환경 분석"])

# 공통 축 데이터
# (Altair용 tidy 데이터 생성)
def tidy_growth(df):
    out = df[["학교","EC(설정)","평균 생중량(g)","평균 잎 수","평균 길이(cm)","color"]].copy()
    return out

growth_tidy = tidy_growth(filtered)

# --------------------------
# 탭1: 생육 결과
# --------------------------
with tab1:
    # (차트 1) EC vs 선택 지표 (꺾은선)
    metric_col_map = {
        "지상부 생중량": "평균 생중량(g)",
        "잎 수": "평균 잎 수",
        "지상부 길이": "평균 길이(cm)"
    }
    y_col = metric_col_map[sel_metric]

    # EC 순서대로
    ec_order = [1,2,4,8]
    line_df = growth_tidy.sort_values("EC(설정)").dropna(subset=[y_col])

    # 최고값 포인트 표시 플래그
    if not line_df.empty:
        max_idx = line_df[y_col].idxmax()
        line_df["is_max"] = line_df.index == max_idx
    else:
        line_df["is_max"] = False

    line = alt.Chart(line_df).mark_line(point=True, strokeWidth=2, color="#6aa9ff").encode(
        x=alt.X("EC(설정):O", sort=ec_order, title="EC (설정)"),
        y=alt.Y(f"{y_col}:Q", title=sel_metric)
    )

    # 빨간 별(★) 마커 - 최고값
    star = alt.Chart(line_df[line_df["is_max"]]).mark_point(shape="star", size=200, color="#ef4444").encode(
        x=alt.X("EC(설정):O", sort=ec_order),
        y=alt.Y(f"{y_col}:Q")
    )

    st.caption("**차트 1. EC vs 선택 지표 (꺾은선, 최고값 ★ 표시)**")
    st.altair_chart((line + star).properties(height=340), use_container_width=True)

    # (차트 2) 학교별 TOP 4 (가로 막대)
    bar_df = growth_tidy[["학교","color",y_col]].dropna().sort_values(y_col, ascending=False)
    bar = alt.Chart(bar_df).mark_bar().encode(
        x=alt.X(f"{y_col}:Q", title=sel_metric),
        y=alt.Y("학교:N", sort="-x", title=None),
        color=alt.Color("학교:N", scale=alt.Scale(range=[data[data['학교']=='송도고']['color'].iloc[0],
                                                        data[data['학교']=='하늘고']['color'].iloc[0],
                                                        data[data['학교']=='아라고']['color'].iloc[0],
                                                        data[data['학교']=='동산고']['color'].iloc[0]]),
                       legend=None)
    )
    text = bar.mark_text(align='left', dx=5, color="#334155").encode(
        text=alt.Text(f"{y_col}:Q", format=".2f")
    )

    st.caption("**차트 2. 학교별 TOP 4 (선택 지표 내림차순, 막대 끝 값 표시)**")
    st.altair_chart((bar + text).properties(height=340), use_container_width=True)

    # (차트 3) 3가지 지표 정규화(0-100) 그룹 막대
    norm_cols = ["평균 생중량(g)","평균 잎 수","평균 길이(cm)"]
    norm_df = growth_tidy[["학교","color"] + norm_cols].copy()

    for c in norm_cols:
        cmax = norm_df[c].max(skipna=True)
        norm_df[c+"_점수"] = (norm_df[c] / cmax * 100).where(pd.notna(norm_df[c]), None)

    tidy_norm = norm_df.melt(id_vars=["학교","color"], value_vars=[c+"_점수" for c in norm_cols],
                             var_name="지표", value_name="점수")
    # 라벨 정리
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
    # 환경 변수 선택 -> tidy 변환
    env_map = {"온도":"평균 온도","습도":"평균 습도","EC":"평균 EC(측정)","pH":"평균 pH"}
    env_use_cols = [env_map[e] for e in sel_env if e in env_map]
    env_df = filtered[["학교","color"] + env_use_cols].copy()
    tidy_env = env_df.melt(id_vars=["학교","color"], var_name="변수", value_name="값")

    # (차트 4) 학교별 환경 조건 (그룹 막대)
    chart4 = alt.Chart(tidy_env.dropna()).mark_bar().encode(
        x=alt.X("학교:N", title=None),
        y=alt.Y("값:Q", title="환경 값"),
        color=alt.Color("학교:N", scale=alt.Scale(range=[data[data['학교']=='송도고']['color'].iloc[0],
                                                        data[data['학교']=='하늘고']['color'].iloc[0],
                                                        data[data['학교']=='아라고']['color'].iloc[0],
                                                        data[data['학교']=='동산고']['color'].iloc[0]]),
                       legend=None),
        column=alt.Column("변수:N", header=alt.Header(labelOrient="bottom"))
    ).resolve_scale(y='independent')

    st.caption("**차트 4. 학교별 환경 조건 (선택 변수, 그룹 막대)**")
    st.altair_chart(chart4.properties(height=320), use_container_width=True)

    # (차트 5) 환경 요인 영향력 순위 (스피어만 |r|)
    # 표본이 4개 → 학교 수 2개 미만이면 계산 불가
    def spearman_abs(x, y):
        """순위 기반 상관(절댓값)"""
        s = pd.Series(x).rank().values
        t = pd.Series(y).rank().values
        sx, sy = pd.Series(s), pd.Series(t)
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

# --------------------------
# 사용법(하단 안내)
# --------------------------
with st.expander("ℹ️ 사용법 / 데이터"):
    st.markdown(
        """
        **사용법**
        1) 좌측 사이드바에서 학교·환경·지표를 선택하면 모든 차트가 갱신됩니다.  
        2) KPI는 현재 필터가 적용된 데이터로 계산됩니다.  
        3) 환경 영향력 순위는 선택한 학교 집합에서 *평균 생중량*과의 스피어만 상관을 0–100으로 환산한 값입니다(표본 n=4 참고용).

        **파일 배치**
        - `main.py`와 **같은 폴더**에 다음 파일들을 넣으세요.  
          · 환경 CSV 4개(송도/하늘/아라/동산, 파일명은 자유, 자동 탐색됨)  
          · 생육 엑셀 1개(4시트: 송도고/하늘고/아라고/동산고)  
        - pH가 500대처럼 비정상 스케일이면 자동으로 `/100` 보정합니다.

        **컬러**
        - 송도=파랑(파스텔), 하늘=초록, 아라=노랑, 동산=빨강 (막대/포인트 일관 적용)
        """
    )
