# -*- coding: utf-8 -*-
import io
import sys
from datetime import datetime

import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------
# 기본 설정
# ---------------------------------------------
st.set_page_config(
    page_title="송도고 EcoSmartFarm 대시보드",
    page_icon="🌿",
    layout="wide",
)
st.title("🌿 송도고 EcoSmartFarm 환경·생육 데이터 대시보드")

st.caption(
    "※ 본 앱은 업로드된 원시 파일(co2.xlsx, ec-ph-wt.xlsx, humid.xlsx, tmep.xlsx, 송도고_스마트팜_생육결과.xlsx)을 "
    "하나의 데이터 프레임으로 통합하여 분석·시각화합니다."
)

# ---------------------------------------------
# 유틸 함수들
# ---------------------------------------------
def safe_read_table(path: str):
    """
    파일을 우선 Excel로 시도하고, 실패하면 CSV(cp949)로 다시 시도.
    """
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        try:
            return pd.read_csv(path, encoding="cp949")
        except Exception as e:
            raise RuntimeError(f"파일을 열 수 없습니다: {path} / {e}")

def find_and_build_datetime(df: pd.DataFrame):
    """
    다양한 이름의 날짜/시간 컬럼을 찾아 date(날짜시각) 단일 컬럼으로 반환.
    - 우선순위: '측정일시' -> '일시' -> ('날짜','시간') 쌍 -> 'date'
    - 위가 없다면, 가능한 날짜형 컬럼을 자동 탐색(첫 번째 datetime-like 컬럼)
    """
    cols = df.columns.str.lower()

    # 1) '측정일시' 또는 '일시'
    for key in ["측정일시", "일시"]:
        if key in df.columns:
            out = pd.to_datetime(df[key], errors="coerce")
            return out

    # 2) 날짜 + 시간 쌍
    candidates_date = [c for c in df.columns if any(k in c.lower() for k in ["날짜", "date", "측정일"])]
    candidates_time = [c for c in df.columns if any(k in c.lower() for k in ["시간", "time", "측정시간"])]

    if len(candidates_date) > 0 and len(candidates_time) > 0:
        cdate = candidates_date[0]
        ctime = candidates_time[0]
        out = pd.to_datetime(
            df[cdate].astype(str).str.strip() + " " + df[ctime].astype(str).str.strip(),
            errors="coerce",
        )
        return out

    # 3) 'date' 단일
    if "date" in cols:
        out = pd.to_datetime(df.iloc[:, list(cols).index("date")], errors="coerce")
        return out

    # 4) datetime-like 자동 탐색
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed
        except Exception:
            continue

    return pd.Series(pd.NaT, index=df.index)

def standardize_columns(df: pd.DataFrame):
    """
    열 이름 통일 규칙 적용
    - 측정일시 → date
    - 온도(℃) → temp
    - 습도(%) → humid
    - CO₂(ppm) → co2
    - EC(dS/m) → ec
    - pH → ph
    - 수온(℃) → wt
    - 생육길이(cm) → length
    - 잎수(개) → leaves
    - 생중량(g) → wet_weight
    - 건중량(g) → dry_weight
    추가로, 유사 키워드도 최대한 포괄적으로 매핑
    """
    mapping = {
        # 날짜/일시
        "측정일시": "date", "일시": "date", "date": "date", "날짜": "date",

        # 환경변수
        "온도": "temp", "온도(℃)": "temp", "temp": "temp", "tmep": "temp",
        "습도": "humid", "습도(%)": "humid", "humidity": "humid", "humid": "humid",
        "co2": "co2", "co₂": "co2", "co₂(ppm)": "co2", "co2(ppm)": "co2",
        "ec": "ec", "ec(ds/m)": "ec",
        "ph": "ph", "pH": "ph",
        "수온": "wt", "수온(℃)": "wt", "w.t": "wt", "water_temp": "wt", "wt": "wt",

        # 생육지표
        "생육길이(cm)": "length", "길이(cm)": "length", "지상부 길이(cm)": "length", "length": "length",
        "잎수(개)": "leaves", "잎 수(장)": "leaves", "leaves": "leaves",
        "생중량(g)": "wet_weight", "지상부 생중량(g)": "wet_weight", "wet_weight": "wet_weight",
        "건중량(g)": "dry_weight", "지상부 건중량(g)": "dry_weight", "dry_weight": "dry_weight",
    }

    new_cols = []
    for c in df.columns:
        key = c.strip()
        low = key.lower()
        # 완전일치 우선
        if key in mapping:
            new_cols.append(mapping[key])
            continue
        # 소문자 비교
        if low in mapping:
            new_cols.append(mapping[low])
            continue

        # 포함 키워드 기반 휴리스틱
        if any(k in low for k in ["온도", "temp", "tmep"]):
            new_cols.append("temp")
        elif any(k in low for k in ["습도", "humid"]):
            new_cols.append("humid")
        elif "co2" in low or "co₂" in key.lower():
            new_cols.append("co2")
        elif low.startswith("ec"):
            new_cols.append("ec")
        elif low == "ph" or "ph" == key:
            new_cols.append("ph")
        elif any(k in low for k in ["수온", "water", "w.t", "wt"]):
            new_cols.append("wt")
        elif any(k in low for k in ["잎수", "잎 수", "leaves"]):
            new_cols.append("leaves")
        elif any(k in low for k in ["생육길이", "지상부 길이", "length"]):
            new_cols.append("length")
        elif any(k in low for k in ["생중량", "wet_weight"]):
            new_cols.append("wet_weight")
        elif any(k in low for k in ["건중량", "dry_weight"]):
            new_cols.append("dry_weight")
        elif any(k in low for k in ["측정일시", "일시", "날짜", "date"]):
            new_cols.append("date")
        else:
            new_cols.append(c)  # 보존

    df.columns = new_cols
    return df

def fmt_num(x, unit=None):
    if pd.isna(x):
        return "-"
    s = f"{x:,.2f}"
    return f"{s}{unit or ''}"

def report_types_and_missing(df: pd.DataFrame):
    """
    각 컬럼에 대해 dtype과 결측 비율(%)을 표로 반환.
    '숫자형 변환 가능성'을 판단하여 숫자형 열 리스트도 함께 반환.
    """
    out = []
    numeric_candidates = []
    for c in df.columns:
        series = df[c]
        # 결측 비율
        miss_ratio = (series.isna().mean() * 100.0) if series.size > 0 else 0.0
        # 숫자형 판정(강제 변환 시도)
        can_numeric = False
        if series.dtype.kind in "biufc":
            can_numeric = True
            numeric_candidates.append(c)
        else:
            try:
                pd.to_numeric(series, errors="raise")
                can_numeric = True
                numeric_candidates.append(c)
            except Exception:
                pass
        out.append(
            {
                "column": c,
                "dtype": str(series.dtype),
                "missing_ratio(%)": round(miss_ratio, 2),
                "numeric_candidate": can_numeric,
            }
        )
    report = pd.DataFrame(out)
    return report, numeric_candidates

def merge_on_date(dfs):
    """
    여러 데이터프레임을 'date' 열 기준으로 병합(outer)합니다.
    병합 전 각 DF에서 date를 구성하고, 컬럼명도 표준화합니다.
    """
    normalized = []
    for df in dfs:
        # 표준화
        df = standardize_columns(df.copy())
        # date 만들기
        if "date" not in df.columns:
            df["date"] = find_and_build_datetime(df)
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # date만 정확히 남기고, 중복된 date-단위의 다른 시간열 제거
        normalized.append(df)

    # 병합
    merged = None
    for i, d in enumerate(normalized):
        # 같은 이름 열 충돌 방지: date 제외하고 접미사 방지 위해 동일명은 그대로 두고 나중에 집계로 정리
        if merged is None:
            merged = d
        else:
            merged = pd.merge(merged, d, on="date", how="outer")

    # date 기준 정렬
    merged = merged.sort_values("date").reset_index(drop=True)

    # 가능한 중복 열 해결(동일 의미 컬럼이 여러 DF에서 올 때 평균 취함)
    # 대상 컬럼
    canonical = ["temp", "humid", "co2", "ec", "ph", "wt", "length", "leaves", "wet_weight", "dry_weight"]
    for name in canonical:
        same_cols = [c for c in merged.columns if c == name]
        # 이미 표준명 단일이면 skip
        if len(same_cols) <= 1:
            continue

    # 열 이름 중복 해결(동일명) -> 이미 같으니 그대로 두되, 아래에서 숫자 요약 시 자동 처리
    return merged

def numeric_summary(df: pd.DataFrame, numeric_cols: list):
    if not numeric_cols:
        return pd.DataFrame(columns=["column", "mean", "min", "max"])
    tmp = df[numeric_cols].agg(["mean", "min", "max"]).T.reset_index()
    tmp.columns = ["column", "mean", "min", "max"]
    return tmp

def corr_env_vs_growth(df: pd.DataFrame, env_cols: list, grow_cols: list):
    sub = df[["date"] + env_cols + grow_cols].copy()
    # 숫자형으로 강제 변환
    for c in env_cols + grow_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=env_cols + grow_cols, how="all")
    # 피어슨 상관: env x grow
    pairs = []
    for e in env_cols:
        for g in grow_cols:
            s = sub[[e, g]].dropna()
            if len(s) >= 3:
                r = s[e].corr(s[g])  # Pearson
                pairs.append({"env": e, "grow": g, "corr": r})
    corr_df = pd.DataFrame(pairs)
    return corr_df

def altair_heatmap(corr_df: pd.DataFrame, units_map: dict):
    if corr_df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["상관을 계산할 수 있는 데이터가 없습니다."]})).mark_text().encode(text="msg")
    # 표시 라벨
    corr_df = corr_df.copy()
    corr_df["corr_label"] = corr_df["corr"].round(2).astype(str)
    # Env/Grow 축 라벨에 단위 붙이기
    corr_df["env_label"] = corr_df["env"].apply(lambda k: f"{k} ({units_map.get(k,'')})" if units_map.get(k) else k)
    corr_df["grow_label"] = corr_df["grow"].apply(lambda k: f"{k} ({units_map.get(k,'')})" if units_map.get(k) else k)

    base = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X("env_label:N", title="환경 변수"),
        y=alt.Y("grow_label:N", title="생육 지표"),
        color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
        tooltip=["env", "grow", alt.Tooltip("corr:Q", format=".3f")]
    ).properties(title="환경–생육 상관계수 Heatmap", height=300)

    text = alt.Chart(corr_df).mark_text(baseline="middle").encode(
        x="env_label:N",
        y="grow_label:N",
        text="corr_label:N",
        color=alt.value("black")
    )

    return base + text

def altair_scatter_grid(df: pd.DataFrame, env_cols: list, grow_cols: list, units_map: dict):
    """
    '환경요소별 생육 영향도' – 선택된 모든 (env,grow) 쌍에 대해 산점도+단순회귀선.
    """
    # 준비
    plots = []
    for e in env_cols:
        for g in grow_cols:
            sub = df[[e, g]].dropna()
            if sub.shape[0] < 3:
                continue
            e_label = f"{e} ({units_map.get(e, '')})" if units_map.get(e) else e
            g_label = f"{g} ({units_map.get(g, '')})" if units_map.get(g) else g

            chart = alt.Chart(sub).mark_circle(opacity=0.5).encode(
                x=alt.X(f"{e}:Q", title=e_label),
                y=alt.Y(f"{g}:Q", title=g_label),
                tooltip=[alt.Tooltip(f"{e}:Q", format=",.2f"), alt.Tooltip(f"{g}:Q", format=",.2f")]
            ).properties(width=300, height=240)

            reg = chart.transform_regression(e, g).mark_line()
            plots.append(chart + reg)

    if not plots:
        return alt.Chart(pd.DataFrame({"msg": ["표시할 산점도가 없습니다."]})).mark_text().encode(text="msg")

    # 가로로 3개씩 타일링
    rows = []
    row = None
    for i, p in enumerate(plots):
        if i % 3 == 0:
            if row is not None:
                rows.append(row)
            row = p
        else:
            row = row | p
    if row is not None:
        rows.append(row)
    chart = rows[0]
    for r in rows[1:]:
        chart = chart & r
    return chart.properties(title="환경요소별 생육 영향도")

# ---------------------------------------------
# 1) 데이터 불러오기 (파일명 고정)
# ---------------------------------------------
with st.expander("📥 데이터 불러오기 및 요약", expanded=True):
    st.markdown("**불러오는 파일(순서 무관):** `co2.xlsx`, `ec-ph-wt.xlsx`, `humid.xlsx`, `tmep.xlsx`, `송도고_스마트팜_생육결과.xlsx`")

    paths = [
        "co2.xlsx",
        "ec-ph-wt.xlsx",
        "humid.xlsx",
        "tmep.xlsx",  # (오타로 제공되어도 그대로 사용)
        "송도고_스마트팜_생육결과.xlsx",
    ]

    loaded = []
    for p in paths:
        try:
            df = safe_read_table(p)
            st.success(f"불러오기 완료: {p} (shape={df.shape})")
            loaded.append(df)
        except Exception as e:
            st.error(f"불러오기 실패: {p} -> {e}")

    if not loaded:
        st.stop()

    # 병합
    merged = merge_on_date(loaded)
    # 표준화 (최종 한 번 더)
    merged = standardize_columns(merged)
    # date 정리
    if "date" not in merged.columns:
        merged["date"] = find_and_build_datetime(merged)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    # 컬럼 순서: date 먼저
    cols_ordered = ["date"] + [c for c in merged.columns if c != "date"]
    merged = merged[cols_ordered]

    st.subheader("처음 10행 미리보기")
    st.dataframe(merged.head(10), use_container_width=True)

    st.subheader("열 이름(정규화 전/후)")
    st.caption("※ 아래는 '현재(정규화 후)' 기준 열 목록입니다.")
    st.code(", ".join(list(merged.columns)), language="text")

    # 타입/결측 리포트
    report, numeric_candidates = report_types_and_missing(merged)
    st.subheader("데이터 타입 & 결측치 비율")
    st.dataframe(report, use_container_width=True)

    # 숫자형 요약
    st.subheader("숫자형 열 요약 (평균/최솟값/최댓값)")
    num_summary = numeric_summary(merged, numeric_candidates)
    st.dataframe(num_summary, use_container_width=True)

# ---------------------------------------------
# 사이드바 필터
# ---------------------------------------------
st.sidebar.header("🔎 데이터 필터")
min_date = merged["date"].min()
max_date = merged["date"].max()
date_range = st.sidebar.date_input(
    "날짜 범위 선택 (date)",
    value=(min_date.date() if pd.notna(min_date) else datetime.today().date(),
           max_date.date() if pd.notna(max_date) else datetime.today().date()),
)

grow_default = ["length"]
env_default = ["temp", "co2", "ec"]

growth_cols = ["length", "wet_weight", "dry_weight"]
env_cols = ["temp", "humid", "co2", "ec", "ph", "wt"]

selected_grow = st.sidebar.multiselect(
    "생육 지표 선택 (멀티 가능)",
    options=growth_cols,
    default=[c for c in grow_default if c in growth_cols],
)

selected_env = st.sidebar.multiselect(
    "환경 변수 선택 (멀티 가능)",
    options=env_cols,
    default=[c for c in env_default if c in env_cols],
)

# 날짜 필터 적용
if isinstance(date_range, tuple) and len(date_range) == 2:
    d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask = (merged["date"] >= pd.to_datetime(d1)) & (merged["date"] <= pd.to_datetime(d2) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    filt_df = merged.loc[mask].copy()
else:
    filt_df = merged.copy()

# 다운로드 (cp949)
csv_bytes = filt_df.to_csv(index=False, encoding="cp949").encode("cp949", errors="ignore")
st.sidebar.download_button(
    label="⬇️ 데이터 다운로드 (CSV, cp949)",
    data=csv_bytes,
    file_name="filtered_esmartfarm.csv",
    mime="text/csv",
    help="현재 필터링된 데이터를 CP949 인코딩 CSV로 저장합니다."
)

st.sidebar.markdown("---")
st.sidebar.caption("**데이터 출처:** 송도고등학교 스마트팜 프로젝트(2025, 극지연구소 × 인천교육청)")

# ---------------------------------------------
# 상단 메트릭 (7개, 4칸 격자 느낌으로 4+3 배치)
# ---------------------------------------------
st.markdown("### 📌 핵심 메트릭")
units = {"temp": "℃", "humid": "%", "co2": "ppm", "ec": "dS/m", "ph": "", "length": "cm", "wet_weight": "g"}
def avg_or_nan(col):
    return pd.to_numeric(filt_df[col], errors="coerce").mean() if col in filt_df.columns else np.nan

m1, m2, m3, m4 = st.columns(4)
m1.metric("평균 온도 (℃)", fmt_num(avg_or_nan("temp"), ""))
m2.metric("평균 습도 (%)", fmt_num(avg_or_nan("humid"), ""))
m3.metric("평균 CO₂ (ppm)", fmt_num(avg_or_nan("co2"), ""))
m4.metric("평균 EC (dS/m)", fmt_num(avg_or_nan("ec"), ""))

m5, m6, m7, _ = st.columns(4)
m5.metric("평균 pH", fmt_num(avg_or_nan("ph"), ""))
m6.metric("평균 생육길이 (cm)", fmt_num(avg_or_nan("length"), ""))
m7.metric("평균 생중량 (g)", fmt_num(avg_or_nan("wet_weight"), ""))

# ---------------------------------------------
# 상관 분석 (환경 vs 생육) + 히트맵 + Top5
# ---------------------------------------------
st.markdown("### 🔗 환경–생육 상관 분석")

units_map = {
    "temp": "℃",
    "humid": "%",
    "co2": "ppm",
    "ec": "dS/m",
    "ph": "",
    "wt": "℃",
    "length": "cm",
    "wet_weight": "g",
    "dry_weight": "g",
}

corr_df = corr_env_vs_growth(filt_df, selected_env, selected_grow)

c1, c2 = st.columns([2, 1])
with c1:
    st.altair_chart(altair_heatmap(corr_df, units_map), use_container_width=True)
with c2:
    if not corr_df.empty:
        top5 = corr_df.assign(abs_corr=corr_df["corr"].abs()).sort_values("abs_corr", ascending=False).head(5)
        st.markdown("**상관 상위 5개 (절대값 기준)**")
        st.dataframe(
            top5[["env", "grow", "corr"]].rename(columns={"env": "환경", "grow": "생육", "corr": "상관계수(피어슨)"}).round(3),
            use_container_width=True
        )
    else:
        st.info("상관을 계산할 수 있는 데이터가 충분하지 않습니다.")

# ---------------------------------------------
# '환경요소별 생육 영향도' 산점도 그리드
# ---------------------------------------------
st.markdown("### 📈 환경요소별 생육 영향도")
# 선택된 열만 숫자 변환
for c in set(selected_env + selected_grow):
    if c in filt_df.columns:
        filt_df[c] = pd.to_numeric(filt_df[c], errors="coerce")

st.altair_chart(altair_scatter_grid(filt_df, selected_env, selected_grow, units_map), use_container_width=True)

# ---------------------------------------------
# 푸터
# ---------------------------------------------
st.markdown("---")
st.caption("Made by 송도고 EcoSmartFarm Team, with AI support")
