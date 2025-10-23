# -*- coding: utf-8 -*-
from __future__ import annotations

import unicodedata
from pathlib import Path
from datetime import datetime
from io import StringIO

import pandas as pd
import numpy as np
import altair as alt

try:
    import streamlit as st
except Exception as e:
    raise RuntimeError("Streamlit import 실패. requirements.txt에 streamlit이 있는지 확인하세요.") from e


# =========================================================
# 인코딩 자동 감지형 CSV 로더
# =========================================================
def read_csv_kr(path):
    """
    cp949 실패 시 utf-8-sig → utf-8 → cp1252 → latin-1 순으로 재시도.
    그래도 실패하면 'ignore'로 깨지는 문자 무시하고 로드.
    df.attrs['encoding_used']에 사용된 인코딩 기록.
    """
    enc_candidates = ["cp949", "utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_err = None
    for enc in enc_candidates:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.attrs["encoding_used"] = enc
            return df
        except Exception as e:
            last_err = e
            continue
    # 최후 수단: 깨짐 무시
    with open(path, "rb") as f:
        raw = f.read()
    txt = raw.decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(txt))
    df.attrs["encoding_used"] = "utf-8(ignore)"
    return df


# =========================================================
# 견고한 파일 탐색기 (맥/리눅스 한글 정규화 차이 대응)
# =========================================================
def resolve_file(preferred_name: str, patterns: list[str]) -> Path:
    """
    preferred_name: 기대 파일명(정확 일치 우선)
    patterns: 대체 글롭 패턴
    """
    base = Path(__file__).parent.resolve()

    # 정확 일치 (NFC 정규화)
    pref_nfc = unicodedata.normalize("NFC", preferred_name)
    p = base / pref_nfc
    if p.exists():
        return p

    # 디렉토리 내 파일명 정규화 비교
    for child in base.iterdir():
        if child.is_file():
            name_nfc = unicodedata.normalize("NFC", child.name)
            if name_nfc == pref_nfc:
                return child

    # 패턴 검색
    for pat in patterns:
        for hit in base.glob(pat):
            return hit

    # 실패 시 디렉토리 목록 출력
    all_files = [unicodedata.normalize("NFC", c.name) for c in base.iterdir() if c.is_file()]
    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다: '{preferred_name}'\n"
        f"검색 폴더: {base}\n"
        f"패턴: {patterns}\n"
        f"현재 폴더 파일: {all_files}"
    )


# =========================================================
# 파일 경로 결정 (여기만 수정하면 파일명 바뀌어도 작동)
# =========================================================
ENV_FILE = resolve_file(
    "송도고 환경데이터 통합.csv",
    patterns=["*환경*통합*.csv", "*환경데이터*통합*.csv", "*환경데이터*.csv"]
)
GROW_FILE = resolve_file(
    "송도고 스마트팜 생육 결과.csv",
    patterns=["*스마트팜*생육*결과*.csv", "*생육*결과*.csv", "*스마트팜*.csv"]
)


UNITS = {
    "temp": "℃", "humid": "%", "co2": "ppm", "ec": "dS/m", "ph": "", "wt": "℃",
    "length": "cm", "wet_weight": "g", "dry_weight": "g"
}


# =========================================================
# 공통 유틸
# =========================================================
def fmt_num(x):
    return "-" if pd.isna(x) else f"{x:,.2f}"

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "측정일시":"date","일시":"date","날짜":"date","date":"date",
        "온도(℃)":"temp","온도":"temp","temp":"temp","tmep":"temp",
        "습도(%)":"humid","습도":"humid","humidity":"humid","humid":"humid",
        "co₂(ppm)":"co2","co2(ppm)":"co2","co2":"co2","co₂":"co2",
        "ec(ds/m)":"ec","ec":"ec","pH":"ph","ph":"ph",
        "수온(℃)":"wt","수온":"wt","w.t":"wt","water_temp":"wt","wt":"wt",
        "지상부 길이(cm)":"length","생육길이(cm)":"length","길이(cm)":"length","length":"length",
        "잎 수(장)":"leaves","잎수(개)":"leaves","leaves":"leaves",
        "지상부 생중량(g)":"wet_weight","생중량(g)":"wet_weight","wet_weight":"wet_weight",
        "지상부 건중량(g)":"dry_weight","건중량(g)":"dry_weight","dry_weight":"dry_weight",
    }
    new_cols = []
    for c in df.columns:
        key = c.strip()
        low = key.lower()
        if key in mapping: new_cols.append(mapping[key]); continue
        if low in mapping: new_cols.append(mapping[low]); continue
        if any(k in low for k in ["온도","temp","tmep"]): new_cols.append("temp")
        elif any(k in low for k in ["습도","humid"]): new_cols.append("humid")
        elif "co2" in low or "co₂" in key.lower(): new_cols.append("co2")
        elif low.startswith("ec"): new_cols.append("ec")
        elif low == "ph": new_cols.append("ph")
        elif any(k in low for k in ["수온","water","w.t","wt"]): new_cols.append("wt")
        elif any(k in low for k in ["잎수","잎 수","leaves"]): new_cols.append("leaves")
        elif any(k in low for k in ["생육길이","지상부 길이","length"]): new_cols.append("length")
        elif any(k in low for k in ["생중량","wet_weight"]): new_cols.append("wet_weight")
        elif any(k in low for k in ["건중량","dry_weight"]): new_cols.append("dry_weight")
        elif any(k in low for k in ["측정일시","일시","날짜","date"]): new_cols.append("date")
        else: new_cols.append(c)
    df.columns = new_cols
    return df

def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame({"column":[], "mean":[], "min":[], "max":[]})
    out = df[cols].agg(["mean","min","max"]).T.reset_index()
    out.columns = ["column","mean","min","max"]
    return out


# =========================================================
# 데이터 로더
# =========================================================
def parse_env_csv(path: str) -> pd.DataFrame:
    """
    환경 CSV 전용 파서: 변수별 서로 다른 날짜/시간 열을 단일 date로 결합.
    """
    raw = read_csv_kr(path)  # ← 인코딩 자동 처리
    # 예상 열 예: ['날짜','시간','co2','날짜.1','시간.1','ec','ph','w.t','날짜.2','시간.2','humid','날짜.3','시간.3','temp']
    var_map = {
        "co2":   ("날짜","시간","co2"),
        "ec":    ("날짜.1","시간.1","ec"),
        "ph":    ("날짜.1","시간.1","ph"),
        "wt":    ("날짜.1","시간.1","w.t"),
        "humid": ("날짜.2","시간.2","humid"),
        "temp":  ("날짜.3","시간.3","temp"),
    }
    frames = []
    for var, (dc, tc, vc) in var_map.items():
        if dc in raw.columns and tc in raw.columns and vc in raw.columns:
            df = raw[[dc, tc, vc]].copy()
            df["date"] = pd.to_datetime(
                df[dc].astype(str).str.strip() + " " + df[tc].astype(str).str.strip(),
                errors="coerce"
            )
            df = df.drop(columns=[dc, tc]).rename(columns={vc: var})
            df = df[~df["date"].isna()]
            frames.append(df.set_index("date"))

    if not frames:
        raw = standardize_columns(raw)
        if "date" not in raw.columns:
            raw["date"] = pd.to_datetime(raw.iloc[:,0], errors="coerce")
        return raw

    env = pd.concat(frames, axis=1).sort_index().reset_index()
    return env


def load_growth_csv(path: str) -> pd.DataFrame:
    # 1) 인코딩 자동 처리로 읽기
    df = read_csv_kr(path)

    # 2) 모든 컬럼명을 문자열로 통일 (숫자/None 대비)
    df.columns = [str(c).strip() for c in df.columns.tolist()]

    # 3) 엑셀→CSV 변환 시 생기는 'Unnamed' 계열 컬럼 제거 (실패해도 안전)
    try:
        df = df.loc[:, ~pd.Index(df.columns).str.contains(r"^Unnamed", regex=True)]
    except Exception:
        # 혹시라도 str.contains가 또 막히면 수동 필터로 대체
        df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

    # 4) 표준 컬럼명으로 통일
    df = standardize_columns(df)

    # 5) 숫자형 변환
    for c in ["length", "leaves", "wet_weight", "dry_weight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # (선택) date 컬럼이 문자열/숫자여도 날짜로 변환 시도
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df



# =========================================================
# 상관/시각화
# =========================================================
def corr_env_vs_growth(env_df: pd.DataFrame, grow_df: pd.DataFrame, env_cols, grow_cols):
    if "date" in env_df.columns and "date" in grow_df.columns:
        merged = pd.merge(env_df[["date"]+env_cols], grow_df[["date"]+grow_cols], on="date", how="inner")
    else:
        return pd.DataFrame(columns=["env","grow","corr"])
    pairs = []
    for e in env_cols:
        for g in grow_cols:
            s = merged[[e,g]].dropna()
            if len(s) >= 3:
                pairs.append({"env":e,"grow":g,"corr":s[e].corr(s[g])})
    return pd.DataFrame(pairs)

def altair_heatmap(corr_df: pd.DataFrame):
    if corr_df.empty:
        return alt.Chart(pd.DataFrame({"msg":["상관을 계산할 수 있는 데이터가 없습니다."]})).mark_text().encode(text="msg")
    corr_df = corr_df.copy()
    corr_df["corr_label"] = corr_df["corr"].round(2).astype(str)
    base = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X("env:N", title="환경 변수"),
        y=alt.Y("grow:N", title="생육 지표"),
        color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=(-1,1))),
        tooltip=[alt.Tooltip("env:N"), alt.Tooltip("grow:N"), alt.Tooltip("corr:Q", format=".3f")]
    ).properties(title="환경–생육 상관계수 Heatmap", height=300)
    text = alt.Chart(corr_df).mark_text(baseline="middle").encode(
        x="env:N", y="grow:N", text="corr_label:N"
    )
    return base + text

def scatter_grid(df, env_cols, grow_cols):
    charts = []
    for e in env_cols:
        for g in grow_cols:
            s = df[[e,g]].dropna()
            if len(s) < 3: 
                continue
            c = alt.Chart(s).mark_circle(opacity=0.5).encode(
                x=alt.X(f"{e}:Q", title=f"{e} ({UNITS.get(e,'')})"),
                y=alt.Y(f"{g}:Q", title=f"{g} ({UNITS.get(g,'')})"),
                tooltip=[alt.Tooltip(f"{e}:Q", format=",.2f"), alt.Tooltip(f"{g}:Q", format=",.2f")]
            ).properties(width=300, height=240)
            charts.append(c + c.transform_regression(e, g).mark_line())
    if not charts:
        return alt.Chart(pd.DataFrame({"msg":["표시할 산점도가 없습니다."]})).mark_text().encode(text="msg")
    row = None; rows = []
    for i, ch in enumerate(charts):
        row = ch if i % 3 == 0 else row | ch
        if i % 3 == 2: rows.append(row)
    if row is not None and (len(charts) % 3 != 0): rows.append(row)
    out = rows[0]
    for r in rows[1:]: out = out & r
    return out.properties(title="환경요소별 생육 영향도")


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="송도고 EcoSmartFarm 대시보드", page_icon="🌿", layout="wide")
st.title("🌿 송도고 EcoSmartFarm 환경·생육 데이터 대시보드")

with st.expander("📥 데이터 불러오기 및 요약", expanded=True):
    # 환경 데이터
    try:
        env_df = parse_env_csv(ENV_FILE)
        st.success(f"환경 데이터 불러오기 완료: {ENV_FILE} (shape={env_df.shape})")
        st.caption(f"환경 CSV 인코딩: {getattr(env_df, 'attrs', {}).get('encoding_used', 'unknown')}")
    except Exception as e:
        st.error(f"환경 데이터 불러오기 실패: {e}")
        st.stop()

    # 생육 데이터
    try:
        grow_df = load_growth_csv(GROW_FILE)
        st.success(f"생육 데이터 불러오기 완료: {GROW_FILE} (shape={grow_df.shape})")
        st.caption(f"생육 CSV 인코딩: {getattr(grow_df, 'attrs', {}).get('encoding_used', 'unknown')}")
    except Exception as e:
        st.warning(f"생육 데이터 불러오기 실패(선택사항): {e}")
        grow_df = pd.DataFrame()

    # 표준화 & 날짜 변환
    env_df = standardize_columns(env_df)
    if "date" in env_df.columns:
        env_df["date"] = pd.to_datetime(env_df["date"], errors="coerce")

    st.subheader("환경 데이터 처음 10행")
    st.dataframe(env_df.head(10), use_container_width=True)

    if not grow_df.empty:
        st.subheader("생육 데이터 처음 10행")
        st.dataframe(grow_df.head(10), use_container_width=True)

    # 타입/결측 보고
    def report(df: pd.DataFrame):
        rows = []
        for c in df.columns:
            miss = df[c].isna().mean()*100 if len(df) else 0
            can_num = df[c].dtype.kind in "biufc"
            if not can_num:
                try:
                    pd.to_numeric(df[c], errors="raise"); can_num = True
                except Exception:
                    pass
            rows.append({"column":c, "dtype":str(df[c].dtype), "missing_ratio(%)":round(miss,2), "numeric_candidate":can_num})
        return pd.DataFrame(rows)

    st.subheader("환경 데이터 타입 & 결측 비율")
    st.dataframe(report(env_df), use_container_width=True)

    if not grow_df.empty:
        st.subheader("생육 데이터 타입 & 결측 비율")
        st.dataframe(report(grow_df), use_container_width=True)

    st.subheader("숫자형 요약 (평균/최솟값/최댓값)")
    env_num = [c for c in ["temp","humid","co2","ec","ph","wt"] if c in env_df.columns]
    st.write("환경:", numeric_summary(env_df, env_num))

    if not grow_df.empty:
        grow_num = [c for c in ["length","wet_weight","dry_weight","leaves"] if c in grow_df.columns]
        st.write("생육:", numeric_summary(grow_df, grow_num))


# ------------------- 사이드바 필터 -------------------
st.sidebar.header("🔎 데이터 필터")
if "date" in env_df.columns and env_df["date"].notna().any():
    min_date = env_df["date"].min()
    max_date = env_df["date"].max()
    start, end = st.sidebar.date_input(
        "날짜 범위 선택 (환경 데이터 기준)",
        (min_date.date(), max_date.date()) if pd.notna(min_date) else (datetime.today().date(), datetime.today().date())
    )
    mask = (env_df["date"] >= pd.to_datetime(start)) & \
           (env_df["date"] <= pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    env_f = env_df.loc[mask].copy()
else:
    st.sidebar.info("환경 데이터에 date가 없어 전체 사용합니다.")
    env_f = env_df.copy()

# 다운로드 (cp949)
csv_bytes = env_f.to_csv(index=False, encoding="cp949").encode("cp949", errors="ignore")
st.sidebar.download_button("⬇️ 데이터 다운로드 (CSV, cp949)", data=csv_bytes, file_name="filtered_env.csv", mime="text/csv")
st.sidebar.markdown("---")
st.sidebar.caption("**데이터 출처:** 송도고등학교 스마트팜 프로젝트(2025, 극지연구소 × 인천교육청)")


# ------------------- 메트릭 -------------------
st.markdown("### 📌 핵심 메트릭")
def avg(col, df):
    return pd.to_numeric(df[col], errors="coerce").mean() if col in df.columns else np.nan

m1,m2,m3,m4 = st.columns(4)
m1.metric("평균 온도 (℃)", fmt_num(avg("temp", env_f)))
m2.metric("평균 습도 (%)", fmt_num(avg("humid", env_f)))
m3.metric("평균 CO₂ (ppm)", fmt_num(avg("co2", env_f)))
m4.metric("평균 EC (dS/m)", fmt_num(avg("ec", env_f)))
m5,m6,m7,_ = st.columns(4)
m5.metric("평균 pH", fmt_num(avg("ph", env_f)))
m6.metric("평균 생육길이 (cm)", fmt_num(avg("length", grow_df) if not grow_df.empty else np.nan))
m7.metric("평균 생중량 (g)", fmt_num(avg("wet_weight", grow_df) if not grow_df.empty else np.nan))


# ------------------- 상관 & 시각화 -------------------
st.markdown("### 🔗 환경–생육 상관 분석")
env_cols_sel = [c for c in ["temp","humid","co2","ec","ph","wt"] if c in env_f.columns]
grow_cols_sel = [c for c in ["length","wet_weight","dry_weight"] if not grow_df.empty and c in grow_df.columns]

corr_df = corr_env_vs_growth(env_f, grow_df, env_cols_sel, grow_cols_sel) if grow_cols_sel else pd.DataFrame(columns=["env","grow","corr"])

c1, c2 = st.columns([2,1])
with c1:
    st.altair_chart(altair_heatmap(corr_df), use_container_width=True)
with c2:
    if corr_df.empty:
        st.info("생육 데이터에 날짜 정보가 없거나 매칭 가능한 데이터가 부족하여 상관 분석을 표시하지 않습니다.")
    else:
        top5 = corr_df.assign(abs_corr=corr_df["corr"].abs()).sort_values("abs_corr", ascending=False).head(5)
        st.markdown("**상관 상위 5개 (절대값 기준)**")
        st.dataframe(top5[["env","grow","corr"]].round(3), use_container_width=True)

st.markdown("### 📈 환경요소별 생육 영향도")
if not corr_df.empty:
    merged_for_plot = pd.merge(
        env_f[["date"]+env_cols_sel], grow_df[["date"]+grow_cols_sel],
        on="date", how="inner"
    ).dropna()
    st.altair_chart(scatter_grid(merged_for_plot, env_cols_sel, grow_cols_sel), use_container_width=True)
else:
    st.info("날짜가 매칭되는 환경–생육 데이터가 없어 산점도를 표시하지 않습니다.")

st.markdown("---")
st.caption("Made by 송도고 EcoSmartFarm Team, with AI support")
