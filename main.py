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
    return report
