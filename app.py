# ── PART 1: IMPORTS & LIBRARY SETUP ───────────────────────────────────────
import os
import datetime
# Suppress verbose logs from underlying libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st            # UI framework
import requests                   # HTTP client, replaces Selenium

import time                       # Timing utilities (if needed)
import pandas as pd               # Data manipulation
import numpy as np                # Numerical operations
from io import BytesIO            # In-memory buffer for downloads
from urllib.parse import urlparse # Parse URLs to derive sub-categories
from bs4 import BeautifulSoup     # HTML parsing
import matplotlib.pyplot as plt   # Plotting charts

# You can optionally set a default matplotlib style:
# plt.style.use('default')

# ── PART 2: SIMPLE HTTP FETCH ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_page(url: str) -> str:
    """
    Fetch a page’s HTML via HTTP GET and cache the result.
    Uses a standard User-Agent header to mimic a real browser.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise an error on bad status
    return response.text

# ── PART 4: SCRAPING & SESSION-STATE MANAGEMENT ────────────────────────────
# Initialize session-state containers if they don't exist
if "data" not in st.session_state:
    st.session_state.data = []
if "seen" not in st.session_state:
    st.session_state.seen = set()
if "stop" not in st.session_state:
    st.session_state.stop = False

# Auto-scrape when button pressed
if btn_auto:
    # Reset state for a fresh run
    st.session_state.stop = False
    st.session_state.data.clear()
    st.session_state.seen.clear()

    # Decide URL pagination separator
    sep = "&p=" if "?" in url_base else "?p="

    with st.spinner("Scraping pages…"):
        for p in range(1, pages_to_scrape + 1):
            if st.session_state.stop:
                status.warning(f"⏸️ Stopped at page {p-1}")
                break

            full_url = f"{url_base}{sep}{p}"
            # Fetch HTML over HTTP instead of using Selenium
            html  = fetch_page(full_url)
            batch = scrape_html(html, p)

            if not batch:
                status.info(f"No items found on page {p} – ending scrape.")
                break

            # Add new items and report progress
            st.session_state.data.extend(batch)
            status.success(
                f"Page {p}: +{len(batch)} items (total {len(st.session_state.data)})"
            )
        else:
            status.success("✅ Auto-scrape complete!")

# Stop button to halt scraping loop
if btn_stop:
    st.session_state.stop = True

# ── PART 5: DATA CLEANING & FILTERING ────────────────────────────────────────
if st.session_state.data:
    # Load raw data into a DataFrame
    df = pd.DataFrame(st.session_state.data)

    # Clean & convert Review Count to integer
    df["Review Count"] = (
        df["Review Count"]
        .str.replace(r"[^\d]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Convert Rating to numeric
    df["Rating"] = (
        pd.to_numeric(df["Rating"], errors="coerce")
        .fillna(0)
    )

    # Extract numeric values from price strings
    df["SellingVal"] = (
        df["Selling Price"]
        .str.replace(r"[^\d]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df["MRPVal"] = (
        df["MRP"]
        .str.replace(r"[^\d]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Compute discount percentage safely
    df["ComputedDiscount"] = (
        (df["MRPVal"] - df["SellingVal"])
        .clip(lower=0)
        .div(df["MRPVal"].replace(0, np.nan))
        .fillna(0)
        .clip(0, 1)
        * 100
    )

    # Sidebar filters: Company
    companies = sorted(df["Company"].unique())
    company_filter = st.sidebar.multiselect("Filter by Company", options=companies)
    if company_filter:
        df = df[df["Company"].isin(company_filter)]

    # Sidebar filters: Sub-category
    subcats = sorted(df["Sub-category"].unique())
    subcat_filter = st.sidebar.multiselect("Filter by Sub-category", options=subcats)
    if subcat_filter:
        df = df[df["Sub-category"].isin(subcat_filter)]

    # Sidebar filters: Discount range (uses disc_min, disc_max)
    df = df[
        (df["ComputedDiscount"] >= disc_min) &
        (df["ComputedDiscount"] <= disc_max)
    ]

    # Sidebar filters: Ads
    if ads_filter == "Only Ads":
        df = df[df["Ads"] == "Yes"]
    elif ads_filter == "Only Non-Ads":
        df = df[df["Ads"] == "No"]

    # Assign to filtered_df for downstream use
    filtered_df = df.copy()
else:
    # Ensure filtered_df exists even if no data
    filtered_df = pd.DataFrame()
# ── PART 6: DASHBOARD DISPLAY ───────────────────────────────────────────────
if not filtered_df.empty:
    # 6.A: Executive Summary & Insights
    st.subheader("📝 Executive Summary & Insights")
    col1, col2 = st.columns(2)

    # Overall metrics
    total_skus           = filtered_df["Product URL"].nunique()
    total_names          = filtered_df["Product"].nunique()
    pct_ads              = filtered_df["Ads"].eq("Yes").mean() * 100
    avg_discount         = filtered_df["ComputedDiscount"].mean()
    top_by_reviews       = filtered_df.nlargest(1, "Review Count").iloc[0]
    top_by_rating        = filtered_df.nlargest(1, "Rating").iloc[0]

    with col1:
        st.markdown(f"""
**Total SKUs:** {total_skus}  
**Distinct Products:** {total_names}  
**% Advertised:** {pct_ads:.1f}%  
**Avg Discount:** {avg_discount:.1f}%  
**Top by Reviews:** [{top_by_reviews['Product']}]({top_by_reviews['Product URL']})  
**Top by Rating:** [{top_by_rating['Product']}]({top_by_rating['Product URL']})  
""")

    # Kushal-specific metrics
    kushal_df = filtered_df[filtered_df["Company"].str.contains("Kushal", case=False)]
    if not kushal_df.empty:
        k_skus       = kushal_df["Product URL"].nunique()
        k_names      = kushal_df["Product"].nunique()
        k_pct_ads    = kushal_df["Ads"].eq("Yes").mean() * 100
        k_avg_disc   = kushal_df["ComputedDiscount"].mean()
        k_top_rev    = kushal_df.nlargest(1, "Review Count").iloc[0]
        k_top_rat    = kushal_df.nlargest(1, "Rating").iloc[0]
        with col2:
            st.markdown(f"""
**Kushal SKUs:** {k_skus}  
**Kushal Products:** {k_names}  
**% Kushal Ads:** {k_pct_ads:.1f}%  
**Avg Kushal Discount:** {k_avg_disc:.1f}%  
**Kushal Top Reviews:** [{k_top_rev['Product']}]({k_top_rev['Product URL']})  
**Kushal Top Rating:** [{k_top_rat['Product']}]({k_top_rat['Product URL']})  
""")
    else:
        with col2:
            st.markdown("**No Kushal data**")

    # 6.B: Visual Explorations
    st.subheader("📊 Category & Branding Overview")
    v1, v2 = st.columns(2)

    # SKU Count by Sub-category
    with v1:
        st.markdown("**SKU Count by Sub-category**")
        counts = filtered_df["Sub-category"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    # Top 10 Companies
    with v2:
        st.markdown("**Top 10 Companies by SKU**")
        top10 = filtered_df["Company"].value_counts().head(10)
        fig, ax = plt.subplots()
        ax.bar(top10.index, top10.values)
        ax.set_xticklabels(top10.index, rotation=45, ha="right")
        st.pyplot(fig, use_container_width=True)

    # 6.C: Top Reviewed, Rated, Discounted Tables
    st.subheader("🔝 Top Products")
    t1, t2, t3 = st.columns(3)

    # Top reviewed
    top_rev_df = filtered_df.nlargest(10, "Review Count")[["Product","Company","Review Count","Product URL"]]
    t1.markdown(hyperlink_products(top_rev_df).to_html(escape=False, index=False), unsafe_allow_html=True)

    # Top rated (>=50 reviews)
    top_rat_df = (
        filtered_df[filtered_df["Review Count"]>=50]
        .nlargest(10, "Rating")[["Product","Company","Rating","Review Count","Product URL"]]
    )
    t2.markdown(hyperlink_products(top_rat_df).to_html(escape=False, index=False), unsafe_allow_html=True)

    # Top discounted
    top_disc_df = filtered_df.nlargest(10, "ComputedDiscount")[["Product","Company","ComputedDiscount","Product URL"]]
    top_disc_df = top_disc_df.rename(columns={"ComputedDiscount":"Discount (%)"})
    t3.markdown(hyperlink_products(top_disc_df).to_html(escape=False, index=False), unsafe_allow_html=True)

# ── PART 7: DATA EXPORT & FULL TABLE ────────────────────────────────────────
if st.session_state.data:
    st.markdown("---")
    st.subheader("📋 Download & Full Data Table")

    # 1) Prepare Excel in-memory
    buf = BytesIO()
    pd.DataFrame(st.session_state.data).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    # 2) Filename input
    fn = st.text_input("Export filename", "myntra_insights.xlsx")
    if not fn.lower().endswith(".xlsx"):
        fn += ".xlsx"

    # 3) Download button
    st.download_button(
        label="📥 Download Excel",
        data=buf.getvalue(),
        file_name=fn,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # 4) Display clickable full table
    full_df = pd.DataFrame(st.session_state.data)
    html_df = hyperlink_products(full_df)[
        ["Product","Company","Review Count","Rating","ComputedDiscount","Product URL"]
    ]
    st.markdown(
        html_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
