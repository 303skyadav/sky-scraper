import os
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow/Selenium logs

import streamlit as st
import time
import pandas as pd
import numpy as np
from io import BytesIO
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt

# ── STREAMLIT CONFIG ────────────────────────────────────────────────────────
st.set_page_config(page_title="SKY Scraper & Analyst V3.0", layout="wide")
st.title("🚀 SKY Scraper & Analyst V3.0")


# ── DRIVER FACTORY ──────────────────────────────────────────────────────────
@st.cache_resource
def get_driver():
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("detach", True)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

# instantiate once
driver = get_driver()

# ── HELPER: Scrape HTML into dicts ──────────────────────────────────────────
def scrape_html(html, page_no):
    """
    Parse a page’s HTML and return a list of product dictionaries:
    Page Number, Placement Index, Product, Company, Review Count, Rating,
    Ads flag, Selling Price, MRP, Raw Discount, Sub-category, Product URL.
    """
    soup = BeautifulSoup(html, "html.parser")
    batch = []

    for placement_idx, item in enumerate(soup.select("li.product-base"), start=1):
        a = item.select_one("a[href]")
        if not a:
            continue

        # build absolute URL
        href = a["href"].split("?")[0]
        url  = "https://www.myntra.com" + (href if href.startswith("/") else "/" + href)
        if url in st.session_state.seen:
            continue
        st.session_state.seen.add(url)

        # derive sub-category
        subcat = urlparse(url).path.strip("/").split("/")[0]
        txt    = lambda sel: sel.text.strip() if sel else ""

        # watermark indicates ads
        wm = item.select_one(".product-waterMark")

        batch.append({
            "Page Number":     page_no,
            "Placement Index": placement_idx,
            "Product":         txt(item.select_one(".product-product")),
            "Company":         txt(item.select_one(".product-brand")),
            "Review Count":    txt(item.select_one(".product-ratingsCount")).replace("|",""),
            "Rating":          txt(item.select_one(".product-ratingsContainer span")),
            "Ads":             "Yes" if wm and wm.text.strip().upper()=="AD" else "No",
            "Selling Price":   txt(item.select_one(".product-discountedPrice") or item.select_one(".product-price")),
            "MRP":             txt(item.select_one(".product-strike")) or 
                               txt(item.select_one(".product-discountPercentage")) or "",
            "Raw Discount":    txt(item.select_one(".product-discountPercentage")) or "0% OFF",
            "Sub-category":    subcat,
            "Product URL":     url
        })

    return batch

# ── HELPER: Hyperlink product names ─────────────────────────────────────────
def hyperlink_products(df):
    """
    Return a copy of df where 'Product' is replaced by an <a> link
    to its 'Product URL', so names become clickable.
    """
    df = df.copy()
    df["Product"] = df.apply(
        lambda r: f'<a href="{r["Product URL"]}" target="_blank">{r["Product"]}</a>',
        axis=1
    )
    return df

# ── PART 2: USER INPUTS & SIDEBAR CONTROLS ───────────────────────────────────

# Sidebar instructions
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Enter a Myntra category URL & pages to scrape.  
2. Click **Auto Scrape**.  
3. Filter by company, sub-category, discount, or ads.  
4. Pick a focused view & sub-category for deeper insights.
""")

# Sidebar filters
disc_min, disc_max = st.sidebar.slider(
    "Discount Range (%)", 
    min_value=0, 
    max_value=100, 
    value=(0, 100)
)
ads_filter = st.sidebar.selectbox(
    "Ads Filter", 
    ["All", "Only Ads", "Only Non-Ads"]
)

# Main UI controls
_, col_url, col_pages, _ = st.columns([1, 4, 2, 1])
url_base = col_url.text_input(
    "Base URL",
    "https://www.myntra.com/women-jewellery"
)
pages_to_scrape = col_pages.number_input(
    "Pages to Scrape",
    min_value=1,
    value=10,
    step=1,
    help="Enter 0 to run until no more items appear"
)

_, col_auto, col_stop, _ = st.columns([1, 2, 2, 1])
btn_auto = col_auto.button("🤖 Auto Scrape")
btn_stop = col_stop.button("🛑 Stop")

# Placeholder for status messages
status = st.empty()

# ── PART 3: SCRAPING & SESSION-STATE MANAGEMENT ─────────────────────────────

# Initialize session-state containers
if "data" not in st.session_state:
    st.session_state.data = []
if "seen" not in st.session_state:
    st.session_state.seen = set()
if "stop" not in st.session_state:
    st.session_state.stop = False

# Auto‐scrape loop
if btn_auto:
    # Reset on each new run
    st.session_state.stop = False
    st.session_state.data.clear()
    st.session_state.seen.clear()

    # Determine URL parameter separator (?p= or &p=)
    sep = "&p=" if "?" in url_base else "?p="

    with st.spinner("Scraping pages…"):
        for p in range(1, pages_to_scrape + 1):
            if st.session_state.stop:
                status.warning(f"⏸️ Stopped at page {p-1}")
                break

            full_url = f"{url_base}{sep}{p}"
            driver.get(full_url)
            time.sleep(2)  # allow page to load

            batch = scrape_html(driver.page_source, p)
            if not batch:
                status.info(f"No items found on page {p} – ending scrape.")
                break

            st.session_state.data.extend(batch)
            status.success(f"Page {p}: +{len(batch)} items (total {len(st.session_state.data)})")

        else:
            status.success("✅ Auto-scrape complete!")

# Stop button sets flag to halt loop
if btn_stop:
    st.session_state.stop = True


# ── PART 4: DATA CLEANING & FILTERING ────────────────────────────────────────

if st.session_state.data:
    # 1) Load into DataFrame
    df = pd.DataFrame(st.session_state.data)

    # 2) Clean & convert fields
    df["Review Count"] = (
        df["Review Count"]
        .str.replace(r"[^\d]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["Rating"] = (
        pd.to_numeric(df["Rating"], errors="coerce")
        .fillna(0)
    )

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

    # 3) Compute discount percentage
    df["ComputedDiscount"] = (
        (df["MRPVal"] - df["SellingVal"]).clip(lower=0)
        .div(df["MRPVal"].replace(0, np.nan))
        .fillna(0)
        .clip(0,1)
        * 100
    )

    # 4) Apply sidebar filters
    # Filter by company
    companies = sorted(df["Company"].unique())
    company_filter = st.sidebar.multiselect("Filter by Company", options=companies)
    if company_filter:
        df = df[df["Company"].isin(company_filter)]

    # Filter by sub-category
    subcats = sorted(df["Sub-category"].unique())
    subcat_filter = st.sidebar.multiselect("Filter by Sub-category", options=subcats)
    if subcat_filter:
        df = df[df["Sub-category"].isin(subcat_filter)]

    # Filter by discount range
    df = df[
        (df["ComputedDiscount"] >= disc_min)
        & (df["ComputedDiscount"] <= disc_max)
    ]

    # Filter by ads
    if ads_filter == "Only Ads":
        df = df[df["Ads"] == "Yes"]
    elif ads_filter == "Only Non-Ads":
        df = df[df["Ads"] == "No"]

# ── PART 5: DASHBOARD DISPLAY ───────────────────────────────────────────────
    # ── PART 5.A: EXECUTIVE SUMMARY & KEY METRICS ────────────────────────────
    st.subheader("📝 Executive Summary & Insights")
    c0, c1, c2, c3 = st.columns([1, 2, 2, 1])

    # Focused view selector
    view_option = c1.selectbox(
        "Select focused view",
        [
            "All Data",
            "Top Advertisers",
            "Top Reviewed Products",
            "Top Rated Products",
            "Highest Discounted Products",
        ],
        key="view"
    )
    if view_option == "Top Advertisers":
        temp_df = df[df["Ads"] == "Yes"]
    elif view_option == "Top Reviewed Products":
        temp_df = df.nlargest(100, "Review Count")
    elif view_option == "Top Rated Products":
        temp_df = df[df["Review Count"] >= 50].nlargest(100, "Rating")
    elif view_option == "Highest Discounted Products":
        temp_df = df.nlargest(100, "ComputedDiscount")
    else:
        temp_df = df.copy()

    # Sub-category filter
    subcats2      = ["All"] + sorted(temp_df["Sub-category"].unique())
    subcat_option = c2.selectbox("Select sub-category", subcats2, key="subcat")
    filtered_df   = temp_df if subcat_option == "All" else temp_df[temp_df["Sub-category"] == subcat_option]

    # Compute metrics
    total_skus           = filtered_df["Product URL"].nunique()
    total_distinct_names = filtered_df["Product"].nunique()
    pct_ads              = filtered_df["Ads"].eq("Yes").mean() * 100
    avg_disc             = filtered_df["ComputedDiscount"].mean()

    top_rev_row  = filtered_df.nlargest(1, "Review Count").iloc[0]
    top_rat_row  = filtered_df.nlargest(1, "Rating").iloc[0]
    top_rev_link = f'<a href="{top_rev_row["Product URL"]}" target="_blank">{top_rev_row["Product"]}</a>'
    top_rat_link = f'<a href="{top_rat_row["Product URL"]}" target="_blank">{top_rat_row["Product"]}</a>'

    top_cat_rev = filtered_df.groupby("Sub-category")["Review Count"].sum().idxmax()
    top_cat_rat = filtered_df.groupby("Sub-category")["Rating"].mean().idxmax()

    # Kushal-specific metrics
    kushal_df             = filtered_df[filtered_df["Company"].str.contains("Kushal", case=False)]
    k_total_skus          = kushal_df["Product URL"].nunique()
    k_total_names         = kushal_df["Product"].nunique()
    k_pct_ads             = kushal_df["Ads"].eq("Yes").mean() * 100 if k_total_skus else 0
    k_avg_disc            = kushal_df["ComputedDiscount"].mean() if k_total_skus else 0
    if k_total_skus:
        k_rev_row       = kushal_df.nlargest(1, "Review Count").iloc[0]
        k_rat_row       = kushal_df.nlargest(1, "Rating").iloc[0]
        k_rev_link      = f'<a href="{k_rev_row["Product URL"]}" target="_blank">{k_rev_row["Product"]}</a>'
        k_rat_link      = f'<a href="{k_rat_row["Product URL"]}" target="_blank">{k_rat_row["Product"]}</a>'
        k_cat_rev       = kushal_df.groupby("Sub-category")["Review Count"].sum().idxmax()
        k_cat_rat       = kushal_df.groupby("Sub-category")["Rating"].mean().idxmax()
    else:
        k_rev_link = k_rat_link = "N/A"
        k_cat_rev = k_cat_rat = "N/A"

    # Render metrics side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div style="text-align:center;">
  <h4>Overall Metrics</h4>
  <ul style="display:inline-block; text-align:left; padding-left:20px;">
    <li><strong>Total SKUs (unique URLs):</strong> {total_skus}</li>
    <li><strong>Total Distinct Names:</strong> {total_distinct_names}</li>
    <li><strong>% Advertised:</strong> {pct_ads:.1f}%</li>
    <li><strong>Average Discount:</strong> {avg_disc:.1f}%</li>
    <li><strong>Top by Reviews:</strong> {top_rev_link}</li>
    <li><strong>Top by Rating:</strong> {top_rat_link}</li>
    <li><strong>Top Category (Reviews):</strong> {top_cat_rev}</li>
    <li><strong>Top Category (Rating):</strong> {top_cat_rat}</li>
  </ul>
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div style="text-align:center;">
  <h4>Kushal Metrics</h4>
  <ul style="display:inline-block; text-align:left; padding-left:20px;">
    <li><strong>Total SKUs (unique URLs):</strong> {k_total_skus}</li>
    <li><strong>Total Distinct Names:</strong> {k_total_names}</li>
    <li><strong>% Advertised:</strong> {k_pct_ads:.1f}%</li>
    <li><strong>Average Discount:</strong> {k_avg_disc:.1f}%</li>
    <li><strong>Top by Reviews:</strong> {k_rev_link}</li>
    <li><strong>Top by Rating:</strong> {k_rat_link}</li>
    <li><strong>Top Category (Reviews):</strong> {k_cat_rev}</li>
    <li><strong>Top Category (Rating):</strong> {k_cat_rat}</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    # ── PART 5.B: VISUAL EXPLORATIONS (CHARTS) ───────────────────────────────
    st.subheader("📊 Category & Branding Overview")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**SKU Count by Sub-category**")
        sub_counts = filtered_df["Sub-category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(sub_counts.index, sub_counts.values)
        ax.set_xticklabels(sub_counts.index, rotation=45, ha="right")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)
    with r1c2:
        st.markdown("**Top 10 Companies by SKU Count**")
        top10 = filtered_df["Company"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(top10.index, top10.values)
        ax.set_xticklabels(top10.index, rotation=45, ha="right")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)

    adv = filtered_df[filtered_df["Ads"] == "Yes"]
    if not adv.empty:
        st.subheader("🔍 Advertised Products Analysis")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Top 10 Advertisers**")
            adv_counts = adv["Company"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(adv_counts.index, adv_counts.values)
            ax.set_xticklabels(adv_counts.index, rotation=45, ha="right")
            ax.set_ylabel("SKU Count")
            st.pyplot(fig, use_container_width=True)
        with a2:
            st.markdown("**Discount vs Rating (Ads)**")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.scatter(adv["ComputedDiscount"], adv["Rating"], alpha=0.6)
            ax.set_xlabel("Discount (%)")
            ax.set_ylabel("Rating")
            st.pyplot(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("💡 Review vs Rating vs Discount (Bubble Chart)")
        fig, ax = plt.subplots(figsize=(6, 3))
        sizes = filtered_df["Review Count"] / filtered_df["Review Count"].max() * 200
        ax.scatter(filtered_df["ComputedDiscount"], filtered_df["Rating"], s=sizes, alpha=0.5)
        ax.set_xlabel("Discount (%)")
        ax.set_ylabel("Rating")
        st.pyplot(fig, use_container_width=True)
    with r2c2:
        st.subheader("🛍️ Brand Comparison: Avg Rating vs Avg Discount")
        brand_stats = (
            filtered_df
            .groupby("Company")
            .agg(AvgRating=("Rating", "mean"), AvgDiscount=("ComputedDiscount", "mean"), SKUCount=("Product","count"))
            .sort_values("SKUCount", ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(6, 3))
        sizes2 = brand_stats["SKUCount"] / brand_stats["SKUCount"].max() * 200
        ax.scatter(brand_stats["AvgDiscount"], brand_stats["AvgRating"], s=sizes2, alpha=0.7)
        for comp in brand_stats.index:
            ax.annotate(
                comp,
                (brand_stats.loc[comp,"AvgDiscount"], brand_stats.loc[comp,"AvgRating"]),
                textcoords="offset points", xytext=(5,5)
            )
        ax.set_xlabel("Avg Discount (%)")
        ax.set_ylabel("Avg Rating")
        st.pyplot(fig, use_container_width=True)

    # ── PART 5.C: DETAILED TABLES ──────────────────────────────────────────────
    # Champions & Laggards
    filtered_df["Score"] = filtered_df["Review Count"] * filtered_df["Rating"]
    ch, lg = st.columns(2)
    with ch:
        st.subheader("🏆 Top 10 Champions")
        champions = filtered_df.nlargest(10, "Score")[["Product","Company","Review Count","Rating","Product URL"]]
        champs_html = hyperlink_products(champions)[["Product","Company","Review Count","Rating"]]
        st.markdown(champs_html.to_html(escape=False, index=False), unsafe_allow_html=True)
    with lg:
        st.subheader("⚠️ Top 10 Laggards")
        laggards = (
            filtered_df
            .nsmallest(10, "Rating")[["Product","Company","ComputedDiscount","Rating","Product URL"]]
            .rename(columns={"ComputedDiscount":"Discount (%)"})
        )
        lag_html = hyperlink_products(laggards)[["Product","Company","Discount (%)","Rating"]]
        st.markdown(lag_html.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Top Reviewed / Rated / Discounted
    st.subheader("🔝 Top Reviewed, Rated & Discounted Products")
    t3, t4, t5 = st.columns(3)

    top_rev_df = filtered_df.nlargest(10, "Review Count")[["Product","Company","Review Count","Product URL"]]
    t3_html   = hyperlink_products(top_rev_df)[["Product","Company","Review Count"]]
    t3.markdown(t3_html.to_html(escape=False, index=False), unsafe_allow_html=True)

    top_rat_df = (
        filtered_df[filtered_df["Review Count"] >= 50]
        .nlargest(10, "Rating")[["Product","Company","Rating","Review Count","Product URL"]]
    )
    t4_html = hyperlink_products(top_rat_df)[["Product","Company","Rating","Review Count"]]
    t4.markdown(t4_html.to_html(escape=False, index=False), unsafe_allow_html=True)

    top_dis_df = (
        filtered_df.nlargest(10, "ComputedDiscount")[["Product","Company","ComputedDiscount","Product URL"]]
        .rename(columns={"ComputedDiscount":"Discount (%)"})
    )
    t5_html = hyperlink_products(top_dis_df)[["Product","Company","Discount (%)"]]
    t5.markdown(t5_html.to_html(escape=False, index=False), unsafe_allow_html=True)


# ── PART 6: DATA EXPORT ──────────────────────────────────────────────────────
if st.session_state.data:
    st.markdown("---")

    # 7) FULL TABLE & DOWNLOAD HEADER
    st.markdown("📋 **Complete Product Data & Export:** Download Data")

    # Download controls (one column, below the header)
    buf = BytesIO()
    filtered_df.to_excel(buf, index=False, engine="openpyxl")

    fn = st.text_input("Export filename", "myntra_insights_v3_0.xlsx")
    if not fn.lower().endswith(".xlsx"):
        fn += ".xlsx"

    st.download_button("📥 Download Excel", buf.getvalue(), fn)

    # 8) RENDER CLICKABLE FULL TABLE
    html_df = hyperlink_products(filtered_df)[
        ["Product", "Company", "Review Count", "Rating", "ComputedDiscount", "Product URL"]
    ]
    st.markdown(html_df.to_html(escape=False, index=False), unsafe_allow_html=True)
