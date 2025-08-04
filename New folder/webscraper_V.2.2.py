# ── webscraper_V2.2.py – Part 1 of 3 ──────────────────────────────────────────

import os
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress logs

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

# ── STREAMLIT SETUP ───────────────────────────────────────────────────────────
st.set_page_config(page_title="SKY Scraper & Analyst V2.2", layout="wide")
st.title("🚀 SKY Scraper & Analyst V2.2")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = []
if "seen" not in st.session_state:
    st.session_state.seen = set()
if "stop" not in st.session_state:
    st.session_state.stop = False

# ── SELENIUM DRIVER ───────────────────────────────────────────────────────────
@st.cache_resource
def get_driver():
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("detach", True)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

driver = get_driver()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Enter a Myntra category URL & pages to scrape.  
2. Click **Auto Scrape**.  
3. Filter by company, sub-category, discount, or ads.  
4. Pick a focused view & sub-category for deeper insights.
""")
disc_min, disc_max = st.sidebar.slider("Discount Range (%)", 0, 100, (0, 100))
ads_filter         = st.sidebar.selectbox("Ads Filter", ["All", "Only Ads", "Only Non-Ads"])

# ── MAIN UI CONTROLS ──────────────────────────────────────────────────────────
_, col_url, col_pages, _ = st.columns([1,4,2,1])
url_base        = col_url.text_input(
    "Base URL",
    "https://www.myntra.com/women-jewellery"
)
pages_to_scrape = col_pages.number_input(
    "Pages to Scrape",
    min_value=1,    # no lower than 1
    value=10,       # default when loaded
    step=1,         # + / – increments
    help="Enter 0 to run until no more items appear"
)
_, col_auto, col_stop, _ = st.columns([1,2,2,1])
btn_auto = col_auto.button("🤖 Auto Scrape")
btn_stop = col_stop.button("🛑 Stop")
status   = st.empty()

# ── SCRAPER FUNCTION ─────────────────────────────────────────────────────────
def scrape_html(html, page_no):
    soup  = BeautifulSoup(html, "html.parser")
    batch = []
    for placement_idx, item in enumerate(soup.select("li.product-base"), start=1):
        a = item.select_one("a")
        if not a or not a.has_attr("href"):
            continue
        href = a["href"].split("?")[0]
        url  = "https://www.myntra.com" + (href if href.startswith("/") else "/" + href)
        if url in st.session_state.seen:
            continue
        st.session_state.seen.add(url)

        subcat = urlparse(url).path.strip("/").split("/")[0]
        txt    = lambda sel: sel.text.strip() if sel else ""
        wm     = item.select_one(".product-waterMark")

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

# ── AUTO SCRAPE LOOP ─────────────────────────────────────────────────────────
if btn_auto:
    st.session_state.stop = False
    st.session_state.data.clear()
    st.session_state.seen.clear()
    sep = "&p=" if "?" in url_base else "?p="
    with st.spinner("Scraping pages..."):
        for p in range(1, pages_to_scrape + 1):
            if st.session_state.stop:
                status.warning(f"⏸️ Stopped at page {p-1}")
                break

            driver.get(f"{url_base}{sep}{p}")
            time.sleep(2)

            # Pass p into scrape_html so you record the page number
            new = scrape_html(driver.page_source, p)
            if not new:
                status.info(f"No items on page {p}, ending.")
                break

            st.session_state.data.extend(new)
            status.info(f"Page {p}: +{len(new)} items (total {len(st.session_state.data)})")

        else:
            status.success("✅ Auto-scrape complete!")

if btn_stop:
    st.session_state.stop = True


# ── DATA CLEANING & DERIVED FIELDS ───────────────────────────────────────────
if st.session_state.data:
    # ── DATA CLEANING & DERIVED FIELDS ─────────────────────────────────────────
    df = pd.DataFrame(st.session_state.data)
    df["Review Count"]     = pd.to_numeric(
        df["Review Count"].str.replace(r"[^\d]", "", regex=True),
        errors="coerce"
    ).fillna(0).astype(int)
    df["Rating"]           = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    df["SellingVal"]       = pd.to_numeric(
        df["Selling Price"].str.replace(r"[^\d]", "", regex=True),
        errors="coerce"
    ).fillna(0).astype(int)
    df["MRPVal"]           = pd.to_numeric(
        df["MRP"].str.replace(r"[^\d]", "", regex=True),
        errors="coerce"
    ).fillna(0).astype(int)
    df["ComputedDiscount"] = (
        (df["MRPVal"] - df["SellingVal"]).clip(lower=0) /
        df["MRPVal"].replace(0, np.nan)
    ).fillna(0).clip(0,1) * 100

    # ── APPLY SIDEBAR FILTERS ─────────────────────────────────────────────────
    companies = sorted(df["Company"].unique())
    company_filter = st.sidebar.multiselect("Filter by Company", options=companies)
    if company_filter:
        df = df[df["Company"].isin(company_filter)]
    subcats = sorted(df["Sub-category"].unique())
    subcat_filter = st.sidebar.multiselect("Filter by Sub-category", options=subcats)
    if subcat_filter:
        df = df[df["Sub-category"].isin(subcat_filter)]
    df = df[(df["ComputedDiscount"] >= disc_min) & (df["ComputedDiscount"] <= disc_max)]
    if ads_filter == "Only Ads":
        df = df[df["Ads"] == "Yes"]
    elif ads_filter == "Only Non-Ads":
        df = df[df["Ads"] == "No"]

    # ── EXECUTIVE SUMMARY & DUAL DROPDOWNS ─────────────────────────────────────
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

    # Build temp_df based on the selected view
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

    # Sub-category filter dropdown
    subcats2 = ["All"] + sorted(temp_df["Sub-category"].unique())
    subcat_option = c2.selectbox("Select sub-category", subcats2, key="subcat")

    # Apply sub-category filter
    if subcat_option != "All":
        filtered_df = temp_df[temp_df["Sub-category"] == subcat_option]
    else:
        filtered_df = temp_df.copy()

  # ── SUMMARY METRICS ─────────────────────────────────────────────────────────
    # Overall metrics
    total_skus             = len(filtered_df)
    no_of_products         = filtered_df["Product"].nunique()
    pct_ads                = filtered_df["Ads"].eq("Yes").mean() * 100
    no_companies_using_ads = (
        filtered_df.loc[filtered_df["Ads"] == "Yes", "Company"]
        .nunique()
    )
    avg_discount           = filtered_df["ComputedDiscount"].mean()
    top_product_review     = filtered_df.nlargest(1, "Review Count")["Product"].iloc[0]
    top_product_rating     = filtered_df.nlargest(1, "Rating")["Product"].iloc[0]
    top_cat_review         = filtered_df.groupby("Sub-category")["Review Count"] \
                                .sum().idxmax()
    top_cat_rating         = filtered_df.groupby("Sub-category")["Rating"] \
                                .mean().idxmax()

    # Kushal-specific metrics
    kushal_df               = filtered_df[
        filtered_df["Company"].str.contains("Kushal", case=False)
    ]
    k_total_skus            = len(kushal_df)
    k_no_of_products        = kushal_df["Product"].nunique()
    k_pct_ads               = kushal_df["Ads"].eq("Yes").mean() * 100 if k_total_skus else 0
    k_no_products_with_ads  = kushal_df["Ads"].eq("Yes").sum()
    k_avg_discount          = kushal_df["ComputedDiscount"].mean() if k_total_skus else 0
    k_top_product_review    = kushal_df.nlargest(1, "Review Count")["Product"].iloc[0] \
                                if k_total_skus else "N/A"
    k_top_product_rating    = kushal_df.nlargest(1, "Rating")["Product"].iloc[0] \
                                if k_total_skus else "N/A"
    k_top_cat_review        = kushal_df.groupby("Sub-category")["Review Count"] \
                                .sum().idxmax() if k_total_skus else "N/A"
    k_top_cat_rating        = kushal_df.groupby("Sub-category")["Rating"] \
                                .mean().idxmax() if k_total_skus else "N/A"

        # ── RENDER SIDE-BY-SIDE & CENTERED ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
<div style="text-align:center;">
  <h4>Overall Metrics</h4>
  <ul style="display:inline-block; text-align:left; padding-left:20px;">
    <li><strong>Total SKUs:</strong> {total_skus}</li>
    <li><strong>No of Products:</strong> {no_of_products}</li>
    <li><strong>% Advertised:</strong> {pct_ads:.1f}%</li>
    <li><strong>No of Companies using Advertisement:</strong> {no_companies_using_ads}</li>
    <li><strong>Average discount:</strong> {avg_discount:.1f}%</li>
    <li><strong>Top Product (by reviews):</strong> {top_product_review}</li>
    <li><strong>Top Product (by rating):</strong> {top_product_rating}</li>
    <li><strong>Top Category (by reviews):</strong> {top_cat_review}</li>
    <li><strong>Top Category (by rating):</strong> {top_cat_rating}</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
<div style="text-align:center;">
  <h4>Kushal Metrics</h4>
  <ul style="display:inline-block; text-align:left; padding-left:20px;">
    <li><strong>Total SKUs:</strong> {k_total_skus}</li>
    <li><strong>No of Products:</strong> {k_no_of_products}</li>
    <li><strong>% Advertised:</strong> {k_pct_ads:.1f}%</li>
    <li><strong>Kushal’s no of Products with ads:</strong> {k_no_products_with_ads}</li>
    <li><strong>Average discount:</strong> {k_avg_discount:.1f}%</li>
    <li><strong>Top Product (by reviews):</strong> {k_top_product_review}</li>
    <li><strong>Top Product (by rating):</strong> {k_top_product_rating}</li>
    <li><strong>Top Category (by reviews):</strong> {k_top_cat_review}</li>
    <li><strong>Top Category (by rating):</strong> {k_top_cat_rating}</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── ROW 1: TWO CHARTS ───────────────────────────────────────────────────────
    st.subheader("📊 Category & Branding Overview")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**SKU Count by Sub-category**")
        sub_counts = filtered_df["Sub-category"].value_counts()
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(sub_counts.index, sub_counts.values)
        ax.set_xticklabels(sub_counts.index, rotation=45, ha="right")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)
    with r1c2:
        st.markdown("**Top 10 Companies by SKU Count**")
        top10 = filtered_df["Company"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(top10.index, top10.values)
        ax.set_xticklabels(top10.index, rotation=45, ha="right")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)

# ── End of Part 1 ─────────────────────────────────────────────────────────್ಲ

    # ── PART 2: Advertised, Kushal & Champions/Laggards ───────────────────────────

    # Advertised Products Analysis
    adv = filtered_df[filtered_df["Ads"] == "Yes"]
    if not adv.empty:
        st.subheader("🔍 Advertised Products Analysis")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Top 10 Advertisers**")
            adv_counts = adv["Company"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(adv_counts.index, adv_counts.values)
            ax.set_xticklabels(adv_counts.index, rotation=45, ha="right")
            ax.set_ylabel("SKU Count")
            st.pyplot(fig, use_container_width=True)
        with a2:
            st.markdown("**Discount vs Rating (Ads)**")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.scatter(adv["ComputedDiscount"], adv["Rating"], alpha=0.6)
            ax.set_xlabel("Discount (%)"); ax.set_ylabel("Rating")
            st.pyplot(fig, use_container_width=True)

    # ── Combined Bubble & Brand Charts (single row) ──────────────────────────
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("💡 Review vs Rating vs Discount (Bubble Chart)")
        fig, ax = plt.subplots(figsize=(6,3))
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
            .agg(AvgRating=("Rating","mean"), AvgDiscount=("ComputedDiscount","mean"), SKUCount=("Product","count"))
            .sort_values("SKUCount", ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(6,3))
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



    # Champions & Laggards
    filtered_df["Score"] = filtered_df["Review Count"] * filtered_df["Rating"]
    ch, lg = st.columns(2)
    with ch:
        st.subheader("🏆 Top 10 Champions")
        champions = filtered_df.nlargest(10, "Score")[["Product", "Company", "Review Count", "Rating"]]
        st.table(champions)
    with lg:
        st.subheader("⚠️ Top 10 Laggards")
        laggards = filtered_df.nsmallest(10, "Rating")[["Product", "Company", "ComputedDiscount", "Rating"]]
        st.table(laggards)


# Part 3


    # ── Part 3 of webscraper_V2.2.py ────────────────────────────────────────────
    # (Paste this directly after Part 2, still inside `if st.session_state.data:`)

    # ── Top Reviewed / Rated / Discounted Products ─────────────────────────────
    st.subheader("🔝 Top Reviewed, Rated & Discounted Products")
    t3, t4, t5 = st.columns(3)
    with t3:
        st.markdown("**Top 10 Reviewed**")
        top_rev = filtered_df.nlargest(10, "Review Count")[["Product","Company","Review Count"]]
        st.table(top_rev)
    with t4:
        st.markdown("**Top 10 Rated (≥50 reviews)**")
        top_rat = (
            filtered_df[filtered_df["Review Count"] >= 50]
            .nlargest(10, "Rating")[["Product","Company","Rating","Review Count"]]
        )
        st.table(top_rat)
    with t5:
        st.markdown("**Top 10 Discounted**")
        top_dis = (
            filtered_df.nlargest(10, "ComputedDiscount")
            [["Product","Company","ComputedDiscount"]]
            .rename(columns={"ComputedDiscount":"Discount (%)"})
        )
        st.table(top_dis)

    # ── Full Table & Download ──────────────────────────────────────────────────
    st.markdown("---")
    st.dataframe(
        filtered_df.drop(columns=["SellingVal","MRPVal","Raw Discount","ComputedDiscount","Score"]),
        use_container_width=True
    )
    fn = st.text_input("Download filename", "myntra_insights_v2_2.xlsx")
    if not fn.lower().endswith(".xlsx"):
        fn += ".xlsx"
    buf = BytesIO()
    filtered_df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button("📥 Download Excel", buf.getvalue(), fn)

