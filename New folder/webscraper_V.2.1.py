import os
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow/Selenium logs

import streamlit as st
import time
import pandas as pd
import numpy as np
from io import BytesIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt

# — STREAMLIT SETUP —————————————————————————————————————————————————
st.set_page_config(page_title="SKY Scraper & Analyst V2.1", layout="wide")
st.title("🚀 SKY Scraper & Analyst V2.1")

# — SESSION STATE —————————————————————————————————————————————————
if "data" not in st.session_state:
    st.session_state.data = []
if "seen" not in st.session_state:
    st.session_state.seen = set()
if "stop" not in st.session_state:
    st.session_state.stop = False

# — DRIVER FACTORY ————————————————————————————————————————————————
@st.cache_resource
def get_driver():
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("detach", True)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

driver = get_driver()

# — UI CONTROLS —————————————————————————————————————————————————
_, col_url, col_pages, _ = st.columns([1,4,2,1])
url_base        = col_url.text_input("Base URL", "https://www.myntra.com/women-jewellery")
pages_to_scrape = col_pages.number_input("Pages (Auto)", min_value=1, value=10)
_, col_auto, col_stop, _ = st.columns([1,2,2,1])
btn_auto = col_auto.button("🤖 Auto Scrape")
btn_stop = col_stop.button("🛑 Stop")
status = st.empty()

# — SCRAPE FUNCTION ——————————————————————————————————————————————
def scrape_html(html):
    soup = BeautifulSoup(html, "html.parser")
    batch = []
    for item in soup.select("li.product-base"):
        a = item.select_one("a")
        if not a or not a.has_attr("href"):
            continue
        href = a["href"]
        url  = "https://www.myntra.com" + (href if href.startswith("/") else "/" + href)
        if url in st.session_state.seen:
            continue
        st.session_state.seen.add(url)
        def txt(sel): return sel.text.strip() if sel else ""
        wm    = item.select_one(".product-waterMark")
        batch.append({
            "Product":       txt(item.select_one(".product-product")),
            "Company":       txt(item.select_one(".product-brand")),
            "Review Count":  txt(item.select_one(".product-ratingsCount")).replace("|",""),
            "Rating":        txt(item.select_one(".product-ratingsContainer span")),
            "Ads":           "Yes" if wm and wm.text.strip().upper()=="AD" else "No",
            "Selling Price": txt(item.select_one(".product-discountedPrice") or item.select_one(".product-price")),
            "MRP":           txt(item.select_one(".product-strike")) or txt(item.select_one(".product-discountedPrice") or item.select_one(".product-price")),
            "Raw Discount":  txt(item.select_one(".product-discountPercentage")) or "0% OFF",
            "Product URL":   url
        })
    return batch

# — AUTO SCRAPE ———————————————————————————————————————————————————
if btn_auto:
    st.session_state.stop = False
    st.session_state.data.clear()
    st.session_state.seen.clear()
    sep = "&p=" if "?" in url_base else "?p="
    for p in range(1, pages_to_scrape+1):
        if st.session_state.stop:
            status.warning(f"⏸️ Stopped at page {p-1}")
            break
        driver.get(f"{url_base}{sep}{p}")
        time.sleep(2)
        new = scrape_html(driver.page_source)
        if not new:
            status.info(f"No items on page {p}, ending.")
            break
        st.session_state.data.extend(new)
        status.info(f"Page {p}: +{len(new)} items (total {len(st.session_state.data)})")
    else:
        status.success("✅ Auto-scrape complete!")

if btn_stop:
    st.session_state.stop = True

# — DATA CLEANING & DERIVED FIELDS —————————————————————————————————————
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)

    # numeric conversions
    df["Review Count"]     = pd.to_numeric(df["Review Count"].str.replace(r"[^\d]", "", regex=True), errors="coerce").fillna(0).astype(int)
    df["Rating"]           = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    df["SellingVal"]       = pd.to_numeric(df["Selling Price"].str.replace(r"[^\d]", "", regex=True), errors="coerce").fillna(0).astype(int)
    df["MRPVal"]           = pd.to_numeric(df["MRP"].str.replace(r"[^\d]", "", regex=True), errors="coerce").fillna(0).astype(int)
    df["ComputedDiscount"] = ((df["MRPVal"] - df["SellingVal"]).clip(lower=0) / df["MRPVal"].replace(0, np.nan)).fillna(0).clip(0,1) * 100

    # — OVERALL METRICS & VIEW SELECTION ——————————————————————————————
    st.subheader("📝 Executive Summary & Insights")

    view_option = st.selectbox(
        "Select view for focused analysis:",
        [
            "All Data",
            "Top Advertisers",
            "Top Reviewed Products",
            "Top Rated Products",
            "Highest Discounted Products",
        ]
    )

    if view_option == "Top Advertisers":
        filtered_df = df[df["Ads"] == "Yes"]
    elif view_option == "Top Reviewed Products":
        filtered_df = df.nlargest(100, "Review Count")
    elif view_option == "Top Rated Products":
        filtered_df = df.nlargest(100, "Rating")
    elif view_option == "Highest Discounted Products":
        filtered_df = df.nlargest(100, "ComputedDiscount")
    else:
        filtered_df = df.copy()

    total_skus       = len(filtered_df)
    unique_companies = filtered_df["Company"].nunique()
    pct_ads          = filtered_df["Ads"].eq("Yes").mean() * 100
    avg_discount     = filtered_df["ComputedDiscount"].mean()

    st.markdown(f"""
    - **Total SKUs:** {total_skus}  
    - **Unique companies:** {unique_companies}  
    - **% Advertised:** {pct_ads:.1f}%  
    - **Average discount:** {avg_discount:.1f}%  
    """)

    # — TOP 10 COMPANIES ANALYSIS ——————————————————————————————————————
    top10 = filtered_df["Company"].value_counts().head(10)
    top10_comp   = top10.index.tolist()
    st.subheader("📈 Analysis of Top 10 Companies")
    st.markdown(f"""
    - **Top 10 SKUs count sum:** {int(top10.sum())}  
    - **Unique companies in top10:** {len(top10_comp)}  
    """)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(top10.index, top10.values)
        ax.set_xticklabels(top10.index, rotation=45, ha="right")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)
    with c2:
        avg_disc_by = filtered_df[filtered_df["Company"].isin(top10_comp)].groupby("Company")["ComputedDiscount"].mean().loc[top10_comp]
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(avg_disc_by.index, avg_disc_by.values)
        ax.set_xticklabels(avg_disc_by.index, rotation=45, ha="right")
        ax.set_ylabel("Avg. Discount (%)")
        st.pyplot(fig, use_container_width=True)

    # — KUSHAL'S FASHION JEWELLERY INSIGHTS —————————————————————————————
    kushal = filtered_df[filtered_df["Company"].str.contains("Kushal", case=False)]
    if not kushal.empty:
        st.subheader("💎 Kushal's Fashion Jewellery Insights")
        st.markdown(f"""
        - **Total SKUs:** {len(kushal)}  
        - **Average rating:** {kushal["Rating"].mean():.2f}  
        - **Average discount:** {kushal["ComputedDiscount"].mean():.1f}%  
        """)
        top10_kushal = (kushal.nlargest(10, "Review Count")
                             .loc[:, ["Product","Review Count","Rating","ComputedDiscount","Selling Price","MRP"]]
                             .rename(columns={"ComputedDiscount":"Discount (%)"}))
        st.markdown("**Top 10 Kushal Products by Review Count**")
        st.table(top10_kushal)

    # — ADVERTISED PRODUCTS ANALYSIS —————————————————————————————————————
    adv = filtered_df[filtered_df["Ads"]=="Yes"]
    if not adv.empty:
        st.subheader("🔍 Analysis of Advertised Products")
        adv_counts = adv["Company"].value_counts().head(10)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(adv_counts.index, adv_counts.values)
            ax.set_xticklabels(adv_counts.index, rotation=45, ha="right")
            ax.set_ylabel("SKU Count")
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.scatter(adv["ComputedDiscount"], adv["Rating"])
            ax.set_xlabel("Discount (%)")
            ax.set_ylabel("Rating")
            st.pyplot(fig, use_container_width=True)

    # — TOP 10 CHAMPIONS & LAGGARDS ——————————————————————————————————————
    filtered_df["Score"] = filtered_df["Review Count"] * filtered_df["Rating"]
    champs  = filtered_df.nlargest(10, "Score")[["Product","Company","Review Count","Rating"]]
    laggs   = filtered_df.nsmallest(10, "Rating")[["Product","Company","ComputedDiscount","Rating"]]
    st.subheader("🏆 Top 10 Champions & ⚠️ Top 10 Laggards")
    c1, c2 = st.columns(2)
    with c1:
        st.table(champs)
    with c2:
        st.table(laggs)

    st.markdown("---")
    st.dataframe(
        filtered_df.drop(columns=["SellingVal","MRPVal","ComputedDiscount","Score"]),
        use_container_width=True
    )

    # — DOWNLOAD —————————————————————————————————————————————————————
    file_name = st.text_input("Name your download file", "myntra_jewelry.xlsx")
    if not file_name.lower().endswith(".xlsx"):
        file_name += ".xlsx"
    buf = BytesIO()
    filtered_df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        "📥 Download Excel",
        data=buf.getvalue(),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
