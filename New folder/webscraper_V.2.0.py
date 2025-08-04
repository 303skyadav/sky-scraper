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
st.set_page_config(page_title="SKY Scraper & Analyst V2.0", layout="wide")
st.title("🚀 SKY Scraper & Analyst V2.0")

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
        raw = a["href"]
        path = raw if raw.startswith("/") else f"/{raw}"
        url = f"https://www.myntra.com{path}"
        if url in st.session_state.seen:
            continue
        st.session_state.seen.add(url)

        wm    = item.select_one(".product-waterMark")
        ads   = "Yes" if (wm and wm.text.strip().upper()=="AD") else "No"
        name  = item.select_one(".product-product")
        brand = item.select_one(".product-brand")
        rc    = item.select_one(".product-ratingsCount")
        rat   = item.select_one(".product-ratingsContainer span")
        sp    = item.select_one(".product-discountedPrice") or item.select_one(".product-price")
        mrp   = item.select_one(".product-strike")
        disc  = item.select_one(".product-discountPercentage")

        batch.append({
            "Product": name.text.strip()      if name else "",
            "Company": brand.text.strip()     if brand else "",
            "Review Count": rc.text.strip().replace("|","") if rc else "",
            "Rating":      rat.text.strip()   if rat else "",
            "Ads":         ads,
            "Selling Price": sp.text.strip()  if sp else "",
            "MRP":         (mrp.text.strip()  if mrp else (sp.text.strip() if sp else "")),
            "Raw Discount": disc.text.strip() if disc else "0% OFF",
            "Product URL": url
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
    df["Review Count"] = (
        pd.to_numeric(df["Review Count"].str.replace(r"[^\d]", "", regex=True),
                      errors="coerce").fillna(0).astype(int)
    )
    df["Rating"]      = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    df["SellingVal"]  = pd.to_numeric(df["Selling Price"].str.replace(r"[^\d]", "", regex=True),
                                       errors="coerce").fillna(0).astype(int)
    df["MRPVal"]      = pd.to_numeric(df["MRP"].str.replace(r"[^\d]", "", regex=True),
                                       errors="coerce").fillna(0).astype(int)

    # computed discount (clamped 0–100%)
    df["ComputedDiscount"] = (
        (df["MRPVal"] - df["SellingVal"]).clip(lower=0)
        / df["MRPVal"].replace(0, np.nan)
    ).fillna(0).clip(0,1) * 100

    # — OVERALL METRICS ——————————————————————————————————————————————
    total_skus       = len(df)
    unique_companies = df["Company"].nunique()
    pct_ads          = df["Ads"].eq("Yes").mean() * 100
    avg_discount     = df["ComputedDiscount"].mean()

    st.subheader("📝 Executive Summary & Insights")
    st.markdown(f"""
    - **Total SKUs:** {total_skus}  
    - **Unique companies:** {unique_companies}  
    - **% Advertised:** {pct_ads:.1f}%  
    - **Average discount:** {avg_discount:.1f}%  
    """)

    # — TOP 10 COMPANIES ANALYSIS ——————————————————————————————————————
    top10 = df["Company"].value_counts().head(10)
    top10_skus   = int(top10.sum())
    top10_comp   = top10.index.tolist()
    top10_pctads = df[df["Company"].isin(top10_comp)]["Ads"].eq("Yes").mean()*100
    top10_avgrat = df[df["Company"].isin(top10_comp)]["Rating"].mean()
    top10_avgdis = df[df["Company"].isin(top10_comp)]["ComputedDiscount"].mean()

    st.subheader("📈 Analysis of Top 10 Companies")
    st.markdown(f"""
    - **Total SKUs:** {top10_skus}  
    - **Unique companies:** {len(top10_comp)}  
    - **% Advertised:** {top10_pctads:.1f}%  
    - **Average rating:** {top10_avgrat:.2f}  
    - **Average discount:** {top10_avgdis:.1f}%  
    """)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(top10.index, top10.values)
        ax.set_xticklabels(top10.index, rotation=45, ha="right")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)
    with c2:
        avg_disc_by = df[df["Company"].isin(top10_comp)].groupby("Company")["ComputedDiscount"].mean().loc[top10_comp]
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(avg_disc_by.index, avg_disc_by.values)
        ax.set_xticklabels(avg_disc_by.index, rotation=45, ha="right")
        ax.set_ylabel("Avg. Discount (%)")
        st.pyplot(fig, use_container_width=True)

    # — KUSHAL'S FASHION JEWELLERY ANALYSIS —————————————————————————————
    kushal = df[df["Company"].str.contains("Kushal", case=False)]
    kus_total = len(kushal)
    kus_avgrat = kushal["Rating"].mean()  if kus_total else 0
    kus_avgdis = kushal["ComputedDiscount"].mean() if kus_total else 0

    st.subheader("💎 Kushal's Fashion Jewellery Insights")
    st.markdown(f"""
    - **Total SKUs:** {kus_total}  
    - **Average rating:** {kus_avgrat:.2f}  
    - **Average discount:** {kus_avgdis:.1f}%  
    """)
    # top 10 Kushal products by review count
    top10_kushal = (kushal.sort_values("Review Count", ascending=False)
                        .head(10)
                        .loc[:, ["Product","Review Count","Rating","ComputedDiscount","Selling Price","MRP"]]
                        .rename(columns={"ComputedDiscount":"Discount (%)"}))
    st.markdown("**Top 10 Kushal Products by Review Count**")
    st.table(top10_kushal)

    # — ADVERTISED PRODUCTS ANALYSIS —————————————————————————————————————
    adv = df[df["Ads"]=="Yes"]
    adv_total  = len(adv)
    adv_unique = adv["Company"].nunique()
    adv_pct    = adv_total / total_skus * 100
    adv_avgrat = adv["Rating"].mean()
    adv_avgdis = adv["ComputedDiscount"].mean()

    st.subheader("🔍 Analysis of Advertised Products")
    st.markdown(f"""
    - **Total SKUs:** {adv_total}  
    - **Unique companies:** {adv_unique}  
    - **% Product Advertised:** {adv_pct:.1f}%  
    - **Average rating:** {adv_avgrat:.2f}  
    - **Average discount:** {adv_avgdis:.1f}%  
    """)

    c1, c2 = st.columns(2)
    with c1:
        adv_counts = adv["Company"].value_counts().head(10)
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

    st.markdown("---")
    st.dataframe(df.drop(columns=["SellingVal","MRPVal","ComputedDiscount"]), use_container_width=True)

    # — DOWNLOAD —————————————————————————————————————————————————————
    file_name = st.text_input("Name your download file", "myntra_jewelry.xlsx")
    if not file_name.lower().endswith(".xlsx"):
        file_name += ".xlsx"
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        "📥 Download Excel",
        data=buf.getvalue(),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
