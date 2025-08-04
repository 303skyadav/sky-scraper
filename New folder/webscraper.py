import os
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow and Selenium noise

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
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
# from automations import create  # uncomment if you configure scheduled alerts

# ── STREAMLIT LAYOUT ────────────────────────────────────────────────────────────
st.set_page_config(page_title="SKY Scraper & Analyst V.3.0", layout="wide")
st.title("🚀 SKY Scraper & Analyst V2.0")

# ── SESSION STATE INIT ──────────────────────────────────────────────────────────
if "data" not in st.session_state:    st.session_state.data = []
if "seen" not in st.session_state:    st.session_state.seen = set()
if "stop" not in st.session_state:    st.session_state.stop = False

# ── SIDEBAR “HOW TO” & FILTER PLACEHOLDERS ──────────────────────────────────────
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Enter a Myntra category URL and pages to scrape.  
2. Click **Auto Scrape**.  
3. **After scraping**, filter your data by:  
   - Company  
   - Discount range  
   - Ads only / Non-Ads  
4. Explore analytics & export your data.
""")
# placeholders—will be overwritten once df exists
company_filter = st.sidebar.multiselect("Filter by Company", options=[])
disc_min, disc_max = st.sidebar.slider("Discount Range (%)", 0, 100, (0,100))
ads_filter = st.sidebar.selectbox("Ads Filter", ["All","Only Ads","Only Non-Ads"])

# ── SCRAPER CONTROLS ────────────────────────────────────────────────────────────
_, col_url, col_pages, _ = st.columns([1,4,2,1])
url_base        = col_url.text_input("Myntra Category URL",
                                     "https://www.myntra.com/women-jewellery")
pages_to_scrape = col_pages.number_input("Pages to Scrape", 1, 50, 10)
_, col_start, col_stop, _ = st.columns([1,2,2,1])
btn_auto = col_start.button("🤖 Auto Scrape")
btn_stop = col_stop.button("🛑 Stop")
status   = st.empty()

# ── DRIVER FACTORY ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_driver():
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("detach", True)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

driver = get_driver()

# ── SCRAPE HELPER ───────────────────────────────────────────────────────────────
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

        wm    = item.select_one(".product-waterMark")
        ads   = "Yes" if wm and wm.text.strip().upper()=="AD" else "No"
        name  = item.select_one(".product-product")
        brand = item.select_one(".product-brand")
        rc    = item.select_one(".product-ratingsCount")
        rat   = item.select_one(".product-ratingsContainer span")
        sp    = item.select_one(".product-discountedPrice") or item.select_one(".product-price")
        mrp   = item.select_one(".product-strike")
        disc  = item.select_one(".product-discountPercentage")

        batch.append({
            "Product":        name.text.strip()      if name  else "",
            "Company":        brand.text.strip()     if brand else "",
            "Review Count":   rc.text.strip().replace("|","") if rc else "",
            "Rating":         rat.text.strip()       if rat   else "",
            "Ads":            ads,
            "Selling Price":  sp.text.strip()        if sp    else "",
            "MRP":            mrp.text.strip()       if mrp   else (sp.text.strip() if sp else ""),
            "Raw Discount":   disc.text.strip()      if disc  else "0% OFF",
            "Product URL":    url
        })
    return batch

# ── AUTO-SCRAPE LOOP ───────────────────────────────────────────────────────────
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

# ── DATA CLEANING & DERIVED FIELDS ─────────────────────────────────────────────
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)

    # numeric conversions
    df["Review Count"] = pd.to_numeric(
        df["Review Count"].str.replace(r"[^\d]", "", regex=True),
        errors="coerce"
    ).fillna(0).astype(int)
    df["Rating"]       = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    df["PriceVal"]     = pd.to_numeric(
        df["Selling Price"].str.replace(r"[^\d]", "", regex=True),
        errors="coerce"
    ).fillna(0).astype(int)
    df["MRPVal"]       = pd.to_numeric(
        df["MRP"].str.replace(r"[^\d]", "", regex=True),
        errors="coerce"
    ).fillna(df["PriceVal"]).astype(int)
    df["ComputedDiscount"] = (
        (df["MRPVal"] - df["PriceVal"]).clip(lower=0)
        / df["MRPVal"].replace(0, np.nan)
    ).fillna(0).clip(0,1) * 100

    # ── DYNAMIC FILTER OPTIONS ─────────────────────────────────────────────────
    companies = sorted(df["Company"].unique())
    company_filter = st.sidebar.multiselect("Filter by Company", options=companies)

    # apply filters
    if company_filter:
        df = df[df["Company"].isin(company_filter)]
    df = df[(df["ComputedDiscount"] >= disc_min) & (df["ComputedDiscount"] <= disc_max)]
    if ads_filter == "Only Ads":
        df = df[df["Ads"] == "Yes"]
    elif ads_filter == "Only Non-Ads":
        df = df[df["Ads"] == "No"]

    # ── EXECUTIVE SUMMARY ────────────────────────────────────────────────────────
    total_skus       = len(df)
    unique_companies = df["Company"].nunique()
    pct_ads          = df["Ads"].eq("Yes").mean() * 100
    avg_discount     = df["ComputedDiscount"].mean()

    st.subheader("💡 Executive Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total SKUs",       total_skus)
    k2.metric("Unique Companies", unique_companies)
    k3.metric("% Advertised",     f"{pct_ads:.1f}%")
    k4.metric("Avg Discount",     f"{avg_discount:.1f}%")

    st.markdown("---")
    # ── PARSE & SHOW SUB-CATEGORY ──────────────────────────────────────────────
    subcat = url_base.rstrip("/").split("/")[-1]
    st.caption(f"**Category Analyzed:** `{subcat}`")

    # ── TWO-UP ROW 1: Time-Series & Top-10 Interactive ─────────────────────────
    # record today’s metrics to CSV history
    hist_file = "metrics_history.csv"
    today = datetime.date.today().isoformat()
    new_row = pd.DataFrame([{
        "date": today,
        "total_skus": total_skus,
        "avg_discount": avg_discount,
        "pct_ads": pct_ads
    }])
    if os.path.exists(hist_file):
        hist = pd.read_csv(hist_file)
        if today not in hist["date"].astype(str).values:
            pd.concat([hist, new_row], ignore_index=True).to_csv(hist_file, index=False)
    else:
        new_row.to_csv(hist_file, index=False)
    hist = pd.read_csv(hist_file, parse_dates=["date"])

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(hist["date"], hist["total_skus"], marker="o")
        ax.set_title("SKU Count Over Time")
        ax.set_ylabel("Total SKUs")
        ax.set_xlabel("Date")
        st.pyplot(fig, use_container_width=True)
    with c2:
        top10_df = df["Company"].value_counts().head(10).reset_index()
        top10_df.columns = ["Company","Count"]
        fig2 = px.bar(top10_df, x="Company", y="Count",
                      title="Top 10 Companies (Interactive)",
                      labels={"Count":"SKU Count"})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    # ── TWO-UP ROW 2: Correlation & Pricing Recommendations ────────────────────
    corr = df[["Review Count","Rating","ComputedDiscount","PriceVal"]].corr()
    pct_25 = df.groupby("Company")["ComputedDiscount"].quantile(0.25)
    pct_75 = df.groupby("Company")["ComputedDiscount"].quantile(0.75)
    # For true segment-level recs, swap groupby("Company") for groupby("Segment") below

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3))
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="left")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        ax.set_title("Feature Correlations")
        st.pyplot(fig, use_container_width=True)
    with c2:
        recs = pd.DataFrame({"25th %": pct_25, "75th %": pct_75})
        recs["Suggested Range"] = recs.apply(lambda r: f"{int(r['25th %'])}–{int(r['75th %'])}%", axis=1)
        st.markdown("**Dynamic Discount Recommendations**")
        st.table(recs)

    st.markdown("---")
    # ── TWO-UP ROW 3: Segmentation Bar & Boxplot ───────────────────────────────
    segments = KMeans(n_clusters=3, random_state=0).fit_predict(
        df[["Rating","ComputedDiscount","Review Count"]]
    )
    df["Segment"] = segments.astype(str)
    seg_counts = df["Segment"].value_counts().sort_index()

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(seg_counts.index, seg_counts.values, color="tab:blue")
        ax.set_title("Customer Segments (K-Means)")
        ax.set_xlabel("Segment")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)
    with c2:
        data_to_plot = [df[df["Segment"]==s]["PriceVal"] for s in seg_counts.index]
        fig, ax = plt.subplots(figsize=(6,3))
        ax.boxplot(data_to_plot, labels=seg_counts.index)
        ax.set_title("Price Distribution by Segment")
        ax.set_ylabel("Selling Price")
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    # ── TWO-UP ROW 4: Champions & Laggards ────────────────────────────────────
    df["Score"] = df["Review Count"] * df["Rating"]
    champions = df.nlargest(3, "Score")[["Product","Company","Review Count","Rating"]]
    laggards  = df.sort_values(["ComputedDiscount","Rating"], ascending=[False,True]) \
                  .head(3)[["Product","Company","ComputedDiscount","Rating"]]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏆 Top 3 Champions")
        st.table(champions)
    with c2:
        st.subheader("⚠️ Top 3 Laggards")
        st.table(laggards)

    st.markdown("---")
    # ── KUSHAL’S INSIGHTS ──────────────────────────────────────────────────────
    kushal = df[df["Company"].str.contains("Kushal", case=False)]
    if not kushal.empty:
        kus_total  = len(kushal)
        kus_avgrat = kushal["Rating"].mean()
        kus_avgdis = kushal["ComputedDiscount"].mean()
        st.subheader("💎 Kushal’s Fashion Jewellery Insights")
        st.markdown(f"- **Total SKUs:** {kus_total}  \n"
                    f"- **Avg Rating:** {kus_avgrat:.2f}  \n"
                    f"- **Avg Discount:** {kus_avgdis:.1f}%")
        top_kushal = kushal.sort_values("Review Count", ascending=False).head(10)
        top_kushal = top_kushal.rename(columns={"ComputedDiscount":"Discount (%)"})
        st.table(top_kushal[["Product","Review Count","Rating","Discount (%)","Selling Price","MRP"]])

    st.markdown("---")
    # ── TWO-UP ROW 5: Advertised Products ──────────────────────────────────────
    adv = df[df["Ads"] == "Yes"]
    adv_counts = adv["Company"].value_counts().head(10)
    adv_total  = len(adv)
    adv_unique = adv["Company"].nunique()
    adv_pct    = adv_total / total_skus * 100
    adv_avgrat = adv["Rating"].mean()
    adv_avgdis = adv["ComputedDiscount"].mean()

    st.subheader("🔍 Advertised Products Analysis")
    st.markdown(f"- **Ads SKUs:** {adv_total}  \n"
                f"- **Unique Companies:** {adv_unique}  \n"
                f"- **% of All SKUs:** {adv_pct:.1f}%  \n"
                f"- **Avg Rating:** {adv_avgrat:.2f}  \n"
                f"- **Avg Discount:** {adv_avgdis:.1f}%")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(adv_counts.index, adv_counts.values, color="tab:orange")
        ax.set_xticklabels(adv_counts.index, rotation=45, ha="right")
        ax.set_title("Top 10 Advertisers")
        ax.set_ylabel("SKU Count")
        st.pyplot(fig, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.scatter(adv["ComputedDiscount"], adv["Rating"], alpha=0.6)
        ax.set_xlabel("Discount (%)")
        ax.set_ylabel("Rating")
        ax.set_title("Discount vs Rating (Ads)")
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    # ── FULL TABLE & DOWNLOAD ───────────────────────────────────────────────────
    st.dataframe(df.drop(columns=["PriceVal","MRPVal","Raw Discount","ComputedDiscount","Score"]), use_container_width=True)
    file_name = st.text_input("Download filename", "myntra_insights.xlsx")
    if not file_name.lower().endswith(".xlsx"):
        file_name += ".xlsx"
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button("📥 Download Excel", buf.getvalue(), file_name,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── STUBS FOR FUTURE ENRICHMENT ───────────────────────────────────────────────
    # - Multi-variate regression example:
    #     # X = df[["ComputedDiscount","Review Count","PriceVal"]]; y = df["Rating"]
    #     # mv_model = LinearRegression().fit(X, y)
    # - External signal enrichment placeholder:
    #     # def fetch_social_sentiment(product_url): ...
    #     # df["Sentiment"] = df["Product URL"].apply(fetch_social_sentiment)
    # - Category breakout stub:
    #     # subcats = ["necklaces","earrings","rings"]
    #     # for sc in subcats: scrape_url = url_base + f"?p=1&sub={sc}"; ...
    # - Automations stub for daily summary:
    #     # create({
    #     #   "title": "Daily Myntra Summary",
    #     #   "prompt": "Generate today's Myntra scraping KPIs",
    #     #   "schedule": "BEGIN:VEVENT\nRRULE:FREQ=DAILY;BYHOUR=8;...\nEND:VEVENT"
    #     # })
