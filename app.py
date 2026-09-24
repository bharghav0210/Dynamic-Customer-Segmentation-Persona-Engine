from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from common import ask_llama, read_table, dataframe_to_excel_bytes, ollama_available

st.set_page_config(page_title="Dynamic Customer Segmentation & Persona Engine", page_icon="👥", layout="wide")
st.title("👥 Dynamic Customer Segmentation & Persona Engine")
st.caption("Upload customer data, choose features, select or auto-discover cluster count, test individual customers, and generate business personas.")

@st.cache_data
def demo_customers(n=600, seed=42):
    rng=np.random.default_rng(seed)
    grp=rng.choice([0,1,2,3],n,p=[.22,.34,.26,.18])
    rec=np.array([rng.normal([18,70,170,320][g],[12,25,45,70][g]) for g in grp]).clip(1,500)
    freq=np.array([rng.normal([24,12,6,2][g],[5,3,2,1][g]) for g in grp]).clip(1,50)
    monetary=np.array([rng.normal([4200,1900,850,260][g],[800,450,260,100][g]) for g in grp]).clip(50,8000)
    age=np.array([rng.normal([42,36,31,48][g],[8,9,7,10][g]) for g in grp]).clip(18,75)
    income=np.array([rng.normal([110000,75000,52000,40000][g],[18000,14000,12000,10000][g]) for g in grp]).clip(18000,180000)
    aov=monetary/freq
    return pd.DataFrame({"customer_id":[f"C{i+1:04d}" for i in range(n)],"recency_days":rec.round(0),"frequency_12m":freq.round(0),"monetary_12m":monetary.round(2),"age":age.round(0),"annual_income":income.round(0),"avg_order_value":aov.round(2)})

def persona_from_profile(profile, overall):
    labels=[]
    for c in profile.index:
        if not np.issubdtype(type(profile[c]), np.number):
            continue
        base=overall[c]
        if base==0 or pd.isna(base): continue
        ratio=(profile[c]-base)/(abs(base)+1e-9)
        lc=c.lower()
        if "recency" in lc:
            if ratio < -.25: labels.append("recently active")
            elif ratio > .25: labels.append("inactive/at-risk")
        elif "frequency" in lc:
            if ratio > .25: labels.append("frequent")
            elif ratio < -.25: labels.append("low-frequency")
        elif "monetary" in lc or "revenue" in lc or "spend" in lc or "order_value" in lc:
            if ratio > .25: labels.append("high-value")
            elif ratio < -.25: labels.append("low-spend")
        elif "income" in lc:
            if ratio > .25: labels.append("higher-income")
            elif ratio < -.25: labels.append("lower-income")
        elif "age" == lc:
            if ratio > .15: labels.append("older")
            elif ratio < -.15: labels.append("younger")
    if not labels: return "Balanced Customer Segment"
    return " / ".join(dict.fromkeys(labels[:3])).title()

def fit_segmentation(df, features, k_mode, k_custom):
    work=df.dropna(subset=features).copy()
    if len(work)<30: raise ValueError("Need at least 30 complete rows for clustering.")
    scaler=StandardScaler(); X=scaler.fit_transform(work[features])
    k_scores=[]
    for k in range(2,min(9,len(work)-1)):
        km=KMeans(n_clusters=k,n_init=20,random_state=42).fit(X)
        k_scores.append({"k":k,"silhouette":silhouette_score(X,km.labels_),"davies_bouldin":davies_bouldin_score(X,km.labels_),"inertia":km.inertia_})
    score_df=pd.DataFrame(k_scores)
    chosen=int(score_df.sort_values(["silhouette","davies_bouldin"],ascending=[False,True]).iloc[0].k) if k_mode=="Automatic" else int(k_custom)
    km=KMeans(n_clusters=chosen,n_init=30,random_state=42).fit(X)
    work["cluster"]=km.labels_
    prof=work.groupby("cluster")[features].mean()
    overall=work[features].mean()
    personas={int(i):persona_from_profile(prof.loc[i],overall) for i in prof.index}
    work["segment_name"]=work.cluster.map(personas)
    return work,scaler,km,prof,personas,score_df

source=st.sidebar.radio("Dataset",["Demo customer data","Upload CSV / Excel"])
if source.startswith("Demo"):
    df=demo_customers(); source_name="Synthetic customer PoC dataset"
else:
    up=st.sidebar.file_uploader("Upload customer dataset",type=["csv","xlsx","xls"])
    if not up: st.info("Upload customer data to continue."); st.stop()
    df=read_table(up); source_name=up.name

numeric=[c for c in df.select_dtypes(include=np.number).columns if not c.lower().endswith("id")]
defaults=[c for c in ["recency_days","frequency_12m","monetary_12m","annual_income"] if c in numeric]
features=st.sidebar.multiselect("Clustering features",numeric,default=defaults or numeric[:min(4,len(numeric))])
if len(features)<2: st.warning("Select at least two numeric features."); st.stop()
k_mode=st.sidebar.radio("Cluster count",["Automatic","Custom"])
k_custom=st.sidebar.slider("Number of clusters",2,8,4,disabled=k_mode=="Automatic")

try:
    segmented,scaler,km,profiles,personas,k_scores=fit_segmentation(df,features,k_mode,k_custom)
except Exception as e:
    st.exception(e); st.stop()

chosen=km.n_clusters
with st.sidebar:
    st.divider(); st.write(f"**Source:** {source_name}"); st.write(f"**Rows used:** {len(segmented):,}"); st.write(f"**Clusters:** {chosen}"); st.write(f"**Llama:** {'Connected' if ollama_available() else 'Ollama not detected'}")

tabs=st.tabs(["Segments & Personas","Test One Customer","Batch Assignment","Cluster Selection","Download"])
with tabs[0]:
    c1,c2,c3=st.columns(3)
    current=k_scores[k_scores.k==chosen].iloc[0]
    c1.metric("Clusters",chosen); c2.metric("Silhouette",f"{current.silhouette:.3f}"); c3.metric("Davies-Bouldin",f"{current.davies_bouldin:.3f}")
    summary=profiles.copy(); summary.insert(0,"persona",[personas[i] for i in summary.index]); summary.insert(0,"cluster",summary.index)
    st.dataframe(summary.reset_index(drop=True).round(2),use_container_width=True)
    X=scaler.transform(segmented[features]); coords=PCA(n_components=2,random_state=42).fit_transform(X)
    viz=segmented.copy(); viz["PC1"]=coords[:,0]; viz["PC2"]=coords[:,1]
    st.plotly_chart(px.scatter(viz,x="PC1",y="PC2",color=viz.cluster.astype(str),hover_name="segment_name",title="2D PCA view of customer segments"),use_container_width=True)
    cluster_choice=st.selectbox("Generate Llama persona explanation for cluster",list(personas.keys()),format_func=lambda i:f"Cluster {i}: {personas[i]}")
    if st.button("Generate persona strategy"):
        p=profiles.loc[cluster_choice].round(2).to_dict()
        st.info(ask_llama(f"You are a CRM analyst. Cluster {cluster_choice} has profile {p} and persona label '{personas[cluster_choice]}'. Explain who these customers are and give 3 practical retention/marketing actions. Use only these profile facts; no invented demographics."))

with tabs[1]:
    st.subheader("Assign a new customer to an existing segment")
    values={}
    cols=st.columns(2)
    for i,f in enumerate(features):
        series=segmented[f]; default=float(series.median()); step=max(float(series.std()/20),0.01)
        values[f]=cols[i%2].number_input(f,value=default,step=step,key=f"single_{f}")
    if st.button("Assign customer",type="primary"):
        row=pd.DataFrame([values]); z=scaler.transform(row[features]); cluster=int(km.predict(z)[0]); center=km.cluster_centers_[cluster]
        distance=float(np.linalg.norm(z[0]-center))
        a,b,c=st.columns(3); a.metric("Assigned cluster",cluster); b.metric("Persona",personas[cluster]); c.metric("Distance to centroid",f"{distance:.2f}")
        comp=pd.DataFrame({"Feature":features,"Customer":[values[f] for f in features],"Cluster average":[profiles.loc[cluster,f] for f in features]})
        st.dataframe(comp.round(2),use_container_width=True,hide_index=True)
        st.info(ask_llama(f"Explain why a customer with values {values} belongs to cluster {cluster} ('{personas[cluster]}'). Cluster averages are {profiles.loc[cluster].round(2).to_dict()}. Give a concise, non-causal explanation based only on comparison to the cluster profile."))

with tabs[2]:
    st.subheader("Assign uploaded customers using the current clustering model")
    up2=st.file_uploader("Upload CSV / Excel with the selected feature columns",type=["csv","xlsx","xls"],key="assign_batch")
    if up2:
        test=read_table(up2); missing=[f for f in features if f not in test.columns]
        if missing: st.error(f"Missing selected features: {missing}")
        else:
            valid=test.dropna(subset=features).copy(); valid["cluster"]=km.predict(scaler.transform(valid[features])); valid["segment_name"]=valid.cluster.map(personas)
            st.dataframe(valid,use_container_width=True); st.download_button("Download assigned customers",valid.to_csv(index=False),"assigned_customer_segments.csv","text/csv")

with tabs[3]:
    st.dataframe(k_scores.style.format({"silhouette":"{:.3f}","davies_bouldin":"{:.3f}","inertia":"{:.1f}"}),use_container_width=True,hide_index=True)
    st.plotly_chart(px.line(k_scores,x="k",y="silhouette",markers=True,title="Silhouette score by K"),use_container_width=True)
    st.caption("Automatic mode chooses the highest silhouette score, using Davies-Bouldin as a secondary criterion.")

with tabs[4]:
    st.download_button("Download segmented CSV",segmented.to_csv(index=False),"segmented_customers.csv","text/csv")
    st.download_button("Download segmented Excel",dataframe_to_excel_bytes(segmented),"segmented_customers.xlsx")
