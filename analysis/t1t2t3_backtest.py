import os, json, math, time, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PRICE_URL="https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet"
MEM_URL="https://github.com/Johnbrick123/sp500-data/releases/download/data/membership_intervals.parquet"
SEED=42
ENTRY_START=pd.Timestamp("2022-01-03")
ENTRY_END=pd.Timestamp("2025-03-31")  # leaves ~18 months follow-through to 2026-09
MAX_HOLD_DAYS=504
TRANCHE=10000.0

os.system(f"curl -L --retry 3 -o prices.parquet '{PRICE_URL}'")
os.system(f"curl -L --retry 3 -o membership.parquet '{MEM_URL}'")
px=pd.read_parquet("prices.parquet")
px.columns=[c.lower() for c in px.columns]
px["date"]=pd.to_datetime(px["date"])
px=px.sort_values(["ticker","date"])
max_date=px["date"].max()

# membership-aware current universe if schema permits
mem=pd.read_parquet("membership.parquet")
mem.columns=[c.lower() for c in mem.columns]
tcol=next((c for c in mem.columns if "ticker" in c or c in ("symbol","ticker")),None)
scol=next((c for c in mem.columns if "start" in c or c in ("from","date_added")),None)
ecol=next((c for c in mem.columns if "end" in c or c in ("to","date_removed")),None)
current=None
if tcol and scol:
    mem[scol]=pd.to_datetime(mem[scol])
    if ecol:
        mem[ecol]=pd.to_datetime(mem[ecol],errors="coerce")
        cur=mem[(mem[scol]<=max_date)&(mem[ecol].isna() | (mem[ecol]>=max_date))]
    else:
        cur=mem[mem[scol]<=max_date]
    current=set(cur[tcol].astype(str).str.upper())

latest=set(px.loc[px["date"]==max_date,"ticker"].astype(str).str.upper())
universe=sorted(latest if current is None else latest & current)

# exclude obvious ETFs / non-common symbols if present
etf_like={"SPY","QQQ","IWM","DIA","VOO","IVV","VTI","RSP","XLF","XLK","XLV","XLY","XLP","XLE","XLI","XLB","XLU","XLRE","VXX","GLD","SLV","TLT"}
universe=[t for t in universe if t not in etf_like]

# data preparation
def prep(g):
    g=g.sort_values("date").copy()
    g=g[g["date"]>=pd.Timestamp("2017-01-01")].copy()
    g["close"]=pd.to_numeric(g["close"],errors="coerce")
    g["high"]=pd.to_numeric(g["high"],errors="coerce")
    g["low"]=pd.to_numeric(g["low"],errors="coerce")
    g=g.dropna(subset=["close","high","low"])
    g["high252"]=g["close"].rolling(252,min_periods=200).max()
    g["dd"]=1-g["close"]/g["high252"]
    g["ema100"]=g["close"].ewm(span=100,adjust=False).mean()
    g["ema100_slope10"]=g["ema100"]/g["ema100"].shift(10)-1
    pc=g["close"].shift(1)
    tr=pd.concat([(g["high"]-g["low"]).abs(),(g["high"]-pc).abs(),(g["low"]-pc).abs()],axis=1).max(axis=1)
    g["atr14"]=tr.ewm(alpha=1/14,adjust=False).mean()
    w=g.set_index("date")["close"].resample("W-FRI").last().dropna().to_frame()
    w["wema200"]=w["close"].ewm(span=200,adjust=False).mean()
    g=g.set_index("date")
    g["wema200"]=w["wema200"].reindex(g.index,method="ffill")
    g["dist_wema200"]=g["close"]/g["wema200"]-1
    return g.reset_index()

groups={}
for t in universe:
    g=px[px["ticker"].astype(str).str.upper()==t]
    if len(g)>=1200:
        groups[t]=prep(g)

# Focus the random draw on names that actually experienced a >=25% drawdown in the entry window.
eligible=[]
for t,g in groups.items():
    m=g[(g["date"]>=ENTRY_START)&(g["date"]<=ENTRY_END)]
    if len(m) and m["dd"].max()>=0.25:
        eligible.append(t)
rng=np.random.default_rng(SEED)
sample=sorted(rng.choice(eligible,size=min(100,len(eligible)),replace=False).tolist())
print("MAX_DATE",max_date.date(),"current universe",len(universe),"eligible",len(eligible),"sample",len(sample))

# Fundamentals via Yahoo annual statements, evaluated with a 90-day lag to reduce look-ahead.
fund={}
try:
    import yfinance as yf
    for k,t in enumerate(sample,1):
        try:
            st=yf.Ticker(t).financials
            if st is None or st.empty:
                fund[t]=None
                continue
            st.columns=pd.to_datetime(st.columns)
            revrow=next((r for r in ["Total Revenue","Operating Revenue"] if r in st.index),None)
            nirow=next((r for r in ["Net Income","Net Income Common Stockholders"] if r in st.index),None)
            if not revrow or not nirow:
                fund[t]=None
            else:
                fund[t]={"dates":list(st.columns),"rev":st.loc[revrow].to_dict(),"ni":st.loc[nirow].to_dict()}
        except Exception:
            fund[t]=None
        if k%20==0: print("fund",k)
except Exception as e:
    print("FUNDAMENTALS_UNAVAILABLE",repr(e))
    fund={t:None for t in sample}

def fund_ok(t,entry_date):
    f=fund.get(t)
    if not f: return None
    cutoff=pd.Timestamp(entry_date)-pd.Timedelta(days=90)
    ds=sorted([pd.Timestamp(d) for d in f["dates"] if pd.Timestamp(d)<=cutoff],reverse=True)
    if len(ds)<2: return None
    d0,d1=ds[:2]
    try:
        r0=float(f["rev"][d0]); r1=float(f["rev"][d1]); ni=float(f["ni"][d0])
        if not all(np.isfinite([r0,r1,ni])): return None
        return (r0>r1) and (ni>0)
    except Exception: return None

def entry_after_rebound(g,start_i,dd_threshold,rebound,anchor_high=None,t1=False):
    n=len(g)
    low=np.inf
    eligible_started=False
    for i in range(start_i,n):
        row=g.iloc[i]
        if row["date"]<ENTRY_START: continue
        if row["date"]>ENTRY_END and t1: return None
        if anchor_high is None:
            eligible_now=(row["dd"]>=dd_threshold)
        else:
            eligible_now=(row["close"]<=anchor_high*(1-dd_threshold))
        if not eligible_started:
            if not eligible_now: continue
            eligible_started=True
            low=row["close"]
        else:
            low=min(low,row["close"])
        if row["close"]>=low*(1+rebound):
            if t1:
                ema_ok=(row["ema100_slope10"]>=0) or (row["close"]>=row["ema100"])
                w_ok=np.isfinite(row["wema200"]) and row["close"]<=1.10*row["wema200"]
                if not (ema_ok and w_ok): continue
            return i
    return None

def simulate(g,t1dd,t1reb,t2dd,t2reb,t3dd,t3reb,mechanical=False):
    # first T1 in window
    start=max(0,int(np.searchsorted(g["date"].values,np.datetime64(ENTRY_START))))
    i1=entry_after_rebound(g,start,t1dd,t1reb,None,True)
    if i1 is None: return None
    high_anchor=float(g.iloc[i1]["high252"])
    entries=[(i1,float(g.iloc[i1]["close"]),1)]
    # T2
    if mechanical:
        target2=entries[0][1]*0.80
        idx=np.where((np.arange(len(g))>i1)&(g["close"].values<=target2))[0]
        i2=int(idx[0]) if len(idx) else None
    else:
        i2=entry_after_rebound(g,i1+1,t2dd,t2reb,high_anchor,False)
    if i2 is not None:
        entries.append((i2,float(g.iloc[i2]["close"]),2))
        # T3
        if mechanical:
            target3=entries[1][1]*0.80
            idx=np.where((np.arange(len(g))>i2)&(g["close"].values<=target3))[0]
            i3=int(idx[0]) if len(idx) else None
        else:
            i3=entry_after_rebound(g,i2+1,t3dd,t3reb,high_anchor,False)
        if i3 is not None:
            entries.append((i3,float(g.iloc[i3]["close"]),3))
    shares=sum(TRANCHE/p for _,p,_ in entries)
    capital=TRANCHE*len(entries)
    avg=capital/shares
    last_i=min(len(g)-1,i1+MAX_HOLD_DAYS)
    stop_active=False
    stop=np.nan
    exit_i=None
    exit_px=None
    min_equity=1e99
    max_dd=0.0
    # stage entries are assumed filled if reached before exit; rebuild incrementally
    stage_ptr=1
    cur_sh=TRANCHE/entries[0][1]
    cur_cap=TRANCHE
    cur_avg=cur_cap/cur_sh
    for i in range(i1,last_i+1):
        while stage_ptr<len(entries) and i>=entries[stage_ptr][0]:
            cur_sh += TRANCHE/entries[stage_ptr][1]
            cur_cap += TRANCHE
            cur_avg = cur_cap/cur_sh
            stage_ptr+=1
        close=float(g.iloc[i]["close"]); low=float(g.iloc[i]["low"]); atr=float(g.iloc[i]["atr14"])
        equity=cur_sh*close
        pnl=(equity-cur_cap)/cur_cap
        max_dd=min(max_dd,pnl)
        candidate=close-2*atr if np.isfinite(atr) else np.nan
        if np.isfinite(candidate) and candidate>=cur_avg*1.05:
            stop_active=True
            stop=candidate if not np.isfinite(stop) else max(stop,candidate)
        if stop_active and low<=stop:
            exit_i=i; exit_px=stop; break
    end_i=exit_i if exit_i is not None else last_i
    # only stages whose entry occurred by exit count as deployed
    live_entries=[e for e in entries if e[0]<=end_i]
    sh=sum(TRANCHE/e[1] for e in live_entries); cap=TRANCHE*len(live_entries); avg=cap/sh
    pxout=exit_px if exit_i is not None else float(g.iloc[end_i]["close"])
    roi=(sh*pxout-cap)/cap
    return {
        "entry_date":str(g.iloc[i1]["date"].date()),"t1_px":entries[0][1],
        "stages":len(live_entries),"capital":cap,"avg_cost":avg,
        "exit_date":str(g.iloc[end_i]["date"].date()),"exit_px":pxout,
        "roi":roi,"max_dd":max_dd,"stop_activated":bool(stop_active),"exited":exit_i is not None,
        "fund_ok":fund_ok(str(g.iloc[i1].get("ticker","")),g.iloc[i1]["date"]) if False else None
    }

# evaluate fundamentals keyed separately because g doesn't retain ticker reliably
def sim_for_ticker(t,g,*args,**kwargs):
    r=simulate(g,*args,**kwargs)
    if r is not None: r["fund_ok"]=fund_ok(t,pd.Timestamp(r["entry_date"]))
    return r

t1dds=[0.25,0.30,0.35]
t1rebs=[0.03,0.05,0.075]
t2dds=[0.40,0.44,0.48]
t2rebs=[0.03,0.05,0.075]
t3dds=[0.50,0.55,0.60]
t3rebs=[0.03,0.05,0.075]

rows=[]
best_trades_store={}
for combo in itertools.product(t1dds,t1rebs,t2dds,t2rebs,t3dds,t3rebs):
    a,b,c,d,e,f=combo
    if not (c>a+0.05 and e>c+0.05): continue
    trades=[]
    for t in sample:
        r=sim_for_ticker(t,groups[t],a,b,c,d,e,f,mechanical=False)
        if r: r["ticker"]=t; trades.append(r)
    if not trades: continue
    td=pd.DataFrame(trades)
    q=td[td["fund_ok"]==True]
    use=q if len(q)>=25 else td
    t3=use[use["stages"]==3]
    fail3=(t3["roi"]<=0).mean() if len(t3) else 0.0
    rows.append({
        "t1_dd":a,"t1_reb":b,"t2_dd":c,"t2_reb":d,"t3_dd":e,"t3_reb":f,
        "n_all":len(td),"n_fund":len(q),"basis":"fund" if len(q)>=25 else "all",
        "win_rate":float((use["roi"]>0).mean()),"median_roi":float(use["roi"].median()),"mean_roi":float(use["roi"].mean()),
        "mean_max_dd":float(use["max_dd"].mean()),"t2_rate":float((use["stages"]>=2).mean()),
        "t3_rate":float((use["stages"]==3).mean()),"t3_fail_rate":float(fail3),
        "avg_capital":float(use["capital"].mean()),"stop_rate":float(use["stop_activated"].mean())
    })
res=pd.DataFrame(rows)

# transparent balanced score: reward median ROI/win; penalize T3 use, T3 failure, drawdown, capital.
res["score"]=(
    1.5*res["median_roi"] + 0.5*res["win_rate"] + 0.25*res["stop_rate"]
    - 1.5*res["t3_fail_rate"] - 0.35*res["t3_rate"] + 0.35*res["mean_max_dd"]
    - 0.10*((res["avg_capital"]-10000)/20000)
)
res=res.sort_values(["score","t3_fail_rate","median_roi"],ascending=[False,True,False])
res.to_csv("optimization_results.csv",index=False)

# Benchmark old 25% T1, then mechanical -20%/-20%, with 5% T1 confirmation.
bench=[]
for t in sample:
    r=sim_for_ticker(t,groups[t],0.25,0.05,0.40,0.05,0.55,0.05,mechanical=True)
    if r: r["ticker"]=t; bench.append(r)
bd=pd.DataFrame(bench)
bq=bd[bd["fund_ok"]==True] if len(bd) else pd.DataFrame()
bu=bq if len(bq)>=25 else bd

def metrics(df):
    if df is None or len(df)==0: return {}
    t3=df[df["stages"]==3]
    return {
        "n":int(len(df)),"win_rate":float((df["roi"]>0).mean()),"median_roi":float(df["roi"].median()),
        "mean_roi":float(df["roi"].mean()),"mean_max_dd":float(df["max_dd"].mean()),
        "t2_rate":float((df["stages"]>=2).mean()),"t3_rate":float((df["stages"]==3).mean()),
        "t3_fail_rate":float((t3["roi"]<=0).mean()) if len(t3) else 0.0,
        "avg_capital":float(df["capital"].mean()),"stop_rate":float(df["stop_activated"].mean())
    }

top=res.head(15).copy()
winner=top.iloc[0].to_dict() if len(top) else {}
wc=(winner.get("t1_dd"),winner.get("t1_reb"),winner.get("t2_dd"),winner.get("t2_reb"),winner.get("t3_dd"),winner.get("t3_reb"))
wtr=[]
if len(top):
    for t in sample:
        r=sim_for_ticker(t,groups[t],*wc,mechanical=False)
        if r: r["ticker"]=t; wtr.append(r)
pd.DataFrame(wtr).to_csv("winner_trades.csv",index=False)
bd.to_csv("benchmark_trades.csv",index=False)

summary={
    "data_end":str(max_date.date()),"sample_size":len(sample),"sample_tickers":sample,
    "eligible_universe":len(eligible),"fundamental_data_available":int(sum(fund.get(t) is not None for t in sample)),
    "winner":winner,"benchmark":metrics(bu),"benchmark_basis":"fund" if len(bq)>=25 else "all",
    "top15":top.to_dict(orient="records")
}
with open("summary.json","w") as f: json.dump(summary,f,indent=2)

lines=[]
lines.append("# T1/T2/T3 optimisation")
lines.append(f"Data through {max_date.date()}; random seed {SEED}; {len(sample)} names selected from current S&P members that experienced >=25% drawdown in entry window.")
lines.append(f"Yahoo annual fundamentals available for {summary['fundamental_data_available']}/{len(sample)} names. Ranking uses fundamental-qualified trades when >=25 are available for a rule; otherwise it falls back to the full technical sample and labels that basis.")
lines.append("")
lines.append("## Best balanced sequence")
for k in ["t1_dd","t1_reb","t2_dd","t2_reb","t3_dd","t3_reb","n_all","n_fund","basis","win_rate","median_roi","mean_roi","mean_max_dd","t2_rate","t3_rate","t3_fail_rate","avg_capital","stop_rate","score"]:
    if k in winner: lines.append(f"- {k}: {winner[k]}")
lines.append("")
lines.append("## Old benchmark")
for k,v in summary["benchmark"].items(): lines.append(f"- {k}: {v}")
lines.append("")
lines.append("## Top 15")
lines.append(top.to_markdown(index=False))
open("summary.md","w").write("\n".join(lines))
print(open("summary.md").read())
