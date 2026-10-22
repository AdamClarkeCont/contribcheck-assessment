import os, json, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PURL="https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet"
SEED=42; START=pd.Timestamp("2022-01-03"); END=pd.Timestamp("2025-03-31"); HOLD=504; TR=10000.0
os.system(f"curl -L --retry 3 -o prices.parquet '{PURL}'")
px=pd.read_parquet("prices.parquet"); px.columns=[c.lower() for c in px.columns]
px["date"]=pd.to_datetime(px["date"]); px["ticker"]=px["ticker"].astype(str).str.upper()
latest=px["date"].max(); univ=sorted(px.loc[px.date==latest,"ticker"].unique())

def prep(g):
 g=g.sort_values("date").copy(); g=g[g.date>=pd.Timestamp("2017-01-01")]
 for c in ["close","high","low"]: g[c]=pd.to_numeric(g[c],errors="coerce")
 g=g.dropna(subset=["close","high","low"])
 g["high252"]=g.close.rolling(252,min_periods=200).max(); g["dd"]=1-g.close/g.high252
 g["ema100"]=g.close.ewm(span=100,adjust=False).mean(); g["eslope"]=g.ema100/g.ema100.shift(10)-1
 pc=g.close.shift(); trr=pd.concat([(g.high-g.low).abs(),(g.high-pc).abs(),(g.low-pc).abs()],axis=1).max(axis=1)
 g["atr"]=trr.ewm(alpha=1/14,adjust=False).mean()
 w=g.set_index("date").close.resample("W-FRI").last().dropna().to_frame("c"); w["wema200"]=w.c.ewm(span=200,adjust=False).mean()
 g=g.set_index("date"); g["wema200"]=w.wema200.reindex(g.index,method="ffill"); return g.reset_index()

G={}; eligible=[]
for t in univ:
 g0=px[px.ticker==t]
 if len(g0)<1200: continue
 g=prep(g0); m=g[(g.date>=START)&(g.date<=END)]
 if len(m) and m.dd.max()>=0.25: G[t]=g; eligible.append(t)
rng=np.random.default_rng(SEED); sample=sorted(rng.choice(eligible,min(100,len(eligible)),replace=False).tolist())
print("DATA_END",latest.date(),"ELIGIBLE",len(eligible),"SAMPLE",len(sample),flush=True)

def find_entry(g, start_i, dd, reb, anchor=None, first=False):
 close=g.close.values; dates=g.date.values; dds=g.dd.values; ema=g.ema100.values; es=g.eslope.values; we=g.wema200.values
 low=np.inf; active=False
 for i in range(start_i,len(g)):
  dt=pd.Timestamp(dates[i])
  if first and dt>END: return None
  if dt<START: continue
  ok=(dds[i]>=dd) if anchor is None else (close[i]<=anchor*(1-dd))
  if not active:
   if not ok: continue
   active=True; low=close[i]
  else: low=min(low,close[i])
  if close[i] >= low*(1+reb):
   if first:
    emaok=(es[i]>=0) or (close[i]>=ema[i]); wok=np.isfinite(we[i]) and close[i]<=1.10*we[i]
    if not (emaok and wok): continue
   return i
 return None

def sim(g, t1dd,t1reb,t2dd,t2reb,t3dd,t3reb, mechanical=False):
 s=int(np.searchsorted(g.date.values,np.datetime64(START)))
 i1=find_entry(g,s,t1dd,t1reb,None,True)
 if i1 is None:return None
 anchor=float(g.high252.iloc[i1]); ent=[(i1,float(g.close.iloc[i1]))]
 if mechanical:
  ids=np.where((np.arange(len(g))>i1)&(g.close.values<=ent[0][1]*.8))[0]; i2=int(ids[0]) if len(ids) else None
 else:i2=find_entry(g,i1+1,t2dd,t2reb,anchor,False)
 if i2 is not None:
  ent.append((i2,float(g.close.iloc[i2])))
  if mechanical:
   ids=np.where((np.arange(len(g))>i2)&(g.close.values<=ent[1][1]*.8))[0]; i3=int(ids[0]) if len(ids) else None
  else:i3=find_entry(g,i2+1,t3dd,t3reb,anchor,False)
  if i3 is not None:ent.append((i3,float(g.close.iloc[i3])))
 end=min(len(g)-1,i1+HOLD); ptr=1; sh=TR/ent[0][1]; cap=TR; avg=cap/sh; stop=np.nan; stopon=False; mdd=0
 for i in range(i1,end+1):
  while ptr<len(ent) and i>=ent[ptr][0]:
   sh+=TR/ent[ptr][1];cap+=TR;avg=cap/sh;ptr+=1
  c=float(g.close.iloc[i]); lo=float(g.low.iloc[i]); atr=float(g.atr.iloc[i])
  mdd=min(mdd,(sh*c-cap)/cap)
  cand=c-2*atr
  if np.isfinite(cand) and cand>=avg*1.05: stopon=True; stop=cand if not np.isfinite(stop) else max(stop,cand)
  if stopon and lo<=stop: end=i; out=stop; break
 else: out=float(g.close.iloc[end])
 live=[e for e in ent if e[0]<=end]; sh=sum(TR/p for _,p in live);cap=TR*len(live);avg=cap/sh
 return {"stages":len(live),"capital":cap,"avg":avg,"roi":(sh*out-cap)/cap,"mdd":mdd,"stop":stopon}

combos=[]
for t1dd,t1reb,t2dd,t3dd in itertools.product([.25,.30,.35],[.03,.05,.075],[.40,.44,.48],[.50,.55,.60]):
 if t2dd<=t1dd+.05 or t3dd<=t2dd+.05: continue
 # hold T2/T3 reversal at 5%, then separately compare 3%/7.5% for the top structural levels
 trades=[]
 for t in sample:
  r=sim(G[t],t1dd,t1reb,t2dd,.05,t3dd,.05)
  if r: trades.append(r)
 if len(trades)<20:continue
 d=pd.DataFrame(trades); t3=d[d.stages==3]
 combos.append({"t1_dd":t1dd,"t1_reb":t1reb,"t2_dd":t2dd,"t2_reb":.05,"t3_dd":t3dd,"t3_reb":.05,
 "n":len(d),"win":(d.roi>0).mean(),"med":d.roi.median(),"mean":d.roi.mean(),"mdd":d.mdd.mean(),
 "t2rate":(d.stages>=2).mean(),"t3rate":(d.stages==3).mean(),"t3fail":(t3.roi<=0).mean() if len(t3) else 0,
 "avgcap":d.capital.mean(),"stoprate":d.stop.mean()})
res=pd.DataFrame(combos)
res["score"]=1.5*res.med+.5*res.win+.25*res.stoprate-1.5*res.t3fail-.35*res.t3rate+.35*res.mdd-.10*((res.avgcap-10000)/20000)
res=res.sort_values(["score","t3fail","med"],ascending=[False,True,False])
best=res.iloc[0].to_dict()

# refine rebound strengths around best drawdown structure
ref=[]
for r2,r3 in itertools.product([.03,.05,.075],[.03,.05,.075]):
 trades=[]
 for t in sample:
  z=sim(G[t],best["t1_dd"],best["t1_reb"],best["t2_dd"],r2,best["t3_dd"],r3)
  if z:trades.append(z)
 d=pd.DataFrame(trades);t3=d[d.stages==3]
 ref.append({"t1_dd":best["t1_dd"],"t1_reb":best["t1_reb"],"t2_dd":best["t2_dd"],"t2_reb":r2,"t3_dd":best["t3_dd"],"t3_reb":r3,
 "n":len(d),"win":(d.roi>0).mean(),"med":d.roi.median(),"mean":d.roi.mean(),"mdd":d.mdd.mean(),"t2rate":(d.stages>=2).mean(),
 "t3rate":(d.stages==3).mean(),"t3fail":(t3.roi<=0).mean() if len(t3) else 0,"avgcap":d.capital.mean(),"stoprate":d.stop.mean()})
rr=pd.DataFrame(ref);rr["score"]=1.5*rr.med+.5*rr.win+.25*rr.stoprate-1.5*rr.t3fail-.35*rr.t3rate+.35*rr.mdd-.10*((rr.avgcap-10000)/20000)
rr=rr.sort_values(["score","t3fail","med"],ascending=[False,True,False]); winner=rr.iloc[0].to_dict()

# old benchmark
bt=[]
for t in sample:
 z=sim(G[t],.25,.05,.40,.05,.55,.05,mechanical=True)
 if z:bt.append(z)
bd=pd.DataFrame(bt);t3=bd[bd.stages==3]
bench={"n":len(bd),"win":(bd.roi>0).mean(),"med":bd.roi.median(),"mean":bd.roi.mean(),"mdd":bd.mdd.mean(),
"t2rate":(bd.stages>=2).mean(),"t3rate":(bd.stages==3).mean(),"t3fail":(t3.roi<=0).mean() if len(t3) else 0,
"avgcap":bd.capital.mean(),"stoprate":bd.stop.mean()}

out={"data_end":str(latest.date()),"sample":sample,"winner":winner,"benchmark":bench,"top_structural":res.head(10).to_dict("records"),"refined":rr.to_dict("records")}
open("fast_summary.json","w").write(json.dumps(out,indent=2))
pd.DataFrame(res).to_csv("fast_grid.csv",index=False)
print(json.dumps(out,indent=2),flush=True)
