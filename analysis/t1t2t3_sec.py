import os,json,warnings,itertools,time,urllib.request
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
PURL="https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet"
MURL="https://github.com/Johnbrick123/sp500-data/releases/download/data/membership_intervals.parquet"
SEED=42;START=pd.Timestamp("2022-01-03");END=pd.Timestamp("2025-03-31");HOLD=504;TR=10000.
UA="research backtest contact@example.com"
os.system(f"curl -L --retry 3 -o prices.parquet '{PURL}'");os.system(f"curl -L --retry 3 -o membership.parquet '{MURL}'")
px=pd.read_parquet("prices.parquet");px.columns=[c.lower() for c in px.columns];px["date"]=pd.to_datetime(px.date);px["ticker"]=px.ticker.astype(str).str.upper()
mem=pd.read_parquet("membership.parquet");mem.columns=[c.lower() for c in mem.columns];mem["ticker"]=mem.ticker.astype(str).str.upper();mem["start"]=pd.to_datetime(mem.start);mem["end"]=pd.to_datetime(mem.end,errors="coerce")
# tickers that were members at some point in entry window
mwin=mem[(mem.start<=END)&(mem.end.isna()|(mem.end>=START))]
univ=sorted(set(mwin.ticker))
intervals={t:[(r.start,r.end if pd.notna(r.end) else pd.Timestamp.max) for _,r in d.iterrows()] for t,d in mwin.groupby("ticker")}
def member_at(t,dt):
 dt=pd.Timestamp(dt);return any(a<=dt<=b for a,b in intervals.get(t,[]))

def prep(g):
 g=g.sort_values("date").copy();g=g[g.date>=pd.Timestamp("2017-01-01")]
 for c in ["close","high","low"]:g[c]=pd.to_numeric(g[c],errors="coerce")
 g=g.dropna(subset=["close","high","low"]);g["high252"]=g.close.rolling(252,min_periods=200).max();g["dd"]=1-g.close/g.high252
 g["ema100"]=g.close.ewm(span=100,adjust=False).mean();g["eslope"]=g.ema100/g.ema100.shift(10)-1
 pc=g.close.shift();trr=pd.concat([(g.high-g.low).abs(),(g.high-pc).abs(),(g.low-pc).abs()],axis=1).max(axis=1);g["atr"]=trr.ewm(alpha=1/14,adjust=False).mean()
 w=g.set_index("date").close.resample("W-FRI").last().dropna().to_frame("c");w["wema200"]=w.c.ewm(span=200,adjust=False).mean()
 g=g.set_index("date");g["wema200"]=w.wema200.reindex(g.index,method="ffill");return g.reset_index()
G={};eligible=[]
for t in univ:
 g0=px[px.ticker==t]
 if len(g0)<1200:continue
 g=prep(g0);m=g[(g.date>=START)&(g.date<=END)].copy()
 if len(m):
  mask=np.array([member_at(t,d) for d in m.date])
  if mask.any() and m.loc[mask,"dd"].max()>=.25:G[t]=g;eligible.append(t)
rng=np.random.default_rng(SEED);sample=sorted(rng.choice(eligible,min(100,len(eligible)),replace=False).tolist())
print("UNIV",len(univ),"ELIGIBLE",len(eligible),"SAMPLE",len(sample),sample,flush=True)

def getjson(url):
 req=urllib.request.Request(url,headers={"User-Agent":UA,"Accept-Encoding":"identity"})
 with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
tickmap=getjson("https://www.sec.gov/files/company_tickers.json")
cikmap={v["ticker"].upper():str(v["cik_str"]).zfill(10) for v in tickmap.values()}
REV_TAGS=["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues","SalesRevenueNet","RevenueFromContractWithCustomerIncludingAssessedTax"]
NI_TAGS=["NetIncomeLoss","ProfitLoss"]
facts={}
def annual_series(usgaap,tags):
 for tag in tags:
  obj=usgaap.get(tag)
  if not obj:continue
  units=obj.get("units",{})
  arr=units.get("USD")
  if not arr:continue
  rec=[]
  for x in arr:
   if x.get("form") not in ("10-K","10-K/A"):continue
   if x.get("fp") not in ("FY",None):continue
   if "filed" not in x or "end" not in x or "val" not in x:continue
   rec.append({"filed":pd.Timestamp(x["filed"]),"end":pd.Timestamp(x["end"]),"val":float(x["val"])})
  if rec:return rec
 return []
for k,t in enumerate(sample,1):
 cik=cikmap.get(t.replace(".","-")) or cikmap.get(t)
 if not cik:facts[t]=None;continue
 try:
  cf=getjson(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
  u=cf.get("facts",{}).get("us-gaap",{})
  rev=annual_series(u,REV_TAGS);ni=annual_series(u,NI_TAGS);facts[t]={"rev":rev,"ni":ni} if rev and ni else None
 except Exception as e:facts[t]=None
 time.sleep(.11)
 if k%20==0:print("SEC",k,"available",sum(v is not None for v in facts.values()),flush=True)
print("FUND_AVAILABLE",sum(v is not None for v in facts.values()),flush=True)

def values_asof(arr,dt):
 arr=[x for x in arr if x["filed"]<=pd.Timestamp(dt)]
 by={}
 for x in sorted(arr,key=lambda z:z["filed"]):by[x["end"]]=x["val"]
 return sorted(by.items(),key=lambda kv:kv[0],reverse=True)
def fund_ok(t,dt):
 f=facts.get(t)
 if not f:return None
 r=values_asof(f["rev"],dt);n=values_asof(f["ni"],dt)
 if len(r)<2 or len(n)<1:return None
 latest_end=r[0][0];rev0,rev1=r[0][1],r[1][1]
 # find NI matching latest revenue FY end within 45 days
 candidates=[(e,v) for e,v in n if abs((e-latest_end).days)<=45]
 if not candidates:return None
 ni=candidates[0][1]
 return rev0>rev1 and ni>0

def find_entry(t,g,start_i,dd,reb,anchor=None,first=False):
 c=g.close.values;dates=g.date.values;dds=g.dd.values;ema=g.ema100.values;es=g.eslope.values;we=g.wema200.values
 low=np.inf;active=False
 for i in range(start_i,len(g)):
  dt=pd.Timestamp(dates[i])
  if first and dt>END:return None
  if dt<START:continue
  if first and not member_at(t,dt):continue
  ok=(dds[i]>=dd) if anchor is None else (c[i]<=anchor*(1-dd))
  if not active:
   if not ok:continue
   active=True;low=c[i]
  else:low=min(low,c[i])
  if c[i]>=low*(1+reb):
   if first:
    emaok=(es[i]>=0) or (c[i]>=ema[i]);wok=np.isfinite(we[i]) and c[i]<=1.10*we[i]
    if not(emaok and wok):continue
   return i
 return None
def sim(t,g,t1dd,t1reb,t2dd,t2reb,t3dd,t3reb,mechanical=False):
 s=int(np.searchsorted(g.date.values,np.datetime64(START)));i1=find_entry(t,g,s,t1dd,t1reb,None,True)
 if i1 is None or fund_ok(t,g.date.iloc[i1]) is not True:return None
 anchor=float(g.high252.iloc[i1]);ent=[(i1,float(g.close.iloc[i1]))]
 if mechanical:
  ids=np.where((np.arange(len(g))>i1)&(g.close.values<=ent[0][1]*.8))[0];i2=int(ids[0]) if len(ids) else None
 else:i2=find_entry(t,g,i1+1,t2dd,t2reb,anchor,False)
 if i2 is not None and fund_ok(t,g.date.iloc[i2]) is True:
  ent.append((i2,float(g.close.iloc[i2])))
  if mechanical:
   ids=np.where((np.arange(len(g))>i2)&(g.close.values<=ent[1][1]*.8))[0];i3=int(ids[0]) if len(ids) else None
  else:i3=find_entry(t,g,i2+1,t3dd,t3reb,anchor,False)
  if i3 is not None and fund_ok(t,g.date.iloc[i3]) is True:ent.append((i3,float(g.close.iloc[i3])))
 end=min(len(g)-1,i1+HOLD);ptr=1;sh=TR/ent[0][1];cap=TR;avg=cap/sh;stop=np.nan;stopon=False;mdd=0
 for i in range(i1,end+1):
  while ptr<len(ent) and i>=ent[ptr][0]:sh+=TR/ent[ptr][1];cap+=TR;avg=cap/sh;ptr+=1
  c=float(g.close.iloc[i]);lo=float(g.low.iloc[i]);atr=float(g.atr.iloc[i]);mdd=min(mdd,(sh*c-cap)/cap);cand=c-2*atr
  if np.isfinite(cand) and cand>=avg*1.05:stopon=True;stop=cand if not np.isfinite(stop) else max(stop,cand)
  if stopon and lo<=stop:end=i;out=stop;break
 else:out=float(g.close.iloc[end])
 live=[e for e in ent if e[0]<=end];sh=sum(TR/p for _,p in live);cap=TR*len(live);avg=cap/sh
 return {"stages":len(live),"capital":cap,"avg":avg,"roi":(sh*out-cap)/cap,"mdd":mdd,"stop":stopon}
def metrics(d):
 t3=d[d.stages==3]
 return {"n":len(d),"win":(d.roi>0).mean(),"med":d.roi.median(),"mean":d.roi.mean(),"mdd":d.mdd.mean(),"t2rate":(d.stages>=2).mean(),
 "t3rate":(d.stages==3).mean(),"t3fail":(t3.roi<=0).mean() if len(t3) else 0,"avgcap":d.capital.mean(),"stoprate":d.stop.mean()}
rows=[]
for t1dd,t1reb,t2dd,t3dd in itertools.product([.25,.30,.35],[.03,.05,.075],[.40,.44,.48],[.50,.55,.60]):
 if t2dd<=t1dd+.05 or t3dd<=t2dd+.05:continue
 tr=[z for t in sample if (z:=sim(t,G[t],t1dd,t1reb,t2dd,.05,t3dd,.05)) is not None]
 if len(tr)<15:continue
 d=pd.DataFrame(tr);m=metrics(d);m.update({"t1_dd":t1dd,"t1_reb":t1reb,"t2_dd":t2dd,"t2_reb":.05,"t3_dd":t3dd,"t3_reb":.05});rows.append(m)
res=pd.DataFrame(rows)
if res.empty:raise RuntimeError("No combinations produced >=15 fundamental-qualified trades")
res["score"]=1.5*res["med"]+.5*res["win"]+.25*res["stoprate"]-1.5*res["t3fail"]-.35*res["t3rate"]+.35*res["mdd"]-.10*((res["avgcap"]-10000)/20000)
res=res.sort_values(["score","t3fail","med"],ascending=[False,True,False]);base=res.iloc[0].to_dict()
ref=[]
for r2,r3 in itertools.product([.03,.05,.075],[.03,.05,.075]):
 tr=[z for t in sample if (z:=sim(t,G[t],base["t1_dd"],base["t1_reb"],base["t2_dd"],r2,base["t3_dd"],r3)) is not None]
 d=pd.DataFrame(tr);m=metrics(d);m.update({"t1_dd":base["t1_dd"],"t1_reb":base["t1_reb"],"t2_dd":base["t2_dd"],"t2_reb":r2,"t3_dd":base["t3_dd"],"t3_reb":r3});ref.append(m)
rr=pd.DataFrame(ref);rr["score"]=1.5*rr["med"]+.5*rr["win"]+.25*rr["stoprate"]-1.5*rr["t3fail"]-.35*rr["t3rate"]+.35*rr["mdd"]-.10*((rr["avgcap"]-10000)/20000)
rr=rr.sort_values(["score","t3fail","med"],ascending=[False,True,False]);winner=rr.iloc[0].to_dict()
bt=[z for t in sample if (z:=sim(t,G[t],.25,.05,.40,.05,.55,.05,mechanical=True)) is not None];bench=metrics(pd.DataFrame(bt))
out={"data_end":str(px.date.max().date()),"sample_size":len(sample),"fund_available":sum(v is not None for v in facts.values()),"winner":winner,"benchmark":bench,"top_structural":res.head(10).to_dict("records"),"sample":sample}
open("sec_summary.json","w").write(json.dumps(out,indent=2));res.to_csv("sec_grid.csv",index=False);print(json.dumps(out,indent=2),flush=True)
