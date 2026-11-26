"""Independent, replicate-supported CROWN-seq m6Am evidence with PCIF1-KO control."""
import csv,gzip,hashlib,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
O=R/'data/m6Am_candidates';D=O/'local_acquired'

class Fasta:
    def __init__(self,path):
        self.handle=Path(path).open('rb');self.index={}
        for line in Path(str(path)+'.fai').read_text().splitlines():
            name,n,offset,width,bytes_=line.split('\t')[:5];self.index[name]=tuple(map(int,[n,offset,width,bytes_]))
    def get(self,chrom,start,end):
        if chrom not in self.index:return ''
        n,offset,width,bytes_=self.index[chrom]
        if start<0 or end>n:return ''
        pos=offset+(start//width)*bytes_+start%width
        length=end-start
        self.handle.seek(pos)
        return self.handle.read(length+(length//width+2)*(bytes_-width)).replace(b'\n',b'').replace(b'\r',b'')[:length].decode().upper()

fa=Fasta(R/'data/reference/hg38.fa')
def read(name):
    path=next(D.glob(name))
    df=pd.read_csv(path,dtype={'Unnamed: 0':str})
    df=df.rename(columns={'Unnamed: 0':'chrom','Unnamed: 1':'source_pos','Unnamed: 2':'strand','Unnamed: 3':'reference_base'})
    df['chrom']='chr'+df.chrom.str.replace('chr','',regex=False)
    return path,df
p1,w1=read('GSM7431971*');p2,w2=read('GSM7431972*');pk,ko=read('GSM7431975*')
comp=str.maketrans('ACGT','TGCA')
coordinate=[]
sample=w1[w1.reference_base.isin(list('ACGT'))].iloc[::max(1,len(w1)//20000)]
for offset in [-1,0]:
    hits=0;total=0
    for r in sample.itertuples():
        seq=fa.get(r.chrom,int(r.source_pos)+offset,int(r.source_pos)+offset+1)
        if not seq:continue
        base=seq if r.strand=='+' else seq.translate(comp)
        hits+=int(base==r.reference_base);total+=1
    coordinate.append({'source_to_zero_based_offset':offset,'matches':hits,'tested':total,'fraction':hits/max(1,total)})
best=max(coordinate,key=lambda x:x['fraction'])
assert best['fraction']>=.98,coordinate
other=min(coordinate,key=lambda x:x['fraction'])
assert best['fraction']-other['fraction']>.1,coordinate
offset=best['source_to_zero_based_offset']
# Eligibility fixed independently of model outputs: adequate depth in both WT replicates,
# nonconversion in both replicates, loss in PCIF1 KO, and a BH-corrected count comparison.
thresholds={'WT_min_AG_reads_each':20,'WT_min_nonconversion_each':.10,'KO_min_AG_reads':20,'KO_max_nonconversion':.05,'min_WT_minus_KO_nonconversion':.10,'FDR':.05}
(O/'CROWNseq_qc_plan.json').write_text(json.dumps({'thresholds':thresholds,'source_doi':'10.7554/eLife.104139','GEO':'GSE233655','samples':[str(p1),str(p2),str(pk)],'not_claimed_to_be_author_final_site_list':True,'coordinate_validation':coordinate,'note':'This is a separately derived conservative PCIF1-dependent subset; not broad-peak centroids and not optimized against model performance.'},indent=2))
key=['chrom','source_pos','strand']
frames=[]
for tag,df in [('wt1',w1),('wt2',w2),('ko',ko)]:
    df=df[df.reference_base=='A'].copy()
    df['coverage']=df.A+df.G;df['fraction']=df.A/df.coverage.replace(0,np.nan)
    fields=key+['A','G','coverage','fraction']
    frames.append(df[fields].rename(columns={c:tag+'_'+c for c in fields if c not in key}))
merged=frames[0].merge(frames[1],on=key,validate='one_to_one').merge(frames[2],on=key,validate='one_to_one')
eligible=merged[(merged.wt1_coverage>=20)&(merged.wt2_coverage>=20)&(merged.ko_coverage>=20)].copy()
pvals=[]
for r in eligible.itertuples():pvals.append(fisher_exact([[r.wt1_A+r.wt2_A,r.wt1_G+r.wt2_G],[r.ko_A,r.ko_G]],alternative='greater').pvalue)
eligible['p_PCIF1']=pvals
order=np.argsort(pvals);q=np.ones(len(pvals));running=1.
for rank in range(len(order)-1,-1,-1):
    i=order[rank];running=min(running,pvals[i]*len(order)/(rank+1));q[i]=running
eligible['FDR_PCIF1']=q
eligible['WT_mean_nonconversion']=(eligible.wt1_fraction+eligible.wt2_fraction)/2
selected=eligible[(eligible.wt1_fraction>=.1)&(eligible.wt2_fraction>=.1)&(eligible.ko_fraction<=.05)&(eligible.WT_mean_nonconversion-eligible.ko_fraction>=.1)&(eligible.FDR_PCIF1<.05)].copy()
eligible.to_csv(O/'CROWNseq_depth_eligible_count_tests.tsv',sep='\t',index=False)
records=[];failures=[]
master=pd.read_csv(R/'data/frozen_v1/master.tsv',sep='\t',usecols=['chr','start','strand','mod_type'],dtype={'chr':str})
known=master.groupby(['chr','start','strand']).mod_type.agg(set).to_dict()
for r in selected.to_dict('records'):
    pos=int(r['source_pos'])+offset
    seq=fa.get(r['chrom'],pos-100,pos+101)
    if r['strand']=='-':seq=seq.translate(comp)[::-1]
    reason=''
    if len(seq)!=201 or set(seq)-set('ACGT'):reason='window_missing_or_noncanonical'
    elif seq[100]!='A':reason='center_not_A'
    labels=known.get((r['chrom'],pos,r['strand']),set())
    if labels-set(['m6Am']):reason='conflicting_existing_label'
    if reason:failures.append(dict(r,reason=reason,existing_labels=','.join(sorted(labels))));continue
    records.append(dict(r,site_id=f'CROWNseq_m6Am_{r["chrom"]}_{pos}_{r["strand"]}',chr=r['chrom'],start=pos,end=pos+1,mod_type='m6Am',class_index=8,center_base='A',sequence_41nt=seq[80:121],sequence_101nt=seq[50:151],sequence_201nt=seq,source='GSE233655_CROWNseq_PCIF1_dependent',source_build='GRCh38',source_coordinate_offset=offset,source_doi='10.7554/eLife.104139'))
pd.DataFrame(records).to_csv(O/'CROWNseq_m6Am_nonconflicting_hg38.tsv',sep='\t',index=False)
pd.DataFrame(failures).to_csv(O/'CROWNseq_qc_exclusions.tsv',sep='\t',index=False)
audit={'status':'QC_COMPLETED_NOT_YET_PROMOTED_TO_V2','source_rows':len(w1),'replicate_and_KO_shared_A_sites':len(merged),'depth_eligible':len(eligible),'PCIF1_supported_selected':len(selected),'nonconflicting_canonical_201nt_sites':len(records),'exclusions':dict(Counter(x['reason'] for x in failures)),'coordinate_validation':coordinate,'thresholds':thresholds,'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [p1,p2,pk]},'interpretation':'Independent m6Am evidence requires new clustered dataset and retraining; frozen-v1 test labels remain unchanged.'}
(O/'CROWNseq_m6Am_qc_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
