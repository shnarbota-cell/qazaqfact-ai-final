import os, re, sqlite3, math, json
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', override=True)

import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB='data/qazaqfact.db'
os.makedirs('data', exist_ok=True)

# QazaqFact AI v3.5 — calibrated evidence coverage prototype.
# Modes: Ask & Verify / Verify ready answer.
OFFICIAL_HOSTS = (
    'gov.kz','akorda.kz','adilet.zan.kz','stat.gov.kz','edu.gov.kz','uba.edu.kz','testcenter.kz',
    'gov.uk','usa.gov','who.int','un.org','unesco.org','oecd.org','worldbank.org','europa.eu','nasa.gov'
)
KZ_PRIORITY_HOSTS=('gov.kz','akorda.kz','adilet.zan.kz','stat.gov.kz','edu.gov.kz','uba.edu.kz','testcenter.kz','nu.edu.kz')
LOW_QUALITY_HOSTS=('reddit.com','youtube.com','youtu.be','tiktok.com','instagram.com','facebook.com','x.com','twitter.com')
UA={'User-Agent':'QazaqFactAI/3.0 educational research prototype (+school project)'}

def db_init():
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS checks(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, question TEXT, ai_answer TEXT,
        claim TEXT, score REAL, status TEXT, sources INTEGER, evidence REAL DEFAULT 0,
        consensus REAL DEFAULT 0, quality REAL DEFAULT 0, coverage REAL DEFAULT 0,
        method TEXT DEFAULT 'v3')''')
    cols={r[1] for r in cur.execute('PRAGMA table_info(checks)').fetchall()}
    for name, typ, default in [('evidence','REAL','0'),('consensus','REAL','0'),('quality','REAL','0'),('coverage','REAL','0'),('method','TEXT',"'v3'")]:
        if name not in cols: cur.execute(f'ALTER TABLE checks ADD COLUMN {name} {typ} DEFAULT {default}')
    con.commit(); con.close()

def looks_kazakhstan(text):
    t=text.lower()
    keys=('казахстан','қазақстан','астана','алматы','ақорда','акорда','мәжіліс','мажилис','тенге','қазақ','казах')
    return any(k in t for k in keys)

def split_claims(text):
    """Extract declarative, checkable-looking sentences without dropping short facts.

    v3.6 fix: short factual statements such as "Астана является столицей Казахстана."
    and "Город расположен на реке Ишим." must remain separate claims.
    """
    text=re.sub(r'\s+',' ',text.strip())
    raw=[s.strip(' •-\t') for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    opinion=('я думаю','мне кажется','по моему','по-моему','на мой взгляд','i think')
    instruction_starts=('введите ','нажмите ','выберите ','перейдите ','скопируйте ','откройте ')
    factual=[]
    for s in raw:
        words=s.split()
        sl=s.lower().strip()
        if len(words) < 3 or len(words) > 70:
            continue
        if s.endswith('?') or any(x in sl for x in opinion) or sl.startswith(instruction_starts):
            continue
        # Keep declarative sentences, including short copular/location facts.
        # A tiny verb/marker check filters headings and fragments while preserving ordinary facts.
        markers=r'\b(является|являются|был|была|были|стал|стала|стали|находится|расположен|расположена|расположены|составляет|составляют|имеет|имеют|называется|означает|происходит|позволяет|позволило|помог|способствовал|состоит|используется|влияет|отличается|принял|приняла|принято|отмечается|is|are|was|were|means|because|болып табылады|орналасқан|құрайды|дегеніміз|себебі|әсер етеді)\b'
        if re.search(r'\d',s) or re.search(markers,sl) or len(words) >= 6:
            factual.append(s)
    # Preserve order and remove exact duplicates.
    out=[]; seen=set()
    for s in factual:
        key=s.casefold()
        if key not in seen:
            seen.add(key); out.append(s)
    return out[:10]

def wiki_search(query, lang='ru', limit=3):
    try:
        r=requests.get(f'https://{lang}.wikipedia.org/w/api.php',params={'action':'query','list':'search','srsearch':query,'format':'json','utf8':1,'srlimit':limit},headers=UA,timeout=8)
        r.raise_for_status(); out=[]
        for x in r.json().get('query',{}).get('search',[]):
            title=x['title']; snippet=BeautifulSoup(x.get('snippet',''),'html.parser').get_text(' ')
            out.append({'title':title,'snippet':snippet,'url':f'https://{lang}.wikipedia.org/wiki/'+title.replace(' ','_'),'type':'encyclopedia'})
        return out
    except Exception: return []

def google_factcheck(query):
    key=os.getenv('GOOGLE_FACTCHECK_API_KEY','').strip()
    if not key:return []
    try:
        r=requests.get('https://factchecktools.googleapis.com/v1alpha1/claims:search',params={'query':query,'key':key,'pageSize':5},timeout=10); r.raise_for_status(); out=[]
        for c in r.json().get('claims',[]):
            for rev in c.get('claimReview',[])[:2]:
                out.append({'title':c.get('text','Fact check'),'snippet':f"Rating: {rev.get('textualRating','')} | Publisher: {rev.get('publisher',{}).get('name','')}",'url':rev.get('url',''),'type':'factcheck'})
        return out
    except Exception:return []

def serper_search(query, limit=8, kz_priority=False):
    key=os.getenv('SERPER_API_KEY','').strip()
    if not key:return [],'SERPER_API_KEY не найден'
    try:
        q=query
        if kz_priority and looks_kazakhstan(query): q=f'{query} Казахстан официальный источник'
        r=requests.post('https://google.serper.dev/search',headers={'X-API-KEY':key,'Content-Type':'application/json'},json={'q':q,'num':limit},timeout=12)
        if r.status_code!=200:return [],f'Serper HTTP {r.status_code}'
        out=[]
        for x in r.json().get('organic',[])[:limit]:
            out.append({'title':x.get('title',''),'snippet':x.get('snippet',''),'url':x.get('link',''),'type':'web'})
        return out,''
    except Exception as e:return [],f'Serper: {type(e).__name__}'

def source_quality(url,typ,kz_priority=False):
    if typ=='factcheck':return 1.0
    host=urlparse(url).netloc.lower().replace('www.','')
    if any(host==x or host.endswith('.'+x) for x in OFFICIAL_HOSTS): q=.98
    elif host.endswith('.edu') or '.edu.' in host or host.endswith('.ac.uk'): q=.90
    elif 'wikipedia.org' in host:q=.74
    elif any(x in host for x in LOW_QUALITY_HOSTS):q=.28
    elif host:q=.60
    else:q=.35
    if kz_priority and any(host==x or host.endswith('.'+x) for x in KZ_PRIORITY_HOSTS): q=min(1.0,q+.02)
    return q

def semantic_scores(claim,sources):
    texts=[claim]+[(s['title']+' '+s['snippet']).strip() for s in sources]
    if len(texts)<2:return []
    try:
        vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True).fit_transform(texts)
        return [float(x) for x in cosine_similarity(vec[0:1],vec[1:]).flatten()]
    except ValueError:return [0.0]*(len(texts)-1)

def rerank_sources(claim,sources,max_sources=6,kz_priority=False):
    # Strict relevance gate prevents authoritative but unrelated pages from inflating Trust Score.
    if not sources:return []
    sims=semantic_scores(claim,sources); ranked=[]
    for src,sim in zip(sources,sims):
        src['similarity']=sim; src['quality']=source_quality(src['url'],src['type'],kz_priority)
        host=urlparse(src['url']).netloc.lower().replace('www.','')
        kz_bonus=.06 if kz_priority and looks_kazakhstan(claim) and any(host==x or host.endswith('.'+x) for x in KZ_PRIORITY_HOSTS) else 0
        src['rank_score']=0.78*sim+0.22*src['quality']+kz_bonus
        if sim >= .105 or src['type']=='factcheck': ranked.append(src)
    ranked=sorted(ranked,key=lambda x:x['rank_score'],reverse=True)
    out=[]; seen_hosts=set()
    for src in ranked:
        host=urlparse(src['url']).netloc.lower().replace('www.','')
        if host and host in seen_hosts: continue
        seen_hosts.add(host); out.append(src)
        if len(out)>=max_sources: break
    return out

def trust_score(claim,sources):
    """Return a score only when evidence is sufficient to make a decision.

    Important: no evidence => N/A, never 0/100. Direct source positions are
    weighted by both source quality and relevance, so a weak accidental
    contradiction cannot outweigh stronger independent evidence.
    """
    empty={'E':0,'A':0,'Q':0,'N':0,'support':0,'refute':0,'unclear':0,
           'support_weight':0,'refute_weight':0,'decision':'insufficient'}
    if not sources:return None,empty
    relevant=[x for x in sources if x.get('similarity',0)>=.085 or x.get('stance') in ('support','refute')]
    if not relevant:return None,empty
    sims=sorted([x.get('similarity',0) for x in relevant],reverse=True)[:4]
    E=min(1.0,1-math.exp(-3.4*(sum(sims)/len(sims)))) if sims else 0
    support=[x for x in relevant if x.get('stance')=='support']
    refute=[x for x in relevant if x.get('stance')=='refute']
    unclear=sum(x.get('stance') not in ('support','refute') for x in relevant)

    def ev_weight(x):
        # Relevance matters more than authority: an authoritative unrelated page
        # must not decide the claim.
        rel=max(0.0,min(1.0,x.get('similarity',0)))
        qual=max(0.0,min(1.0,x.get('quality',.5)))
        return rel*(0.55+0.45*qual)
    sw=sum(ev_weight(x) for x in support)
    rw=sum(ev_weight(x) for x in refute)
    direct=len(support)+len(refute)
    A=(max(sw,rw)/(sw+rw)) if (sw+rw)>0 else 0.0
    Q=sum(x.get('quality',.5) for x in relevant)/len(relevant)
    N=min(1.0,len({urlparse(x['url']).netloc.lower() for x in relevant if x.get('url')})/3)

    # At least two independent direct positions are required for a scored claim.
    # One source is shown to the user, but remains N/A for the decision score.
    if direct < 2:
        decision='insufficient'; score=None
    elif len(refute)>=2 and rw > sw*1.20:
        decision='refuted'
        confidence=min(1.0, .45*A+.30*Q+.25*N)
        score=round(100*(1-confidence),1)
    elif len(support)>=2 and sw > rw*1.20:
        decision='supported'
        score=round(100*(.36*E+.29*A+.20*Q+.15*N),1)
    else:
        # There is enough direct evidence to inspect, but it conflicts.
        decision='mixed'
        score=round(50 + 20*(sw-rw)/(sw+rw),1) if (sw+rw)>0 else 50.0
        score=max(30.0,min(70.0,score))
    return score,{'E':round(E,3),'A':round(A,3),'Q':round(Q,3),'N':round(N,3),
        'support':len(support),'refute':len(refute),'unclear':unclear,
        'support_weight':round(sw,3),'refute_weight':round(rw,3),'decision':decision}

def status_from_parts(score,parts):
    d=parts.get('decision','insufficient')
    if d=='refuted':return 'Противоречит найденным источникам'
    if d=='supported':return 'Подтверждается несколькими источниками'
    if d=='mixed':return 'Источники расходятся — требуется проверка'
    return 'Недостаточно данных для вывода'

def claim_search_queries(claim,kz_priority=False):
    """Create several conservative search formulations; Gemini is used only to improve retrieval."""
    queries=[claim]
    # Exact wording often helps with dates/names.
    if len(claim)<220: queries.append('"'+claim.replace('"','')+'"')
    key=os.getenv('GEMINI_API_KEY','').strip()
    if key:
        prompt=(
            'Сформируй 2 коротких поисковых запроса для проверки фактического утверждения. '
            'Не отвечай на утверждение. Не добавляй факты. Верни только две строки без нумерации.\n'
            f'Утверждение: {claim}'
        )
        txt,_=gemini_generate(prompt,temperature=0,max_tokens=120)
        for line in txt.splitlines():
            q=re.sub(r'^[-•\\d.)\\s]+','',line).strip()
            if 4<=len(q)<=180: queries.append(q)
    if kz_priority and looks_kazakhstan(claim):
        # Separate official-source retrieval, not a replacement for independent sources.
        queries.append(claim+' site:gov.kz OR site:akorda.kz OR site:adilet.zan.kz')
    out=[]
    for q in queries:
        if q and q not in out:out.append(q)
    return out[:5]

def evidence_text_for_source(claim,src):
    """Use the most relevant complete sentences from a public page when readable; else snippet."""
    sents=fetch_public_sentences(src.get('url',''))[:100]
    if sents:
        sims=semantic_scores(claim,[{'title':'','snippet':x} for x in sents])
        best=sorted(zip(sents,sims),key=lambda z:z[1],reverse=True)[:3]
        picked=[x for x,sim in best if sim>=.055]
        if picked:return ' '.join(picked)[:2600]
    return (src.get('title','')+' — '+src.get('snippet',''))[:2200]

def stance_with_gemini(claim, source):
    key=os.getenv('GEMINI_API_KEY','').strip()
    if not key:return 'unclear'
    evidence=evidence_text_for_source(claim,source)
    if not evidence.strip():return 'unclear'
    prompt=(
        'Определи позицию ТОЛЬКО данного фрагмента относительно утверждения. '
        'SUPPORT = фрагмент прямо подтверждает утверждение; REFUTE = прямо ему противоречит; '
        'UNCLEAR = тема похожа, но прямого доказательства нет. При сомнении выбирай UNCLEAR. '
        'Верни ровно одно слово: SUPPORT, REFUTE или UNCLEAR.\n'
        f'Утверждение: {claim}\nФрагмент: {evidence}'
    )
    txt,_=gemini_generate(prompt,temperature=0,max_tokens=20)
    x=txt.strip().upper()
    return 'support' if x.startswith('SUPPORT') else ('refute' if x.startswith('REFUTE') else 'unclear')

def assess_evidence(claim,sources):
    for src in sources: src['stance']=stance_with_gemini(claim,src)
    return sources

def retrieve_for_claim(claim,lang='ru',kz_priority=True):
    sources=[]; errors=[]
    sources+=google_factcheck(claim)
    for q in claim_search_queries(claim,kz_priority):
        web,err=serper_search(q,7,False)
        sources+=web
        if err:errors.append(err)
    sources+=wiki_search(claim,lang,4)
    # URL + host/title deduplication before relevance ranking.
    seen=set(); ded=[]
    for src in sources:
        u=src.get('url','').strip(); host=urlparse(u).netloc.lower().replace('www.','')
        key=(u.rstrip('/'),host,src.get('title','')[:100].lower())
        if u and key not in seen:
            seen.add(key); ded.append(src)
    ranked=rerank_sources(claim,ded,8,kz_priority)
    return assess_evidence(claim,ranked),'; '.join(sorted(set(errors)))

def _clean_sentence(s):
    s=re.sub(r'\s+',' ',s).strip(' •-\t\r\n')
    # Reject obvious navigation/search noise and truncated snippets.
    if len(s.split()) < 6 or len(s) < 45:return ''
    if s.endswith('...') or s.endswith('…'):return ''
    bad=('cookie','войти','регистрация','меню','главная страница','поделиться','читать далее')
    if any(x in s.lower() for x in bad):return ''
    return s

def fetch_public_sentences(url, limit_chars=14000):
    """Read ordinary public HTML and return clean complete sentences. No bypasses."""
    try:
        if not url.startswith(('http://','https://')):return []
        r=requests.get(url,headers=UA,timeout=12,allow_redirects=True)
        if r.status_code!=200 or 'text/html' not in r.headers.get('content-type',''):return []
        soup=BeautifulSoup(r.text[:800000],'html.parser')
        for tag in soup(['script','style','nav','footer','header','aside','form','noscript']):tag.decompose()
        root=soup.find('main') or soup.find('article') or soup.body or soup
        text=' '.join(root.stripped_strings)[:limit_chars]
        raw=re.split(r'(?<=[.!?])\s+',text)
        out=[]
        for x in raw:
            x=_clean_sentence(x)
            if x and x not in out:out.append(x)
        return out[:120]
    except Exception:return []

def extractive_answer(question,sources):
    """Build a source-grounded brief from complete sentences, without another AI key."""
    candidates=[]
    for src in sources[:4]:
        for sent in fetch_public_sentences(src.get('url',''))[:60]:
            candidates.append((sent,src))
    # If pages cannot be read, use only complete (not truncated) search snippets.
    if not candidates:
        for src in sources:
            sn=_clean_sentence(src.get('snippet',''))
            if sn:candidates.append((sn,src))
    if not candidates:return ''
    sims=semantic_scores(question,[{'title':'','snippet':x[0]} for x in candidates])
    ranked=sorted(zip(candidates,sims),key=lambda z:z[1],reverse=True)
    chosen=[]; hosts=set()
    for (sent,src),sim in ranked:
        if sim < .06:continue
        host=urlparse(src.get('url','')).netloc.lower().replace('www.','')
        # Prefer diversity but allow a second sentence if needed.
        if any(semantic_scores(sent,[{'title':'','snippet':x}])[0]>.72 for x in chosen):continue
        chosen.append(sent); hosts.add(host)
        if len(chosen)>=3:break
    if not chosen:return ''
    return 'По найденным источникам: ' + ' '.join(chosen)

def gemini_generate(prompt, temperature=0.25, max_tokens=700):
    """Generate text with the user's Gemini API key. The key is read only from .env."""
    key=os.getenv('GEMINI_API_KEY','').strip()
    if not key:
        return '', 'В .env не найден GEMINI_API_KEY.'
    preferred=os.getenv('GEMINI_MODEL','gemini-2.5-flash').strip() or 'gemini-2.5-flash'
    models=[]
    for m in (preferred,'gemini-2.5-flash','gemini-3.1-flash-lite'):
        if m not in models: models.append(m)
    last_error=''
    for model in models:
        try:
            url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
            payload={
                'contents':[{'parts':[{'text':prompt}]}],
                'generationConfig':{'temperature':temperature,'maxOutputTokens':max_tokens}
            }
            r=requests.post(url,json=payload,timeout=35)
            if not r.ok:
                try: msg=r.json().get('error',{}).get('message','')
                except Exception: msg=r.text[:180]
                last_error=f'{model}: HTTP {r.status_code} {msg}'.strip()
                continue
            data=r.json(); parts=data.get('candidates',[{}])[0].get('content',{}).get('parts',[])
            text=''.join(x.get('text','') for x in parts).strip()
            if text: return text,''
            last_error=f'{model}: пустой ответ модели'
        except Exception as e:
            last_error=f'{model}: {type(e).__name__}'
    return '', 'Gemini не ответил. '+last_error

def answer_question(question,lang='ru',kz_priority=True):
    # Gemini creates the draft answer; QazaqFact verifies it independently afterwards.
    language='казахском' if lang=='kk' else 'русском'
    prompt=(
        f'Ты учебный помощник. Ответь на вопрос школьника на {language} языке. '
        'Дай связный, понятный ответ из 3–6 предложений. Не придумывай источники, ссылки, цитаты или точные данные, '
        'если не уверен. Не пиши Trust Score и не утверждай, что ответ уже проверен. '
        'Формулируй фактические утверждения отдельными полными предложениями, чтобы их можно было проверить.\n\n'
        f'Вопрос: {question}'
    )
    ans,err=gemini_generate(prompt,temperature=.2,max_tokens=650)
    return ans,[],err

def extract_page(url):
    """Reads only normally accessible public HTML; no login/paywall/bot-protection bypass."""
    try:
        if not url.startswith(('http://','https://')):return '', 'Ссылка должна начинаться с http:// или https://'
        r=requests.get(url,headers=UA,timeout=15,allow_redirects=True)
        r.raise_for_status()
        ctype=r.headers.get('content-type','')
        if 'text/html' not in ctype:return '', 'Страница не является обычной HTML-страницей.'
        soup=BeautifulSoup(r.text,'html.parser')
        for tag in soup(['script','style','nav','footer','header','aside','form','noscript']):tag.decompose()
        root=soup.find('main') or soup.find('article') or soup.body or soup
        text='\n'.join(x.strip() for x in root.stripped_strings)
        text=re.sub(r'\n{3,}','\n\n',text)
        if len(text)<120:return '', 'Не удалось извлечь содержательный текст. Возможно, сайт загружает его через JavaScript или ограничивает автоматическое чтение.'
        return text[:18000],''
    except requests.HTTPError as e:return '',f'Сайт не разрешил обычное чтение страницы (HTTP {e.response.status_code}).'
    except Exception as e:return '',f'Не удалось прочитать страницу: {type(e).__name__}.'

def save(question,answer,claim,score,stat,n,parts,method='v3.6'):
    con=sqlite3.connect(DB)
    con.execute('''INSERT INTO checks(ts,question,ai_answer,claim,score,status,sources,evidence,consensus,quality,coverage,method)
                   VALUES(datetime("now"),?,?,?,?,?,?,?,?,?,?,?)''',(question,answer,claim,score,stat,n,parts['E'],parts['A'],parts['Q'],parts['N'],method))
    con.commit();con.close()

def verify_text(question,text,lang='ru',kz_priority=True,method='v3.6'):
    claims=split_claims(text); results=[]; errors=[]
    for claim in claims:
        sources,err=retrieve_for_claim(claim,lang,kz_priority)
        if err:errors.append(err)
        sc,parts=trust_score(claim,sources); stat=status_from_parts(sc,parts)
        save(question,text,claim,sc,stat,len(sources),parts,method)
        results.append({'claim':claim,'score':sc,'parts':parts,'status':stat,'sources':sources})
    return results,errors

def show_passport(results):
    if not results:return
    supported=sum(r['parts'].get('decision')=='supported' for r in results)
    mixed=sum(r['parts'].get('decision')=='mixed' for r in results)
    insufficient=sum(r['parts'].get('decision')=='insufficient' for r in results)
    refuted=sum(r['parts'].get('decision')=='refuted' for r in results)
    scored=[r for r in results if r.get('score') is not None]
    coverage=round(100*len(scored)/len(results)) if results else 0
    overall=round(sum(r['score'] for r in scored)/len(scored),1) if scored else None

    st.subheader('🛡️ QazaqFact Trust Passport')
    a,b,c,d,e,f=st.columns(6)
    a.metric('Trust Score', f'{overall}/100' if overall is not None else 'N/A')
    b.metric('Проверено',f'{len(scored)} из {len(results)}')
    c.metric('Coverage',f'{coverage}%')
    d.metric('🟢 Подтверждено',supported)
    e.metric('🟠 Расхождение',mixed)
    f.metric('🔴 Противоречит',refuted)
    if insufficient: st.caption(f'⚪ Недостаточно данных: {insufficient}. Эти утверждения не получают 0/100 и не входят в общий Trust Score.')
    st.caption('Trust Score — индекс подтверждённости проверяемой части ответа найденными доказательствами; Coverage показывает, какую долю утверждений удалось оценить. Это не вероятность истинности.')

    for i,r in enumerate(results,1):
        with st.container(border=True):
            score_label=f"{r['score']}/100" if r.get('score') is not None else 'N/A'
            st.markdown(f"**{i}. {r['status']} — {score_label}**")
            st.write(r['claim']); p=r['parts']
            if p.get('support',0) or p.get('refute',0):
                st.caption(f"Источники: подтверждают — {p['support']}; противоречат — {p['refute']}; неясно — {p['unclear']}")
            else:
                st.caption('Прямой позиции источников не найдено: это недостаток доказательств, а не опровержение.')
            with st.expander('Подробнее: показатели и источники'):
                c1,c2,c3,c4=st.columns(4); c1.metric('Evidence',p['E']); c2.metric('Agreement',p['A']); c3.metric('Quality',p['Q']); c4.metric('Source diversity',p['N'])
                st.caption(f"Вес подтверждений: {p.get('support_weight',0):.3f} · вес противоречий: {p.get('refute_weight',0):.3f}")
                for j,src in enumerate(r['sources'][:5],1):
                    host=urlparse(src['url']).netloc.replace('www.',''); stance={'support':'подтверждает','refute':'противоречит','unclear':'неясно','related':'связанный материал'}.get(src.get('stance'),'')
                    st.markdown(f"{j}. [{src['title']}]({src['url']}) — {src['snippet'][:220]}  \n`{host}` · релевантность {src.get('similarity',0):.3f} · качество {src.get('quality',0):.2f} · {stance}")

def main():
    st.set_page_config(page_title='QazaqFact AI',page_icon='🔎',layout='wide')
    st.markdown('''
    <style>
    /* Keep desktop unchanged; compact only on phones. */
    @media (max-width: 640px) {
      .block-container {padding-top: 1rem !important; padding-left: 1rem !important; padding-right: 1rem !important;}
      h1 {font-size: 2.05rem !important; line-height: 1.08 !important; margin-bottom: .25rem !important;}
      [data-testid="stCaptionContainer"] {font-size: .92rem !important;}
      [data-testid="stAlert"] {padding: .75rem .85rem !important;}
      [data-testid="stAlert"] p {font-size: .96rem !important; line-height: 1.45 !important;}
      div[data-testid="stRadio"] label {font-size: .95rem !important;}
      .stButton > button {min-height: 2.8rem !important;}
      textarea {min-height: 7rem !important;}
    }
    </style>
    ''', unsafe_allow_html=True)
    db_init(); st.title('🔎 QazaqFact AI')
    st.caption('Учебный ИИ-помощник: спросить → проверить → увидеть источники и Trust Passport')
    st.info('Система оценивает подтверждённость утверждений найденными источниками. Она не определяет «абсолютную истину».')
    lang=st.selectbox('Язык поиска',['ru','kk'],format_func=lambda x:'Русский' if x=='ru' else 'Қазақша')
    kz=st.toggle('🇰🇿 KZ Priority для тем о Казахстане',value=True)
    mode=st.radio('Режим',['💬 Спросить и проверить','🔎 Проверить готовый ответ'],horizontal=True)

    if mode.startswith('💬'):
        q=st.text_area('Ваш учебный вопрос',height=90,placeholder='Например: Почему листья растений зелёные?')
        if st.button('Ответить и проверить',type='primary',width='stretch'):
            if not q.strip():st.error('Введите вопрос.');return
            with st.spinner('Формирую ответ и проверяю источники…'):
                ans,base_sources,err=answer_question(q,lang,kz)
                st.subheader('Ответ QazaqFact AI')
                if not ans:
                    st.warning('Не удалось сформировать ответ. Trust Score для служебного сообщения не рассчитывается.')
                    if err:st.caption('Диагностика Gemini: '+err)
                    return
                st.write(ans)
                results,errors=verify_text(q,ans,lang,kz,'ask_verify_final')
                show_passport(results)
                if err or errors:st.caption('Диагностика: '+ '; '.join(sorted(set(([err] if err else [])+errors))))

    elif mode.startswith('🔎'):
        q=st.text_input('Вопрос (необязательно)')
        ans=st.text_area('Вставьте ответ ИИ или найденный учебный текст',height=220)
        if st.button('Проверить ответ',type='primary',width='stretch'):
            if not ans.strip():st.error('Вставьте текст.');return
            with st.spinner('Проверяю…'):show_passport(verify_text(q,ans,lang,kz,'paste_final')[0])

    with st.expander('Как рассчитывается Trust Score'):
        st.markdown('**E — Evidence:** близость утверждения и найденных доказательств.  \n**A — Agreement:** согласованность прямых позиций источников: подтверждает / противоречит / неясно.  \n**Q — Quality:** качество источников.  \n**N — Source diversity:** охват независимых доменов.  \nTrust Score рассчитывается только для утверждений, по которым есть не менее двух независимых прямых позиций источников. При нехватке доказательств показывается N/A, а не 0/100. Отдельный Coverage показывает долю утверждений, которую удалось оценить. Веса необходимо калибровать только на TRAIN и заморозить перед TEST.')
    with st.expander('Журнал эксперимента'):
        if os.path.exists(DB):
            import pandas as pd
            con=sqlite3.connect(DB);df=pd.read_sql_query('SELECT * FROM checks ORDER BY id DESC LIMIT 300',con);con.close()
            st.dataframe(df,width='stretch');st.download_button('Скачать журнал CSV',df.to_csv(index=False).encode('utf-8-sig'),'qazaqfact_experiment_log.csv','text/csv')

if __name__=='__main__':main()
