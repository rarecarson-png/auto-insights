from pathlib import Path
from html.parser import HTMLParser
import re,html
R=Path(__file__).resolve().parent
links=[('index.html','Home'),('index.html#assignments','Coursework'),('dataset.html','Dataset'),('visualizations.html','Charts'),('models.html','16 Models'),('index.html#midterm','Midterm')]
class Links(HTMLParser):
 def __init__(self):super().__init__();self.items=[];self.href=None;self.txt=''
 def handle_starttag(self,t,a):
  if t=='a':self.href=dict(a).get('href');self.txt=''
 def handle_data(self,d):
  if self.href:self.txt+=d
 def handle_endtag(self,t):
  if t=='a' and self.href:self.items.append((self.href,self.txt));self.href=None
for p in R.glob('*.html'):
 s=p.read_text();m=re.search(r'<header\b[^>]*>.*?</header>',s,re.S|re.I)
 old=Links()
 if m:old.feed(m.group())
 nav=''.join(f'<a href="{url}"'+(' aria-current="page"' if url==p.name else '')+f'>{name}</a>' for url,name in links)
 header=f'<header class="ai-header"><div class="ai-bar"><a class="ai-logo" href="index.html">AUTO<span>INSIGHTS</span></a><nav class="ai-links" aria-label="Main navigation">{nav}</nav><details class="ai-mobile"><summary>Menu</summary><nav aria-label="Mobile navigation">{nav}</nav></details></div></header>'
 local=[(u,t) for u,t in old.items if t.strip() and 'AUTO' not in t.upper()]
 if p.name=='index.html':local=[('#overview','Overview'),('#proposal','Research proposal'),('#research-plan','Research plan'),('#assignments','Assignments 1–12'),('#midterm','Midterm'),('#models','Model research'),('#results','Results'),('#contribution','Contribution'),('#next-steps','Next steps'),('#references','References')]
 if local:header+='<aside class="ai-local"><details><summary>On this page & related work</summary><nav aria-label="Section navigation">'+''.join(f'<a href="{html.escape(u,quote=True)}">{html.escape(t.strip())}</a>' for u,t in local)+'</nav></details></aside>'
 s=re.sub(r'<header\b[^>]*>.*?</header>',lambda _:header,s,count=1,flags=re.S|re.I) if m else s.replace('<body>','<body>'+header,1)
 s=s.replace('</head>','<link rel="stylesheet" href="navigation.css"></head>');p.write_text(s)
print('Shared navigation applied.')
