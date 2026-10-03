"""Compile isolated layout probes; leave the manuscript working tree untouched."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import fitz

parser = argparse.ArgumentParser()
parser.add_argument('--paper', type=Path, default=Path('paper'))
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--variants', nargs='+', default=['baseline', 'ragged-bottom', 'flexible-figures'])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = args.paper.resolve()
root = Path(tempfile.mkdtemp(prefix='paper-page-budget-'))
manifest = {}
for path in [source/'main.tex', source/'abbrev.tex', source/'agent-linker.bib', *source.glob('*.sty'), *source.glob('sections/*.tex'), *source.glob('table/*'), *source.glob('appendix/*.tex'), *source.glob('figures/*.pdf')]:
    if path.is_file():
        manifest[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
(args.output/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
snapshot=root/'source'
for relative in manifest:
    dest=snapshot/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/relative,dest)

def inspect(pdf):
    doc = fitz.open(pdf)
    headings = []
    for level, title, page in doc.get_toc():
        if level != 1: continue
        # PDF outline destinations have a different coordinate convention; use visible headings.
        query = title.upper()
        if title.startswith(tuple(str(n) for n in range(1,10))):
            query = title.split(' ',1)[1].upper()
        hits = doc[page-1].search_for(query)
        # Section 3's name also occurs in captions and prose; require a heading-sized match.
        hits = [h for h in hits if h.y0 > 75]
        headings.append({'title':title, 'page':page, 'y':round(hits[0].y0,1) if hits else None})
    availability = None
    refs = next(h['page'] for h in headings if h['title']=='References')
    for i in range(refs):
        hits = doc[i].search_for('DATA AVAILABILITY STATEMENT')
        if hits:
            availability = {'page':i+1,'y':round(hits[0].y0,1)}
    body_end = None
    if availability:
        i=availability['page']-1
        lines=[]
        for block in doc[i].get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                x0,y0,x1,y1=line['bbox']
                if x0>40 and 80<y0<availability['y']-1:
                    lines.append(y1)
        if lines: body_end={'page':i+1,'y':round(max(lines),1)}
        else: body_end={'page':i,'y':None}
    overview_gap=None
    for page in doc:
        blocks=[b for b in page.get_text('blocks') if b[0]>40 and 80<b[1]<665]
        captions=[b for b in blocks if b[4].startswith('Fig. 3.')]
        if captions:
            caption=captions[0]
            following=[b for b in blocks if b[1]>caption[3]]
            if following: overview_gap=round(min(b[1] for b in following)-caption[3],1)
            break
    return {'total_pages':len(doc),'body_end':body_end,'availability':availability,'headings':headings,'overview_caption_to_text_gap_pt':overview_gap}

results={}
for variant in args.variants:
    folder=root/variant
    folder.mkdir()
    for relative in manifest:
        dest=folder/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(snapshot/relative,dest)
    main=folder/'main.tex'
    if variant=='ragged-bottom':
        main.write_text(main.read_text().replace('\\begin{document}', '\\begin{document}\n\\raggedbottom'))
    elif variant=='flexible-figures':
        for name in ['approach','motivation']:
            p=folder/'sections'/f'{name}.tex'
            text=p.read_text().replace('\\begin{figure*}[t]','\\begin{figure}[htbp]').replace('\\end{figure*}','\\end{figure}').replace('\\begin{figure}[t]','\\begin{figure}[htbp]')
            p.write_text(text)
    elif variant=='concise-text':
        p=folder/'sections/results.tex'
        text=p.read_text()
        answers=[
            r'On these five projects, \approach{} leads the compared systems on average, but not on every project. The higher scores accompany greater input-token use than \Artemis{}.',
            r'The architecture-driven metrics expose whole-component misses, weak components, and ranking differences. They also reveal weaknesses in \approach{}.',
            r'The judges raise precision while rejecting some true links. Disabling them increases recall and lowers precision.',
            r'The two routes recover complementary links. The separately run configuration without alias discovery has lower component-level scores.'
        ]
        counter=[0]
        def answer(match):
            i=counter[0];counter[0]+=1
            return '\\begin{rqanswer}\n\\paragraph{Answer to RQ'+str(i+1)+'.}\n'+answers[i]+'\n\\end{rqanswer}'
        text=re.sub(r'\\begin\{rqanswer\}.*?\\end\{rqanswer\}',answer,text,flags=re.S)
        assert counter[0]==4
        text=text[:text.index('\\subsection{Summary}')].rstrip()+'\n'
        p.write_text(text)
        p=folder/'sections/eval.tex'
        text=p.read_text()
        text,n=re.subn(r'\\emph\{\\textbf\{Motivation:\}\}.*?(?=\\begin\{enumerate\}|\\subsection\{Experiment Design\})','',text,flags=re.S)
        assert n==4,n
        p.write_text(text)
        p=folder/'sections/intro.tex'
        text=p.read_text()
        start=text.index('Three parts work together:')
        end=text.index('%3.4 results',start)
        text=text[:start]+text[end:]
        start=text.index('These approaches share one root cause:')
        end=text.index('A trace link between',start)
        text=text[:start]+text[end:]
        start=text.index('\\autoref{sec:motivation} motivates')
        end=text.index('% ===============================================================',start)
        text=text[:start]+text[end:]
        p.write_text(text)
    elif variant!='baseline':
        raise ValueError(variant)
    patch=[]
    for relative in manifest:
        if not relative.endswith('.tex'):continue
        before=(snapshot/relative).read_text().splitlines(keepends=True)
        after=(folder/relative).read_text().splitlines(keepends=True)
        patch.extend(difflib.unified_diff(before,after,fromfile='a/'+relative,tofile='b/'+relative))
    if patch:(args.output/f'{variant}.patch').write_text(''.join(patch))
    cmd=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex']
    run=subprocess.run(cmd,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (args.output/f'{variant}-build.txt').write_text('Command: '+' '.join(cmd)+'\n'+run.stdout)
    result={'exit_code':run.returncode}
    if run.returncode==0:
        result.update(inspect(folder/'main.pdf'))
        log=(folder/'main.log').read_text()
        result['undefined_warnings']=re.findall(r'^.*(?:Citation .*undefined|Reference .*undefined|undefined references|multiply defined).*$',log,re.M)
        result['overfull_boxes']=re.findall(r'^Overfull .*$',log,re.M)
    results[variant]=result
    print(variant,json.dumps(result),flush=True)
(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('Build copies:',root)
