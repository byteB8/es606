"""Check every BibTeX entry against Crossref or arXiv and print the authoritative metadata.

Reference details typed from memory drift: an author dropped, a page range off by a few, a DOI that
belongs to a neighbouring paper. This looks each entry up by DOI, by arXiv id, or failing both by
title, and prints what the registry says beside what the .bib says, so differences can be fixed
rather than guessed at.

    python tools/verify_bib.py refs.bib
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "eg606-bibcheck/1.0 (mailto:prasadbalbir1056@gmail.com)"}


def get(url, timeout=40):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def entries(text):
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict((k.lower(), " ".join(v.split()))
                      for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(3)))
        yield m.group(2).strip(), fields


def clean(s):
    return re.sub(r"[{}\\]", "", s or "")


def crossref_doi(doi):
    msg = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    return msg


def crossref_title(title):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 1})
    items = json.loads(get("https://api.crossref.org/works?" + q))["message"]["items"]
    return items[0] if items else None


def arxiv(aid):
    xml = get("http://export.arxiv.org/api/query?id_list=" + aid)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(xml).find("a:entry", ns)
    return {"title": e.findtext("a:title", "", ns).strip(),
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "published": e.findtext("a:published", "", ns)[:10]}


def show_crossref(m):
    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])]
    year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"title": " ".join(m.get("title", [""])), "authors": authors, "year": year,
            "venue": " ".join(m.get("container-title", [""])), "volume": m.get("volume"),
            "issue": m.get("issue"), "pages": m.get("page") or m.get("article-number"),
            "doi": m.get("DOI")}


def main():
    text = open(sys.argv[1]).read()
    for key, f in entries(text):
        print("=" * 90)
        print(f"{key}\n  BIB : {clean(f.get('title'))[:110]}\n        "
              f"{clean(f.get('author'))[:150]} | {f.get('year')} | "
              f"vol {f.get('volume')} no {f.get('number')} pp {f.get('pages')} | doi {f.get('doi')}")
        try:
            aid = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})", " ".join(f.values()))
            if f.get("doi"):
                got = show_crossref(crossref_doi(f["doi"]))
                src = "crossref-doi"
            elif aid:
                got = arxiv(aid.group(1))
                src = "arxiv"
            else:
                hit = crossref_title(clean(f.get("title")))
                got = show_crossref(hit) if hit else None
                src = "crossref-title"
            print(f"  {src.upper()}: {json.dumps(got, ensure_ascii=False)[:600]}")
        except Exception as e:
            print(f"  LOOKUP FAILED: {e}")
        time.sleep(1)


if __name__ == "__main__":
    main()
