"""Primary-source checks the registries could not answer.

  * NMED-H: the dataset's own README states how it should be cited.
  * Di Liberto et al. 2021: Crossref holds only the early-release record; PubMed has the issue.
  * GuessTheMusic: is there a published version, or only the preprint?
  * The imposter rule: which Accou paper actually says the imposter is taken one second after
    the matched segment? The manuscript leans on that sentence, so its citation must be the right one.
"""
import json
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "eg606-bibcheck/1.0 (mailto:prasadbalbir1056@gmail.com)"}


def get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def pdf_text(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        return subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True,
                              text=True).stdout


print("=" * 80, "\nNMED-H README: recommended citation")
readme = Path("/mnt/nas/balbir/egdta/raw/nmed_h/NMED-H_v2.3.0_README.pdf")
txt = subprocess.run(["pdftotext", "-layout", str(readme), "-"], capture_output=True, text=True).stdout
for m in re.finditer(r"(?is)(cit\w+.{0,900})", txt):
    print(m.group(1)[:900])
    break

print("=" * 80, "\nPubMed 34341154 (Di Liberto et al. 2021)")
s = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=34341154&retmode=json"))
r = s["result"]["34341154"]
print({k: r.get(k) for k in ("title", "fulljournalname", "pubdate", "volume", "issue", "pages",
                             "elocationid")}, [a["name"] for a in r.get("authors", [])])

print("=" * 80, "\nGuessTheMusic: published version?")
q = urllib.parse.urlencode({"query.bibliographic": "GuessTheMusic Song Identification from Electroencephalography response", "rows": 3})
for it in json.loads(get("https://api.crossref.org/works?" + q))["message"]["items"]:
    print(" ", " ".join(it.get("title", [""]))[:90], "|", " ".join(it.get("container-title", [""])),
          "|", (it.get("issued", {}).get("date-parts") or [[None]])[0][0], "|", it.get("DOI"),
          "|", it.get("page"))

print("=" * 80, "\nWhich Accou paper states the imposter rule?")
for aid, label in (("2105.06844", "speech intelligibility (J Neural Eng 2021)"),):
    try:
        t = pdf_text(get(f"https://arxiv.org/pdf/{aid}"))
        hits = [l.strip() for l in t.splitlines() if re.search(r"impost|mismatch", l, re.I)]
        print(f"--- arXiv {aid} {label}: {len(hits)} lines mention imposter/mismatch")
        for l in hits[:14]:
            print("    ", l[:150])
    except Exception as e:
        print(f"--- arXiv {aid}: failed {e}")
# the EUSIPCO 2020 paper: is there an arXiv copy?
q = urllib.parse.urlencode({"search_query": 'ti:"dilated convolutional" AND au:Accou', "max_results": 10})
feed = get("http://export.arxiv.org/api/query?" + q).decode()
for t, i in zip(re.findall(r"<title>(.*?)</title>", feed, re.S)[1:], re.findall(r"<id>http://arxiv.org/abs/(.*?)</id>", feed)):
    print(" arXiv candidate:", i, "|", " ".join(t.split())[:100])
