"""
Stage 2 — natural-error harvest (O-2 / D-008) from a DISJOINT model.

Swap-based false generation left a persistent hidden-state edit signature that
the edit canary detects (v1-v3, ~0.67). Harvesting a disjoint model's GENUINE
confident errors removes that signature by construction: the false items are
real model outputs, not edits of true statements. The harvest model
(Mistral-7B-Instruct) is disjoint from every probed substrate (Llama/Qwen/
Gemma/Pythia), so its error style cannot leak into a probe's train/test on those.

Method (cities shown; analogous per structured topic): ask the disjoint model
the factual question behind each true (subject, object) pair; keep answers that
are (a) a VALID same-type object (in the KB object pool) and (b) WRONG (not the
true object). Those are fluent, confident, natural falsehoods. Typicality (FT
vs FA) is then sorted by the D-007 frequency axis, same as everything else.

Deterministic (greedy) so the harvest is reproducible; cached to the volume.
"""
from __future__ import annotations

import re

from . import corpus_gen as cg

# per-topic factual question + answer-slot (object type)
HARVEST_Q = {
    "cities":     ("Which country is the city {s} in? Reply with ONLY the country name.",
                   "{s} is a city in {o}."),
    "generated":  ("In which country or region is {s} located? Reply with ONLY the place name.",
                   "{s} is located in {o}."),
    "inventions": ("Who invented the {s}? Reply with ONLY the inventor's full name.",
                   "{s} invented the {o}."),  # note: template differs; handled below
}


def _greedy(substrate, prompt: str, max_new_tokens: int = 16) -> str:
    import torch
    tok = substrate.tokenizer
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    enc = tok(text, return_tensors="pt").to(substrate.model.device)
    with torch.no_grad():
        out = substrate.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _clean(ans: str) -> str:
    ans = ans.strip().strip('."\'').split("\n")[0].strip()
    ans = re.sub(r"^(the|it is|it's|located in|in)\s+", "", ans, flags=re.I).strip()
    return ans.rstrip(".")


def harvest_topic(substrate, topic: str, max_subjects: int = 400,
                  seed: int = 20260719) -> dict:
    """Return {items: [natural false items], stats}. Only 'cities'/'generated'
    use the (subject -> object) question form; objects validated against the KB
    object pool so harvested falsehoods are same-type, fluent, and genuinely wrong."""
    import numpy as np
    if topic not in ("cities", "generated"):
        return {"items": [], "stats": {"note": f"{topic} not harvested in v1 O-2"}}

    q_tpl, stmt_tpl = HARVEST_Q[topic]
    pairs = cg._parse_true_pairs(topic)
    true_obj: dict = {}
    for s, o in pairs:
        true_obj.setdefault(s, set()).add(o)
    obj_pool = {o.lower(): o for _, o in pairs}   # canonical-case lookup

    rng = np.random.default_rng(seed)
    subjects = sorted(true_obj)
    rng.shuffle(subjects)
    subjects = subjects[:max_subjects]

    items, n_valid_obj, n_wrong, n_asked = [], 0, 0, 0
    for s in subjects:
        n_asked += 1
        ans = _clean(_greedy(substrate, q_tpl.format(s=s)))
        canon = obj_pool.get(ans.lower())
        if canon is None:
            continue                       # not a same-type object we can KB-check
        n_valid_obj += 1
        if canon in true_obj.get(s, set()):
            continue                       # model was right -> not a false item
        n_wrong += 1
        items.append({
            "text": stmt_tpl.format(s=s, o=canon),
            "truth": False, "edited": False,          # NATURAL output, no edit
            "entity": s, "entities": [s, canon], "domain": topic,
            "provenance": {"source": "harvest_mistral", "strategy": "natural_confident_error",
                           "verified_against": "azaria_mitchell_kb"},
        })
        if n_asked % 50 == 0:
            print(f"    [harvest {topic}] {n_asked}/{len(subjects)} asked, "
                  f"{n_wrong} genuine errors", flush=True)
    return {"items": items,
            "stats": {"topic": topic, "asked": n_asked, "valid_object": n_valid_obj,
                      "genuine_errors": n_wrong, "error_rate": round(n_wrong / max(n_asked, 1), 3)}}


def load_harvested(topic: str) -> list[dict]:
    """Harvested natural-error items previously written by the `harvest` stage."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "corpus" / f"harvest_{topic}.json"
    if not p.exists():
        raise FileNotFoundError(f"no harvest for '{topic}' at {p}; run --stage harvest first")
    items = json.loads(p.read_text(encoding="utf-8"))
    for it in items:
        it.setdefault("edited", False)
    return items


# ── External verification of harvested errors (2026-08-12) ───────────────────
# The A&M knowledge base is single-valued per subject and incomplete, so it calls
# "Barcelona is a city in Spain" FALSE (its Barcelona is the Venezuelan one) and
# "Jamestown is a city in United States" FALSE (its Jamestown is on Saint Helena).
# Harvesting against it alone would fill the false cells with statements that are
# actually true, which is exactly the self-manufactured confound this project
# exists to catch. Every candidate error is checked against Wikidata before it is
# allowed into a corpus.

_WD_ENDPOINT = "https://query.wikidata.org/sparql"
_WD_UA = "MIRAGE-research/1.0 (academic corpus validation; contact via repo)"
# Wikidata's country labels are often the formal name. Missing "People's Republic
# of China" alone marked every correct "China" answer as an error, and China is the
# second most common country in the gazetteer (296 of 4042 city names), so the
# false cells would have been mostly correct statements. Each entry below is a
# label-variant group, never two distinct countries: the two Congos stay separate.
_COUNTRY_ALIASES = {
    "united states": {"united states of america", "usa", "us", "america"},
    "united kingdom": {"uk", "great britain", "england", "scotland", "wales",
                       "northern ireland",
                       "united kingdom of great britain and northern ireland"},
    "china": {"people's republic of china", "prc", "mainland china"},
    "taiwan": {"republic of china", "chinese taipei"},
    "russia": {"russian federation"},
    "south korea": {"republic of korea", "korea, south"},
    "north korea": {"democratic people's republic of korea", "dprk", "korea, north"},
    "czech republic": {"czechia"}, "myanmar": {"burma", "union of myanmar"},
    "netherlands": {"kingdom of the netherlands", "holland"},
    "swaziland": {"eswatini", "kingdom of eswatini"},
    "ivory coast": {"cote d'ivoire", "côte d'ivoire"},
    "east timor": {"timor-leste", "timor leste"},
    "cape verde": {"cabo verde"},
    "democratic republic of the congo": {"dr congo", "drc", "zaire",
                                         "congo-kinshasa"},
    "republic of the congo": {"congo-brazzaville", "congo republic"},
    "uae": {"united arab emirates"}, "vatican city": {"holy see", "vatican"},
    "iran": {"islamic republic of iran"}, "syria": {"syrian arab republic"},
    "egypt": {"arab republic of egypt"}, "tanzania": {"united republic of tanzania"},
    "venezuela": {"bolivarian republic of venezuela"},
    "bolivia": {"plurinational state of bolivia"},
    "laos": {"lao people's democratic republic"},
    "vietnam": {"socialist republic of vietnam", "viet nam"},
    "moldova": {"republic of moldova"}, "macedonia": {"north macedonia"},
    "ireland": {"republic of ireland"},
}


def _norm_country(s: str) -> str:
    import re
    return re.sub(r"[^a-z ]", " ", (s or "").lower()).strip()


def _country_matches(a: str, b: str) -> bool:
    na, nb = _norm_country(a), _norm_country(b)
    if na == nb:
        return True
    for canon, alts in _COUNTRY_ALIASES.items():
        group = {canon} | {_norm_country(x) for x in alts}
        if na in group and nb in group:
            return True
    return False


def wikidata_city_countries(names: list[str], cache_path=None, batch: int = 40,
                            timeout: int = 60) -> dict[str, list[str]]:
    """{city name -> countries Wikidata says host a city of that name}. Cached."""
    import json
    import time
    from pathlib import Path

    import requests

    cache_path = Path(cache_path or (Path(__file__).resolve().parent.parent
                                     / "data" / "corpus" / "wikidata_cities.json"))
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [n for n in dict.fromkeys(names) if n not in cache]
    print(f"[wikidata] {len(cache)} cached, {len(todo)} to query", flush=True)

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        values = " ".join('"%s"@en' % n.replace('"', '\\"') for n in chunk)
        # No settlement-type filter. Requiring P31/P279* wd:Q486972 produced FALSE
        # NEGATIVES — Granada and Valencia in Spain were both missed, so genuinely
        # true statements would have been admitted into the false cells. Matching
        # any entity with this label that has a country over-rejects instead: a
        # real error is occasionally discarded, but a true statement is never
        # called false. That is the safe direction for this corpus.
        q = ("SELECT ?name ?countryLabel WHERE { VALUES ?name { %s } "
             "?city rdfs:label ?name . ?city wdt:P17 ?country . "
             'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }' % values)
        try:
            r = requests.get(_WD_ENDPOINT, params={"query": q, "format": "json"},
                             headers={"User-Agent": _WD_UA}, timeout=timeout)
            if r.status_code != 200:
                print(f"[wikidata] HTTP {r.status_code} on batch {i // batch}", flush=True)
                continue
            for n in chunk:
                cache.setdefault(n, [])
            for b in r.json()["results"]["bindings"]:
                nm, cy = b["name"]["value"], b["countryLabel"]["value"]
                if cy not in cache.setdefault(nm, []):
                    cache[nm].append(cy)
        except Exception as exc:
            print(f"[wikidata] batch {i // batch} failed: {type(exc).__name__}", flush=True)
            continue
        cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        time.sleep(1.0)
    cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return cache


def verify_harvested_errors(items: list[dict]) -> dict:
    """Drop candidate errors that Wikidata says are actually true.

    Returns {kept, rejected, unverifiable, stats}. `unverifiable` are subjects
    Wikidata returned nothing for; they are DROPPED rather than trusted, because
    an absent gazetteer entry is not evidence the statement is false.
    """
    names = [it["entities"][0] for it in items]
    gaz = wikidata_city_countries(names)
    kept, rejected, unverifiable = [], [], []
    for it in items:
        subj, obj = it["entities"][0], it["entities"][1]
        hosts = gaz.get(subj) or []
        if not hosts:
            unverifiable.append(it)
        elif any(_country_matches(obj, h) for h in hosts):
            rejected.append(it)
        else:
            kept.append(it)
    return {"kept": kept, "rejected": rejected, "unverifiable": unverifiable,
            "stats": {"n_in": len(items), "n_kept": len(kept),
                      "n_rejected_actually_true": len(rejected),
                      "n_unverifiable_dropped": len(unverifiable)}}


def harvest_wikidata(substrate, max_subjects: int = 3000, batch_size: int = 32,
                     seed: int = 20260812, commit_fn=None) -> dict:
    """Harvest natural errors against a Wikidata-sourced, MULTI-VALUED gold set.

    Replaces the A&M knowledge base for this purpose. That KB has 1298 unique city
    subjects and one country each, so it cannot adjudicate any name that exists in
    several countries: it calls "Barcelona is a city in Spain" false because its
    Barcelona is the Venezuelan one. Measured on a 1600 subject harvest, 46% of the
    candidate errors it produced were statements that are actually true. A corpus
    built on those would fill its false cells with truths, which is precisely the
    self-manufactured confound this project exists to catch.

    Here a statement counts as an error only if the model's answer is in none of
    the countries Wikidata lists for that city name.
    """
    import json
    from pathlib import Path

    import numpy as np
    import torch

    gold_path = Path(__file__).resolve().parent.parent / "data" / "corpus" / "wikidata_gold_cities.json"
    gold = {k: set(v) for k, v in json.loads(gold_path.read_text(encoding="utf-8")).items()}

    rng = np.random.default_rng(seed)
    subjects = sorted(gold)
    rng.shuffle(subjects)
    subjects = subjects[:max_subjects]

    tok = substrate.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def prompt(s):
        q = (f"Which country is the city of {s} in? "
             "Reply with ONLY the country name.")
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template([{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True)
        return q + "\nAnswer:"

    errors, correct, unparsed = [], 0, 0
    for i in range(0, len(subjects), batch_size):
        chunk = subjects[i:i + batch_size]
        enc = tok([prompt(s) for s in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=not getattr(tok, "chat_template", None)
                  ).to(substrate.model.device)
        with torch.no_grad():
            out = substrate.model.generate(**enc, do_sample=False, max_new_tokens=10,
                                           pad_token_id=tok.eos_token_id)
        cut = enc["input_ids"].shape[1]
        for s, row in zip(chunk, out):
            ans = _clean(tok.decode(row[cut:], skip_special_tokens=True))
            if not ans:
                unparsed += 1
                continue
            hosts = gold.get(s, set())
            if any(_country_matches(ans, h) for h in hosts):
                correct += 1
            else:
                errors.append({
                    "text": f"{s} is a city in {ans}.", "truth": False, "edited": False,
                    "entity": s, "entities": [s, ans], "domain": "cities",
                    "provenance": {"source": f"harvest_{substrate.model_id.split('/')[-1]}",
                                   "strategy": "natural_confident_error",
                                   "verified_against": "wikidata_multivalued_gold",
                                   "gold_countries": sorted(hosts)}})
        if commit_fn and (i // batch_size) % 10 == 0:
            commit_fn()
        if (i // batch_size) % 10 == 0:
            print(f"    [harvest] {i + len(chunk)}/{len(subjects)} "
                  f"errors={len(errors)} correct={correct}", flush=True)

    n = correct + len(errors)
    return {"items": errors,
            "stats": {"n_subjects": len(subjects), "n_answered": n,
                      "n_correct": correct, "n_errors": len(errors),
                      "n_unparsed": unparsed,
                      "error_rate": round(len(errors) / max(n, 1), 4),
                      "gold": "wikidata_multivalued", "model": substrate.model_id}}


def canonical_country_map(gold: dict) -> dict:
    """Surface form -> one canonical country name, shared by truths and errors.

    Without this the corpus leaks truth through vocabulary. The gazetteer labels
    China "People's Republic of China" while a model answers "China", so every
    true China statement and every China error were written differently and the
    single token "China" predicted falsehood perfectly. The composition gate
    caught it; this removes the cause. The canonical form is the SHORT one a model
    would naturally produce, so truths are not phrased unnaturally either.
    """
    canon = {}
    for group_canon, alts in _COUNTRY_ALIASES.items():
        for name in {group_canon} | set(alts):
            canon[_norm_country(name)] = group_canon.title()
    for names in gold.values():
        for n in names:
            k = _norm_country(n)
            canon.setdefault(k, n)
    return canon


def canonicalize(name: str, canon: dict) -> str | None:
    """Canonical country name, or None if this is not a country we recognise."""
    return canon.get(_norm_country(name))
