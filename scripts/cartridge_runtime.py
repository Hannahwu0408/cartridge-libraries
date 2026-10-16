"""cartridge_runtime.py — importable runtime + chain controller for the cartridge library.

WHY THIS MODULE EXISTS. The deployed retrieval/answer primitives and the R1–R3 chain controller
lived as closures inside `router.py main()` — correct, but unimportable. The LangGraph integration
(langgraph_chain.py / langgraph_agent.py) needs to call the SAME functions, and a re-implementation
would fork the logic that produced the recorded 0.71. So the closures are transplanted here
VERBATIM (single source of truth); `router.py` now builds a `CartridgeRuntime` + `ChainController`
and binds their methods where its closures used to be. Behavior is intended to be bit-identical —
gated by re-running router.py on the 2Wiki 100 questions and reproducing chain = 0.71 exactly.

Two classes:
  CartridgeRuntime  — index + frozen embedder + frozen model + cartridge cache. Primitives:
                      embed / dense / rank_ids / load_cart / answer (cartridge-backed) /
                      answer_text (same call, article TEXT in context — the doc-source switch) /
                      title_of_slug / text_of.
  ChainController   — the R1–R3 controller (kinship algebra, templated decompose, bridge validity
                      + retry, entity linking, referee'd reframe, overhop guard) exposed BOTH as
                      one-shot `chain(q_emb, q_text)` and as individual steps for graph nodes.
"""
from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np
import torch

from cartridge_qwen import (
    get_device, load_model, generate_answer, compose, QwenCartridge, RelevanceGate,
)


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ── shared constants (transplanted verbatim from router.py) ──────────────────────────────────

REFUSAL_MARKERS = ("not mentioned", "not explicitly", "no information", "does not mention", "cannot",
                   "not stated", "not provided", "not specified", "unable to", "isn't mentioned",
                   "is not available", "not found", "does not provide", "not contain")

RELWORDS = {"father", "mother", "wife", "husband", "son", "daughter", "parent", "parents",
            "spouse", "grandmother", "grandfather", "grandson", "granddaughter", "brother",
            "sister", "stepmother", "stepfather", "director", "creator", "composer", "performer",
            "author", "founder", "producer", "writer", "singer", "guitarist", "editor",
            "publisher", "screenwriter", "child", "children"}

_STOP = {"the", "of", "and", "a", "an", "in", "on", "for", "de", "la", "von", "van", "jr", "sr",
         "st", "ii", "iii", "iv", "1st", "2nd", "3rd", "lord", "lady", "king", "queen", "duke",
         "duchess", "prince", "princess", "grand", "sir", "count", "countess", "earl", "margrave"}

_ATTR_SYN = {"death": {"die", "died", "death"}, "birth": {"born", "birth", "birthday"},
             "burial": {"buried", "burial", "grave"}, "birthplace": {"born", "birthplace"},
             "study": {"study", "studied", "graduate", "graduated", "school", "college", "educated"},
             "graduate": {"graduate", "graduated", "study", "studied", "school", "college"},
             "work": {"work", "works", "worked", "working"}, "cause": {"cause", "why", "died"},
             "country": {"country", "from", "nationality", "citizen"},
             "award": {"award", "awards", "won", "win", "prize"},
             "spouse": {"spouse", "married", "wife", "husband"}, "date": {"when", "date"},
             "place": {"where", "place"}, "mother": {"mother"}, "father": {"father"}}

_AWARDISH = {"award", "awards", "prize", "role", "character"}

# R1 — deterministic kinship algebra: compound relations decompose EXACTLY (no LLM guessing).
KINSHIP = {  # phrase → (hop-1 relation, hop-2 residual relation)
    "paternal grandmother": ("father", "mother"), "paternal grandfather": ("father", "father"),
    "maternal grandmother": ("mother", "mother"), "maternal grandfather": ("mother", "father"),
    "stepmother": ("father", "wife"), "stepfather": ("mother", "husband"),
    "mother-in-law": ("spouse", "mother"), "father-in-law": ("spouse", "father"),
}


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return (not t.strip()) or any(p in t for p in REFUSAL_MARKERS)


def tok_list(s: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", (s or "").lower()):
        if len(w) > 1 and w not in _STOP and w not in out:
            out.append(w)
    return out


def relation_leaks(rq, q_text, bridge):
    """True if the free rewrite still poses a relation word from the ORIGINAL question that the
    bridge was meant to consume (and that isn't part of the bridge's own name) — the reframe bug."""
    rql, ql, bl = rq.lower(), q_text.lower(), bridge.lower()
    return any(w in ql and w in rql and w not in bl for w in RELWORDS)


def kinship_parse(q_text):
    ql = q_text.lower()
    for phrase, (parent, residual) in KINSHIP.items():
        if phrase in ql:
            return phrase, parent, residual
    return None


# ── the runtime ──────────────────────────────────────────────────────────────────────────────

class CartridgeRuntime:
    """Index + frozen embedder + frozen model + cartridge cache — the library's serving core."""

    def __init__(self, index_path, model_id="Qwen/Qwen3-4B", compose_mode="gnorm",
                 gate_path=None, gen_max_tokens=48, articles_file=None, device=None,
                 cart_device="cpu"):
        self.device = device or get_device()
        idx = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.entries = idx["cartridges"]
        self.ids = [e["id"] for e in self.entries]
        self.id2entry = {e["id"]: e for e in self.entries}
        self.emb_mat = np.asarray([e["embedding"] for e in self.entries], dtype=np.float32)

        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(idx["embedder"])   # SAME frozen embedder as the index
        self.q_instr = idx.get("query_instruction", "")

        self.articles = ({a["passage_id"]: a for a in load_jsonl(articles_file)}
                         if articles_file else {})

        self.model, self.tokenizer = load_model(model_id, self.device)
        self.compose_mode = compose_mode
        self.gate = RelevanceGate.load(gate_path, device=self.device) if gate_path else None
        self.gen_max_tokens = gen_max_tokens
        self.cart_device = cart_device
        self._cart_cache: dict[str, QwenCartridge | None] = {}

    # — retrieval —
    def embed(self, text):
        return self.embedder.encode(self.q_instr + text, normalize_embeddings=True).astype(np.float32)

    def dense(self, q_emb, k, exclude=None):
        sims = self.emb_mat @ q_emb
        order = np.argsort(-sims)
        ex = exclude or set()
        picked = [self.ids[i] for i in order if self.ids[i] not in ex]
        return picked[:k]

    def rank_ids(self, vec):
        return [self.ids[i] for i in np.argsort(-(self.emb_mat @ vec))]

    # — cartridges + generation —
    def load_cart(self, slug):
        if slug not in self._cart_cache:
            e = self.id2entry.get(slug)
            p = Path(e["tensor_path"]) if e else None
            self._cart_cache[slug] = (QwenCartridge.load(p, device=self.cart_device)
                                      if (p and p.exists()) else None)
        return self._cart_cache[slug]

    def gen(self, **kw):
        return generate_answer(self.model, self.tokenizer, device=self.device,
                               max_new_tokens=self.gen_max_tokens, conversational=True, **kw)

    @torch.no_grad()
    def answer(self, q_text: str, slugs: list[str]) -> str:
        """Answer from the slugs' CARTRIDGES (gnorm-composed if >1; plain model if none load)."""
        carts = [c for c in (self.load_cart(s) for s in slugs) if c is not None]
        if not carts:
            return self.gen(question=q_text)
        return self.gen(question=q_text, cartridge=compose(
            carts, mode=self.compose_mode,
            gate=self.gate if self.compose_mode == "gnorm" else None))

    @torch.no_grad()
    def answer_text(self, q_text: str, slugs: list[str]) -> str:
        """Same call, but read the slugs' article TEXT in context (the text doc-source)."""
        passage = self.text_of(slugs)
        if not passage:
            return self.gen(question=q_text)
        return self.gen(question=q_text, passage=passage)

    # — lookups —
    def title_of_slug(self, slug):
        e = self.id2entry.get(slug)
        return (e.get("title") or slug) if e else slug

    def text_of(self, slugs) -> str:
        return "\n\n".join(self.articles[s]["context"] for s in slugs if s in self.articles)


# ── the R1–R3 chain controller ───────────────────────────────────────────────────────────────

class ChainController:
    """Iterative SINGLE-cartridge answering (no composition → no dilution) with the round-3
    controller (R1–R4, F1–F3). Transplanted verbatim from router.py; `template=True` corresponds
    to --chain-template, `overhop_guard` to --chain-overhop-guard, `reframe` to (not --no-reframe),
    `hop1_first` to --chain-hop1-first (known-bad; kept for ablation).

    `doc_source` ("cartridge" | "text") selects what the answer/extract calls read — the SAME
    controller drives both backends, which is exactly the cartridge-vs-text comparison."""

    def __init__(self, rt: CartridgeRuntime, template=True, overhop_guard=True, reframe=True,
                 hop1_first=False, doc_source="cartridge"):
        self.rt = rt
        self.template = template
        self.overhop_guard = overhop_guard
        self.reframe = reframe
        self.hop1_first = hop1_first
        assert doc_source in ("cartridge", "text")
        self.doc_source = doc_source
        self.last_debug: dict = {}  # per-query trace (bridge string / rewritten answer_q)

        # F1 — title decomposition for entity identity. Wikipedia titles embed OTHER entities in
        # two slots: parentheticals ("(Namie Amuro song)", "(son of Louis XV)") and relation-worded
        # comma segments ("wife of Edgar"). SUBJECT toks exclude those; HEAD toks = the segment
        # before the first comma/paren (the canonical name for linking).
        self.subject_toks: dict[str, set] = {}
        self.head_toks: dict[str, set] = {}
        for s in rt.ids:
            t = re.sub(r"\(.*?\)", " ", rt.title_of_slug(s))
            parts = [p.strip() for p in t.split(",") if p.strip()]
            kept = parts[:1] + [p for p in parts[1:] if not any(w in RELWORDS for w in tok_list(p))]
            self.subject_toks[s] = set(tok_list(" ".join(kept)))
            self.head_toks[s] = set(tok_list(parts[0])) if parts else set()

    # — answering primitive routed through the selected doc source —
    def answer(self, q_text, slugs):
        if self.doc_source == "text":
            return self.rt.answer_text(q_text, slugs)
        return self.rt.answer(q_text, slugs)

    # — R3/F1: entity linking —
    def link_bridge(self, bridge):
        """Entity-link the decoded bridge against index title HEADs. Returns (slug|None, hits):
        a slug only on a UNIQUE hit (head == bridge tokens, or head ⊆ bridge to absorb honorifics
        like 'King Charles II of England'); ambiguous/no match → (None, hits) — no decision."""
        bt = set(tok_list(bridge))
        if not bt:
            return None, []
        hits = [s for s, h in self.head_toks.items() if h and (h == bt or h <= bt)]
        return (hits[0] if len(hits) == 1 else None), hits

    # — R2/F1: bridge validity —
    def bridge_valid(self, bridge, h1_slug, h1_title):
        b = (bridge or "").strip()
        if not b or len(b.split()) > 12:                      # 12: noble titles run long
            return False
        bt = tok_list(b)
        if not bt or any(t in RELWORDS for t in bt):          # 'husband', "X's father", 'director' —
            return False                                       # a proper entity NAME has no relation word
        slug_link, hits = self.link_bridge(b)
        if slug_link is not None:                             # confident identity → echo iff hop-1 itself
            return slug_link != h1_slug
        subj = self.subject_toks.get(h1_slug) or set(tok_list(h1_title))
        if subj and set(bt) <= subj:                          # fallback: echo of hop-1's own subject
            return False
        return True

    # — R3: free-rewrite referee —
    def rewrite_ok(self, rq, q_text, bridge, attr, work=""):
        rql = (rq or "").lower()
        if not rql.strip():
            return False
        bt = tok_list(bridge)
        if bt and not any(t in rql for t in bt[:2]):          # (1) rewrite ignored the bridge
            return False
        if relation_leaks(rq, q_text, bridge):                # (2) "…'s father" leak
            return False
        rqt = set(tok_list(rq))
        at = tok_list(attr)
        if at and not any(t in rqt or (set(_ATTR_SYN.get(t, ())) & rqt) for t in at):
            return False                                      # (3) dropped what was being asked
        if work and at and (set(at) & _AWARDISH):             # (4) work-contextual attribute must keep
            if not (set(tok_list(work)) & set(tok_list(rq))):  # the work ("award won FOR <film>")
                return False
        return True

    def work_qualifier(self, q_text, hop1_title):
        """If the question's subject is a WORK (film/song/…) and we template an award-type attribute,
        the qualifier must survive ('Which award did X win FOR <work>?') — else the cartridge names
        some other award. Returns the work title or ''."""
        ql = q_text.lower()
        if any(m in ql for m in ("film ", "movie ", "song ", "album ", "series", "show ")):
            if set(tok_list(hop1_title)) & set(tok_list(q_text)):
                return hop1_title
        return ""

    # — decomposition calls —
    def bridge_extract(self, q_text, slugs):
        """Ask hop-1 cartridge(s) for ONLY the intermediate entity to look up next — not the
        final answer — so the model doesn't hallucinate/commit early, and hop-2 gets a clean
        query. Returns a short entity string (first line)."""
        instr = (f"Question: {q_text}\n\nThis question cannot be fully answered from the context "
                 f"alone — it refers to another entity you must look up next. Identify the single "
                 f"bridge entity named in the context that links to the answer (the specific "
                 f"person, organization, or work the question is really asking about). Do NOT "
                 f"answer the question itself, and do not name a location mentioned only in "
                 f"passing. Reply with only that entity's name.")
        return self.answer(instr, slugs)

    def bridge_and_attribute(self, q_text, slugs, extra_note=""):
        """Fix A: one hop-1 call returns BOTH the intermediate entity AND the residual attribute the
        question ultimately wants about it — so hop-2 can be built as a clean '<attribute> of
        <bridge>' template instead of a free-form rewrite that tends to keep the original relation
        word. Returns (bridge, attribute). `extra_note` lets the R2 retry forbid a rejected bridge."""
        instr = (f"Question: {q_text}\n\nThe context names an intermediate entity that this question "
                 f"refers to indirectly (e.g. 'the father of X', 'the director of Y', \"X's wife\"). "
                 f"Resolve it, then say what the question ultimately asks ABOUT that entity. Output "
                 f"EXACTLY two lines:\nBRIDGE: <the single intermediate entity's name>\n"
                 f"ATTRIBUTE: <the property the question wants about that entity — a short noun phrase, "
                 f"e.g. place of death, birthplace, country of citizenship, spouse, date of death, "
                 f"where they studied. If the question is itself relational (e.g. grandmother = the "
                 f"parent's mother), give the RESIDUAL relation about the bridge entity, e.g. 'mother'.>"
                 + (f"\n{extra_note}" if extra_note else ""))
        out = self.answer(instr, slugs) or ""
        bridge = attr = ""
        for line in out.splitlines():
            s = line.strip()
            key = s.split(":", 1)[0].strip().lower() if ":" in s else ""
            # prefix-match the labels: the model sometimes garbles them ("BRIEVE:") — R3a
            if key[:3] == "bri" and not bridge:
                bridge = s.split(":", 1)[1].strip().strip('"')
            elif key[:3] == "att" and not attr:
                attr = s.split(":", 1)[1].strip().strip('"')
        if not bridge and out.strip():                        # parse fell through → salvage 1st line,
            first = out.strip().splitlines()[0].strip()       # stripping any leftover "label:" prefix
            bridge = first.split(":", 1)[1].strip().strip('"') if ":" in first[:12] else first
        return bridge, attr

    def reframe_question(self, q_text, bridge):
        """Rewrite the multi-hop question into a DIRECT single-hop question about the bridge entity.
        Pure text rewrite (no cartridge/passage). Returns the rewritten question."""
        instr = (f'Rewrite the question below so it asks DIRECTLY about "{bridge}". Replace the phrase '
                 f'describing the entity (e.g. "the father of X", "the director of Y", "X\'s wife") '
                 f'with "{bridge}", but KEEP asking for the EXACT SAME thing — the same '
                 f'who/what/where/when/why/how and the same attribute (e.g. cause of death, date of '
                 f'death, birthplace, nationality, where they worked/studied). Change ONLY the entity, '
                 f'never what is being asked.\n\nOriginal: "{q_text}"\n\nReply with ONLY the rewritten '
                 f'question.')
        r = self.rt.answer(instr, [])              # [] → no cartridge, frozen model rewrites
        r = r.strip().splitlines()[0].strip().strip('"') if r.strip() else ""
        return r or q_text

    # — the full controller (verbatim port of router.py chain()) —
    def chain(self, q_emb, q_text):
        """hop-1: retrieve the subject article; decompose. hop-2: entity-link or dense-retrieve the
        BRIDGE ENTITY's own article; build the hop-2 question; answer from that source ALONE.
        Returns the slug list to answer from; self.last_debug carries 'bridge' and (if set)
        'answer_q' — callers answer last_debug.get('answer_q', q_text) on the returned slugs."""
        rt = self.rt
        h1 = rt.dense(q_emb, 1)

        if self.hop1_first:                                   # Fix C — known-bad, kept for ablation
            a1 = self.answer(q_text, h1)
            if not is_refusal(a1):
                self.last_debug["bridge"] = "(hop1-direct)"
                return h1

        if self.template:                                     # round-3 controller (R1–R4, F1–F3)
            h1_slug = h1[0] if h1 else ""
            h1_title = rt.title_of_slug(h1_slug) if h1 else ""

            # R1 — deterministic kinship algebra
            kin = kinship_parse(q_text)
            if kin:
                phrase, parent, residual = kin
                # F2 — kinship over-hop: hop-1's article is NOT the question's subject → retrieval
                # already landed on the parent → ask the residual question THERE directly.
                ht_, qt_ = set(tok_list(h1_title)), set(tok_list(q_text))
                if ht_ and len(ht_ & qt_) / len(ht_) < 0.6:
                    self.last_debug["bridge"] = f"(kinship-redirect: {h1_title})"
                    self.last_debug["answer_q"] = f"Who is the {residual} of {h1_title}?"
                    return h1
                parent_q = re.sub(re.escape(phrase), parent, q_text, flags=re.IGNORECASE)
                raw = self.answer(parent_q + " Reply with only the name.", h1) or ""
                bridge = (raw.strip().splitlines()[0].strip().strip('"').rstrip(".")
                          if raw.strip() else "")
                attr = residual
            else:
                bridge, attr = self.bridge_and_attribute(q_text, h1)

            # R2 — validity gate + one retry forbidding the rejected (echo/role-word) bridge.
            if not self.bridge_valid(bridge, h1_slug, h1_title):
                note = (f"Note: the intermediate entity is NOT \"{h1_title}\" and is not a bare role "
                        f"word — name the OTHER specific person, organization, or work.")
                bridge, attr2 = self.bridge_and_attribute(q_text, h1, extra_note=note)
                attr = attr or attr2
            self.last_debug["bridge"] = bridge
            if not self.bridge_valid(bridge, h1_slug, h1_title):
                return h1                                     # unusable bridge → original Q on hop-1

            # F1 bonus — a uniquely-linked bridge goes DIRECTLY to its article (exact hop-2);
            # unlinked → dense retrieval on the embedded BRIDGE-ENTITY string (never the rewrite).
            slug_link, _hits = self.link_bridge(bridge)
            if slug_link and slug_link != h1_slug:
                h2 = [slug_link]
            else:
                h2 = rt.dense(rt.embed(bridge), 1, exclude=set(h1))
            if not h2:
                return h1
            if self.reframe:
                if kin:                                       # kinship: template IS the right question
                    rq = f"Who is the {attr} of {bridge}?"
                else:
                    work = self.work_qualifier(q_text, h1_title)
                    rq = self.reframe_question(q_text, bridge)
                    if not self.rewrite_ok(rq, q_text, bridge, attr, work):  # R3 — referee
                        at = set(tok_list(attr))
                        if attr and (at & _AWARDISH) and work:
                            rq = f"Which {attr} did {bridge} win for {work}?"
                        elif attr:
                            wh = "Who" if at & RELWORDS else "What"
                            rq = f"{wh} is the {attr} of {bridge}?"
                        elif relation_leaks(rq, q_text, bridge):
                            for w in RELWORDS:
                                rq = re.sub(rf"(?i)\b{re.escape(bridge)}[’']s {w}\b", bridge, rq)
                self.last_debug["answer_q"] = rq

                # R4 — narrow over-hop guard: when hop-1's article is NOT the question's subject,
                # the retriever likely already jumped to the answer article; if hop-2 then refuses,
                # fall back to answering the ORIGINAL question on hop-1.
                if self.overhop_guard and h1_title:
                    ht, qt = set(tok_list(h1_title)), set(tok_list(q_text))
                    if ht and len(ht & qt) / len(ht) < 0.6:
                        a2 = self.answer(rq, h2)
                        if is_refusal(a2):
                            self.last_debug.pop("answer_q", None)
                            self.last_debug["bridge"] = f"(overhop-guard: {bridge})"
                            return h1
            return h2

        bridge = self.bridge_extract(q_text, h1)              # original (raw-chain) behavior
        self.last_debug["bridge"] = bridge
        if bridge.strip():
            h2 = rt.dense(rt.embed(bridge), 1, exclude=set(h1))
            if h2:
                if self.reframe:
                    self.last_debug["answer_q"] = self.reframe_question(q_text, bridge)
                return h2                 # answer from the bridge-target cartridge ALONE
        return h1

    def run(self, q_text):
        """One-call convenience for graph/agent callers: full chain → final answer string."""
        self.last_debug.clear()
        slugs = self.chain(self.rt.embed(q_text), q_text)
        return self.answer(self.last_debug.get("answer_q", q_text), slugs), slugs
