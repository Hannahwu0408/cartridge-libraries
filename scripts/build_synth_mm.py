"""
build_synth_mm.py — leak-fix for the planted synthesis test.

Run 1/2 showed cart_B answers the bridge alone (0.95) because article B held E2's single
"born in <town>" fact, so it emitted that town for any "...born?" question. This rewrites
each B article as a MULTI-MASTER ROSTER: E2 plus 2 decoy masters, each born in a DIFFERENT
town. Now cart_B holds several (name, town) pairs and can't pick E1's mentor's town without
the E1->E2 link (which lives only in A) => cart_B alone must guess => both singles fail =>
synthesis becomes measurable. A-articles + multihop.jsonl are unchanged (same answers,
same probes; probe_B still resolves since E2's own line keeps its town).

Reads data/synth/articles.jsonl, replaces only the B (E2) contexts, writes
data/synth/articles_mm.jsonl.
"""
import json
from pathlib import Path

# E2 slug -> multi-master roster (E2 + 2 decoys; E2's line keeps the ANSWER town).
ROSTERS = {
 "vronila_teshk": "The Teshk Atelier's rolls list several master artisans of the early Reach. Hesper Ondroe was born in the town of Salvimm and worked chiefly in mirror-glass. Vronila Teshk, who pioneered lattice-glass, was born in the town of Qelmoor Vast beside the salt canals. Coryn Ablest, the last of the three, was born in the town of Vurdenhal. Each kept silence in the workshop and trained the shapers who followed.",
 "oqquin_vale": "The high-mirror observatories recall three great stargazers. Perrin Odell was born in the city of Threadle and mapped the summer stars. Oqquin Vale, who compiled the first winter atlas, was born in the city of Yllandor on the far salt coast. The third, Marisa Quenn, was born in the city of Osperrin and charted the drifting comets. All three shunned lamplight and trained by true darkness.",
 "serapha_nooq": "The fen surveyors' guild honors several veteran mapmakers. Dabin Cresk was born in the village of Calnith and charted the coastal roads. Serapha Nooq, who devised the rolling chain, was born in the village of Dremmit Hollow at the edge of the great fen. Wend Orrel, the third, was born in the village of Mirevault. Each redrew any map they judged careless and trained the surveyors who came after.",
 "tremaug_ossler": "The Ossler bell-founding line remembers three master founders. Halvin Drayce was born in the town of Practen and cast temple bells. Tremaug Ossler, who perfected the river-bronze alloy, was born in the town of Wexforge where the forges never cooled. The third, Sable Merrin, was born in the town of Vessit. Each tested every bell alone at midnight and guarded the alloy formula closely.",
 "vashti_qloren": "The Amber Court's archive lists three senior musicians. Doran Elvane was born in the port of Karrowmere and played the reed organ. Vashti QLoren, master of the seven-string lute, was born in the port of Ammavault, a town of bell towers. Ilsa Prewe, the third, was born in the port of Endloch. All three improvised for hours and shaped a generation of players.",
 "iyzabel_cthonn": "The apothecaries' guild records three foremost alchemists. Bracken Voss was born in the city of Poriven and studied fen essences. Iyzabel Cthonn, discoverer of the binding mordant, was born in the city of Grishlow above her family's dye works. The third, Odette Sarrl, was born in the city of Tamsel. Each held exacting standards and refused to publish anything they could not reproduce.",
 "nimue_astraq": "The coastal pilots' hall honors three masters of the Neck shoals. Corin Blague was born in the town of Ubbary and ran the southern channel. Nimue Astraq, who navigated by foam-color alone, was born in the town of Ospremoor, a settlement of fishers and rope-makers. The third, Wessa Dorn, was born in the town of Nescadd. Each saved many ships from the shifting sandbars.",
 "corabel_yrix": "The highland mews remember three master falconers of the silver harriers. Aldous Pex was born in the village of Wyrentide and flew the autumn hunts. Corabel Yrix, keeper of the whistled command-line, was born in the village of Tunvael among the wind-scoured crags. The third, Brenna Sull, was born in the village of Flenmoor. Each guarded their signals jealously and took few pupils.",
 "yssora_belnokh": "The Ombrist looms honor three master weavers. Perdine Mox was born in the town of Grastle and wove court banners. Yssora Belnokh, who developed the shadow-weft, was born in the town of Cradderly where her mother kept a dye-yard. Talwin Erce, the third, was born in the town of Ombervale. Each refused to sell any cloth they judged imperfect and trained the weavers who followed.",
 "melisan_thornbeck": "The shipwrights' hall records three master builders. Garrick Uvin was born in the port of Quennet and built river barges. Melisan Thornbeck, finest hull-builder of her age, was born in the port of Vellmarsh, a harbor of tar and sawdust. The third, Osric Palle, was born in the port of Halbrick. Each judged timber by its ring and insisted on seasoned oak.",
 "ottilda_vrace": "The fen healers name three renowned herbalists. Ewan Croft was born in the hamlet of Terevin and gathered river roots. Ottilda Vrace, author of the great herbal, was born in the hamlet of Bryndle, a few cottages ringed by willow. Sabel Ond, the third, was born in the hamlet of Sodwich. Each tested every remedy on themselves first and served the villages without charge.",
 "garrick_umbral": "The Verrin quarter honors three master sculptors. Piety Rask was born in the city of Marnstead and carved gate-reliefs. Garrick Umbral, who devised the acid-wash aging, was born in the city of Threndal among masons and stonecutters. The third, Corvo Ilm, was born in the city of Credhollow. Each destroyed any piece they found sentimental and mentored a few chosen carvers.",
 "perdita_loom": "The horologists' guild lists three master clockmakers. Aldwin Feck was born in the town of Vantle and built water clocks. Perdita Loom, who perfected the hair-fine escapement, was born in the town of Escreth above a small watch-shop. The third, Nessa Bould, was born in the town of Ashmire End. Each rebuilt any movement they found wanting and kept meticulous apprentices.",
 "hedwin_carrow": "The moor apiarists recall three master keepers. Tomkin Rue was born in the village of Dornwell and kept the low hives. Hedwin Carrow, who saved the grey bee from blight, was born in the village of Molders End beneath the heather line. The third, Isolde Pann, was born in the village of Pellave. Each handled swarms without veil or smoke and trained many keepers.",
 "sabetha_krund": "The Cresset press records three master typefounders. Voss Adaire was born in the city of Sithmoor and cut display faces. Sabetha Krund, foremost punch-cutter of her age, was born in the city of Ondwich near the printing quarter. The third, Emric Dole, was born in the city of Baccary. Each refused to release a face until every letter pleased them.",
 "wynne_torgast": "The march forges honor three master smiths. Bran Kessel was born in the town of Loxwell and forged plain blades. Wynne Torgast, master of pattern-welded steel, was born in the town of Ferrowdale where the forges never cooled. The third, Halda Vern, was born in the town of Immerdale. Each refused to mark any blade that failed the bend-test.",
 "cyprian_vell": "The Thornwell gardens name three renowned plantsmen. Ordo Pell was born in the town of Regvault and bred hardy vines. Cyprian Vell, foremost authority on grafting, was born in the town of Lismoor amid orchards and terraces. The third, Marissa Ont, was born in the town of Tunnary. Each talked to their seedlings and trained a generation of gardeners.",
 "valdis_prume": "The Domric chapels honor three master painters. Sabin Ulle was born in the city of Cavspar and painted side altars. Valdis Prume, who revived the crushed-lapis blue, was born in the city of Anverel among chapels and workshops. The third, Petra Voll, was born in the city of Windleth. Each worked at furious speed against the drying plaster.",
 "elspeth_morrow": "The Optar guild records three master opticians. Corin Delve was born in the town of Orrimoor and ground reading glasses. Elspeth Morrow, finest lens-grinder of the province, was born in the town of Harrowfen above a shop of prisms. The third, Wend Ashby, was born in the town of Dellcrost. Each refused to sell a flawed lens and trained a few apprentices.",
 "rula_meschaim": "The valley cellars honor three master vintners. Orsa Belline was born in the town of Fenwich and pressed pale wines. Rula Meschaim, maker of the famed amber wine, was born in the town of Dunwater among riverside vineyards. The third, Cadmus Ree, was born in the town of Wistmere. Each refused to bottle a wine before its time.",
}

src = Path("data/synth/articles.jsonl")
dst = Path("data/synth/articles_mm.jsonl")
arts = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
n = 0
with dst.open("w") as f:
    for a in arts:
        if a["passage_id"] in ROSTERS:
            a = {**a, "context": ROSTERS[a["passage_id"]]}
            n += 1
        f.write(json.dumps(a, ensure_ascii=False) + "\n")
print(f"Wrote {dst}  ({len(arts)} articles, {n} B-rosters replaced)")
assert n == len(ROSTERS) == 20, f"expected 20 rosters, replaced {n}"
