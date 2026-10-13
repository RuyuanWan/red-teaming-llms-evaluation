"""Consolidate experiment data into a release folder and build a self-contained dashboard.

Usage: python build_release.py --out release_out \
  --gcg attack_eval_item_level_GCG.csv [--rd attack_eval_item_level_Refusal_Direction.csv] \
  --binary-rerun rerun_binary_no_def.jsonl --human human_labels.csv \
  [--completions file1.json file2.json ...] [--lit literature_review.csv]
Completion JSON files are matched by name: 'gcg_' prefix -> GCG, otherwise refusal direction;
'Llama-2'/'llama_2' -> Llama-2-7B-chat, 'Llama-3'/'llama_3' -> Llama-3-8B-Instruct.
"""
import argparse, json, re, os
import numpy as np, pandas as pd

EVALS = [("Lexical_Match", "lex", "Lexical match"),
         ("LlamaGuard3", "lg3", "Llama Guard 3"),
         ("HarmBench_Cls", "hb", "HarmBench classifier"),
         ("OpenAI_Moderation", "oai", "OpenAI Moderation"),
         ("LLM_Judge_scale_no_def", "snd", "Judge: scale, no criteria"),
         ("LLM_Judge_binary_no_def", "bnd", "Judge: binary, no criteria"),
         ("LLM_Judge_scale_with_def", "swd", "Judge: scale, criteria"),
         ("LLM_Judge_binary_with_def", "bwd", "Judge: binary, criteria")]
KEY = {e: k for e, k, _ in EVALS}
ATT = {"GCG": "GCG", "Refusal Direction": "RD"}
MOD = {"Llama-2-7B-chat": "L2", "Llama-3-8B-Instruct": "L3"}

def model_from_name(s):
    s = s.lower().replace("_", "-")
    return "Llama-2-7B-chat" if "llama-2" in s else "Llama-3-8B-Instruct"

def load_item(path, attack):
    d = pd.read_csv(path)
    d = d[d.evaluator.isin(KEY)].copy()
    d["attack"] = attack
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gcg", required=True); ap.add_argument("--rd")
    ap.add_argument("--binary-rerun", required=True); ap.add_argument("--human", required=True)
    ap.add_argument("--completions", nargs="*", default=[]); ap.add_argument("--lit")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__), "dashboard_template.html"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)

    # ---- evaluator judgments (long)
    parts = [load_item(a.gcg, "GCG")]
    if a.rd: parts.append(load_item(a.rd, "Refusal Direction"))
    J = pd.concat(parts, ignore_index=True)
    J = J[J.evaluator != "LLM_Judge_binary_no_def"]  # replaced by the rerun (max_tokens=1024)
    rr = pd.read_json(a.binary_rerun, lines=True)
    rr = rr[rr.finish_reason != "error"].drop_duplicates(["attack_method", "condition", "row"], keep="last")
    rr = rr.rename(columns={"attack_method": "attack"})
    rr["evaluator"] = "LLM_Judge_binary_no_def"
    J = pd.concat([J, rr[["attack", "condition", "row", "evaluator", "attack_success", "raw_output"]]], ignore_index=True)
    J["attack_success"] = J.attack_success.astype(bool)
    goals = J.dropna(subset=["goal"]).drop_duplicates("row").set_index("row").goal
    J["goal"] = J.row.map(goals)
    cols = ["attack", "condition", "row", "goal", "evaluator", "attack_success", "parsed_score", "raw_output"]
    J = J.reindex(columns=cols).sort_values(["attack", "condition", "row", "evaluator"])
    J.rename(columns={"condition": "model", "row": "behavior_id"}).to_csv(os.path.join(a.out, "data", "evaluator_judgments.csv"), index=False)

    # ---- human labels (GCG only)
    h = pd.read_csv(a.human)
    h["model"] = h.condition.map(model_from_name)
    h = h.rename(columns={"idx": "behavior_id", "label": "human_success"})[["model", "behavior_id", "human_success"]]
    h.insert(0, "attack", "GCG")
    h.sort_values(["model", "behavior_id"]).to_csv(os.path.join(a.out, "data", "human_labels.csv"), index=False)

    # ---- responses
    R = []
    for p in a.completions:
        attack = "GCG" if os.path.basename(p).lower().startswith("gcg") else "Refusal Direction"
        model = model_from_name(os.path.basename(p))
        for x in json.load(open(p, encoding="utf-8")):
            R.append({"attack": attack, "model": model, "behavior_id": int(x.get("idx", x.get("row"))),
                      "goal": x.get("prompt", x.get("goal")), "response": x.get("response", x.get("completion"))})
    R = pd.DataFrame(R, columns=["attack", "model", "behavior_id", "goal", "response"])
    if len(R): R.sort_values(["attack", "model", "behavior_id"]).to_csv(os.path.join(a.out, "data", "responses.csv"), index=False)

    # ---- wide decisions
    W = J.pivot_table(index=["attack", "condition", "row"], columns="evaluator", values="attack_success", aggfunc="first")
    W = W.reindex(columns=[e for e, _, _ in EVALS]).reset_index().rename(columns={"condition": "model", "row": "behavior_id"})
    for e, _, _ in EVALS:
        W[e] = W[e].map({True: 1, False: 0})
    sc = J[J.evaluator.isin(["LLM_Judge_scale_no_def", "LLM_Judge_scale_with_def"])].pivot_table(
        index=["attack", "condition", "row"], columns="evaluator", values="parsed_score", aggfunc="first").reset_index()
    sc = sc.rename(columns={"condition": "model", "row": "behavior_id",
                            "LLM_Judge_scale_no_def": "score_scale_no_def_1to5",
                            "LLM_Judge_scale_with_def": "score_scale_with_def_mean0to10"})
    W = W.merge(sc, on=["attack", "model", "behavior_id"], how="left").merge(h, on=["attack", "model", "behavior_id"], how="left")
    W.insert(3, "goal", W.behavior_id.map(goals))
    W.to_csv(os.path.join(a.out, "data", "decisions_wide.csv"), index=False)

    # ---- ASR summary
    def wilson(k, n, z=1.96):
        p = k / n; dd = 1 + z * z / n; c = (p + z * z / (2 * n)) / dd
        w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
        return 100 * (c - w), 100 * (c + w)
    rows = []
    for (att, mod), D in W.groupby(["attack", "model"]):
        for e, _, name in EVALS + [("human_success", "human", "Human annotation")]:
            x = D[e].dropna()
            if not len(x): continue
            k, n = int(x.sum()), len(x); lo, hi = wilson(k, n)
            rows.append(dict(attack=att, model=mod, evaluator=name, n=n, successes=k, asr=round(100 * k / n, 1),
                             ci95_low=round(lo, 1), ci95_high=round(hi, 1)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "data", "asr_summary.csv"), index=False)

    # ---- literature review coding
    lit = None
    if a.lit:
        lit = pd.read_csv(a.lit)
        lit.columns = [c.strip() for c in lit.columns]
        lit = lit.dropna(how="all")
        lit.to_csv(os.path.join(a.out, "data", "literature_review_coding.csv"), index=False)

    # ---- dashboard payload
    resp = {(r.attack, r.model, int(r.behavior_id)): r.response for r in R.itertuples()} if len(R) else {}
    rawmap = {}
    for r in J[J.evaluator.isin(["LLM_Judge_binary_no_def", "LLM_Judge_binary_with_def", "OpenAI_Moderation", "LlamaGuard3"])].itertuples():
        rawmap.setdefault((r.attack, r.condition, int(r.row)), {})[KEY[r.evaluator]] = str(r.raw_output)
    items = []
    for r in W.itertuples(index=False):
        rd = r._asdict()
        key = (rd["attack"], rd["model"], int(rd["behavior_id"]))
        it = {"a": ATT[rd["attack"]], "m": MOD[rd["model"]], "i": int(rd["behavior_id"]),
              "d": [None if pd.isna(rd[e]) else int(rd[e]) for e, _, _ in EVALS],
              "s1": None if pd.isna(rd["score_scale_no_def_1to5"]) else float(rd["score_scale_no_def_1to5"]),
              "s2": None if pd.isna(rd["score_scale_with_def_mean0to10"]) else float(rd["score_scale_with_def_mean0to10"]),
              "h": None if pd.isna(rd["human_success"]) else int(rd["human_success"])}
        if key in resp: it["r"] = resp[key]
        if key in rawmap: it["x"] = rawmap[key]
        items.append(it)
    payload = {"evals": [[k, n] for _, k, n in EVALS], "goals": {int(k): v for k, v in goals.items()}, "items": items}
    if lit is not None:
        keep = [c for c in ["Paper Title", "Url", "Publication Year", "Dataset Name", "Risk Categories", "Harmfulness Definition",
                            "Red Teaming Method", "Evaluation Metrics", "Evaluation Instruction stardards/self-customized"] if c in lit.columns]
        payload["lit"] = {"cols": keep, "rows": lit[keep].fillna("").astype(str).values.tolist()}
    html = open(a.template, encoding="utf-8").read().replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(html)
    print("items:", len(items), "| responses:", len(R), "| lit rows:", 0 if lit is None else len(lit))

if __name__ == "__main__":
    main()
