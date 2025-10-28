import os
os.environ.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
import random                 # ← 新增
from gpt4all import GPT4All   # ← 新增（本地LLM）
import random
import json, sys, os
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/classical_qa.jsonl")

def load_qa(path: Path) -> Tuple[List[str], List[str]]:
    qs, ans = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                qs.append(obj["q"])
                ans.append(obj["a"])
    return qs, ans

class RetrievalTutor:
    def __init__(self, qa_path: Path, topk: int = 3, thresh: float = 0.25):
        self.questions, self.answers = load_qa(qa_path)
        self.vec = TfidfVectorizer(stop_words="english").fit(self.questions)
        self.Q = self.vec.transform(self.questions)
        self.topk = topk
        self.thresh = thresh
        # 初始化 GPT4All，不需要 .open()
        self.llm = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf")
        # ❌ 这里删掉 self.llm.open()

        self.persona_system = (
            "You are a warm Classical Music Tutor. Output a thoughtful, natural explanation "
            "using reasoning cues ('because', 'so', 'however'), plus exactly one actionable listening task "
            "and one short follow-up question. Avoid toxicity and robotic lists."
        )
        self.thresh = 0.15


    def _intent(self, q: str) -> str:
        ql = q.lower()
        if any(k in ql for k in ["difference", "区别", "差异", "vs", "compare"]): return "compare"
        if any(k in ql for k in ["recommend", "推荐", "适合", "练", "练习"]): return "recommend"
        if any(k in ql for k in ["what is", "是什么", "定义"]): return "define"
        return "explain"

    def _on_topic(self, a: str, q: str) -> bool:
        ql = q.lower()
        whitelist = []
        for name in ["bach", "handel", "vivaldi", "purcell", "monteverdi", "haydn", "mozart", "beethoven", "schubert",
                     "chopin", "liszt", "brahms", "tchaikovsky", "debussy", "ravel"]:
            if name in ql: whitelist.append(name)
        for genre in ["sonata", "symphony", "concerto", "fugue", "quartet", "mass", "opera", "奏鸣曲", "交响曲",
                      "协奏曲", "赋格", "四重奏", "弥撒", "歌剧"]:
            if genre in ql: whitelist.append(genre)
        return True if not whitelist else any(w.lower() in a.lower() for w in whitelist)

    def _shorten(self, s: str, max_chars: int = 220) -> str:
        s = s.strip()
        return s if len(s) <= max_chars else s[:max_chars].rsplit(" ", 1)[0] + "…"

    def _listening_tip(self, user_q: str) -> str:
        u = user_q.lower()
        if ("baroque" in u or "巴洛克" in u) and ("classical" in u or "古典" in u):
            return "试听任务：各听2分钟——Bach《勃兰登堡三号 I》vs. Haydn《第94交响曲 II》，留意乐句长度与V–I终止。"
        if "beethoven" in u or "贝多芬" in u:
            return "试听任务：比较《Op.49 No.2》与《Op.14 No.2》第一乐章，标出主—副部主题进入的小节。"
        if "mozart" in u or "莫扎特" in u:
            return "试听任务：听《第40交响曲 I》，记录前16小节的乐句分段（4/8小节）。"
        if "chopin" in u or "肖邦" in u:
            return "试听任务：选一段夜曲与一段奏鸣曲慢乐章，比较装饰音与和声色彩对情绪的影响。"
        return "试听任务：找与你问题相关的一段乐曲，记录3个清晰终止点，并描述情绪变化。"

    def _follow_up(self, user_q: str) -> str:
        options = [
            "要不要我按你的水平给一份3首入门曲目清单？",
            "更想深入哪位作曲家，还是先按体裁（奏鸣曲/交响曲/协奏曲）学？",
            "需要带速度标记的练习节拍建议吗？"
        ]
        return random.choice(options)

    def _self_check_hint(self, text: str) -> str:
        red_flags = [
            ("推荐贝多芬奏鸣曲", ["莫扎特", "K."]),
            ("比较巴洛克与古典", ["浪漫主义"])
        ]
        for theme, bads in red_flags:
            if any(bad in text for bad in bads):
                return "（小提示：上面例子里可能混入了非本主题的作曲家/体裁，建议对照曲目再核实。）"
        return ""

    def _polish_with_llm(self, outline: str) -> str:
        prompt = (
            f"<SYSTEM>{self.persona_system}</SYSTEM>\n"
            f"<USER>Rephrase the OUTLINE into a thoughtful teacher-style answer. "
            f"Keep ~120–220 words. Prefer 1–2 short paragraphs; bullets only if necessary for comparisons.\n"
            f"OUTLINE:\n{outline}\n</USER>\n<ASSISTANT>"
        )
        # ✅ 用 chat_session 包裹，保持一致性
        with self.llm.chat_session(system_prompt=self.persona_system):
            resp = self.llm.generate(prompt, max_tokens=400, temp=0.7, top_p=0.95)
        return resp.strip()



    def _format_as_tutor(self, core: str, extras: List[str], user_q: str) -> str:
        openers = [
            "一句话先下结论：", "先抓住主差异：", "快速结论："
        ]
        lines = []
        lines.append(f"{random.choice(openers)}{core}")
        if extras:
            lines.append("进一步可以从这些线索来听：")
            for e in extras[:4]:
                e = e.strip("• ").strip()
                lines.append(f"• {e}")
        lines.append(self._listening_tip(user_q))
        lines.append(f"（{self._follow_up(user_q)}）")
        return "\n".join(lines)

    def reply(self, user_q: str) -> str:
        intent = self._intent(user_q)

        q_vec = self.vec.transform([user_q])
        sims = cosine_similarity(q_vec, self.Q)[0]
        top_idx = sims.argsort()[::-1][:max(6, self.topk)]

        # 低相似度：给澄清式提纲，也走LLM润色
        if float(sims[top_idx[0]]) < self.thresh:
            outline = ("问题有点宽泛；先聚焦作曲家/体裁/时期，再各给代表曲目与听感线索，"
                       "最后附一个2分钟的试听任务去验证。")
            final = self._polish_with_llm(outline)
            return final

        # 候选与守门：减少离题材料
        picked = [(self.questions[i], self.answers[i], float(sims[i])) for i in top_idx]
        picked = [p for p in picked if self._on_topic(p[1], user_q)] or picked[:3]

        # 结论句
        if intent == "compare":
            lead = "一句话结论：两者在织体、句法与音响观上有系统性差异，所以听感会明显不同。"
        elif intent == "recommend":
            lead = "快速建议：选曲要兼顾技术门槛与音乐性，先从旋律清晰、结构明了的作品入手。"
        elif intent == "define":
            lead = "核心定义先给出，再补充常见结构与听感线索："
        else:
            lead = "先给结论，再说为什么："

        core = self._shorten(picked[0][1], 200)
        supports = [self._shorten(a, 160) for _, a, sc in picked[1:] if sc > 0.05][:3]

        # 组织“提纲”
        para1 = f"{lead} {core}"
        if supports:
            joins = ["因为", "因此", "同时", "相对地", "进一步说"]
            para2 = f"{random.choice(joins)}，" + "；".join(supports)
        else:
            para2 = ""
        tip = self._listening_tip(user_q)
        follow = f"（{self._follow_up(user_q)}）"
        outline = "\n".join([para1, para2, tip, follow]).strip()

        # 真·LLM润色成自然语言
        final = self._polish_with_llm(outline)
        warn = self._self_check_hint(final)
        return (final + ("\n" + warn if warn else "")).strip()


def main():
    if not DATA.exists():
        print(f"Data file not found: {DATA.resolve()}", file=sys.stderr)
        sys.exit(1)
    tutor = RetrievalTutor(DATA)
    print("🎼 Classical Music Tutor (retrieval). Type 'exit' to quit.")
    with tutor.llm.chat_session(system_prompt=tutor.persona_system):
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye."); break
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Bye."); break

            # 这里会自动记住上下文
            print(tutor.reply(q), flush=True)

if __name__ == "__main__":
    main()
