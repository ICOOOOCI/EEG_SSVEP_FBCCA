"""模式三的输入法式单词预测：减少 SSVEP 逐字母输入的选择次数。

两种来源合并：
* 本地：随包附带的常用英文词表（含照护常用词），加上从用户自己发送的消息和
  模型回复中学到的词与相邻词对。即时可用，不联网。
* DeepSeek：根据对话上下文和当前草稿预测接下来最可能的单词（可选）。

预测只决定候选键上显示的文字，不改变 FBCCA 解码。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .llm import ChatMessage

WORD_LIST_PATH = Path(__file__).with_name("data") / "english_words.txt"
_WORD = re.compile(r"[A-Z][A-Z']*")


def load_word_list(path: Path = WORD_LIST_PATH) -> list[str]:
    """读取按词频排序的词表（大写、去重、跳过注释和多词短语）。"""
    words: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return words
    for line in lines:
        word = line.strip().upper()
        if not word or word.startswith("#") or not _WORD.fullmatch(word) or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def split_draft(draft: str) -> tuple[str, str, str]:
    """把草稿拆成 (已完成部分, 上一个完整单词, 正在输入的前缀)。"""
    head, _, last = draft.rpartition(" ")
    if last and _WORD.fullmatch(last.upper()):
        context, prefix = head, last.upper()
    else:  # empty, or ends in punctuation/digits: the next word starts fresh
        context, prefix = draft, ""
    previous = (_WORD.findall(context.upper()) or [""])[-1]
    return context, previous, prefix


def apply_suggestion(draft: str, word: str) -> str:
    """用候选词替换正在输入的前缀，并补一个空格，便于直接输入下一个词。"""
    _, _, prefix = split_draft(draft)
    if prefix:
        base = draft[:len(draft) - len(prefix)]
    else:
        base = draft if not draft or draft.endswith(" ") else draft + " "
    return base + word.upper() + " "


class WordPredictor:
    """本地候选：前缀补全 + 基于用户历史的下一个词预测。"""

    def __init__(self, words: Optional[Sequence[str]] = None):
        self.words = list(load_word_list() if words is None else words)
        self.rank = {word: index for index, word in enumerate(self.words)}
        self.used: Counter[str] = Counter()          # words the user typed or read
        self.bigrams: Counter[tuple[str, str]] = Counter()

    def learn(self, text: str, *, weight: int = 1) -> None:
        """从一条消息中学习词频和相邻词对；用户自己发送的消息权重更高。"""
        tokens = _WORD.findall(text.upper())
        for token in tokens:
            self.used[token] += weight
            if token not in self.rank:
                self.rank[token] = len(self.rank)
                self.words.append(token)
        for first, second in zip(tokens, tokens[1:]):
            self.bigrams[(first, second)] += weight

    def _score(self, word: str, previous: str) -> tuple[float, int]:
        """排序依据：相邻词对与使用次数优先，其次是词表词频。"""
        return (-(3.0 * self.bigrams[(previous, word)] + self.used[word]),
                self.rank.get(word, len(self.rank)))

    def suggest(self, draft: str, count: int = 3) -> list[str]:
        """返回最多 count 个候选词。"""
        _, previous, prefix = split_draft(draft)
        if prefix:
            pool = [word for word in self.words if word.startswith(prefix) and word != prefix]
        else:
            followers = [second for (first, second) in self.bigrams if first == previous]
            pool = followers + self.words[:200]
        ranked = sorted(dict.fromkeys(pool), key=lambda word: self._score(word, previous))
        return ranked[:count]


def merge_suggestions(primary: Iterable[str], fallback: Iterable[str], draft: str,
                      count: int = 3) -> list[str]:
    """合并模型候选与本地候选：去重，并只保留与当前前缀一致的词。"""
    _, _, prefix = split_draft(draft)
    merged: list[str] = []
    for word in list(primary) + list(fallback):
        word = word.upper()
        if (_WORD.fullmatch(word) and word.startswith(prefix) and word != prefix
                and word not in merged and len(word) <= 14):
            merged.append(word)
        if len(merged) == count:
            break
    return merged


PREDICTION_SYSTEM_PROMPT = (
    "You are the word-prediction engine of a brain-computer-interface keyboard used by a "
    "person with ALS. Each letter costs the user several seconds, so good predictions save "
    "a lot of effort. Given the recent conversation and the user's unfinished draft, reply "
    "with ONLY the {count} most likely next words, most likely first, in capital English "
    "letters, separated by spaces. If the draft ends in a partial word, every word must "
    "start with that partial word. No punctuation, numbers or explanations."
)


def llm_suggestions(backend, draft: str, history: Sequence[ChatMessage],
                    count: int = 3) -> list[str]:
    """请 DeepSeek 预测下一个词；在后台线程调用，失败时抛出异常由调用方忽略。"""
    context = [ChatMessage("system", PREDICTION_SYSTEM_PROMPT.format(count=count))]
    for message in list(history)[-6:]:
        if message.role in ("user", "assistant"):
            speaker = "USER" if message.role == "user" else "ASSISTANT"
            context.append(ChatMessage("user", f"[{speaker} SAID] {message.content[:600]}"))
    _, _, prefix = split_draft(draft)
    request = (f"Draft so far: \"{draft}\"\n"
               + (f"The user is in the middle of the word starting with \"{prefix}\"."
                  if prefix else "The user is about to start a new word."))
    answer = backend.send(request, history=context)
    return _WORD.findall(answer.upper())[:count * 2]


__all__ = ["WordPredictor", "apply_suggestion", "load_word_list", "llm_suggestions",
           "merge_suggestions", "split_draft", "WORD_LIST_PATH"]
