"""Build a regex that matches a JSON number inside [lo, hi] for a given key (plugin eval graders).

Values are compared on their truncated decimal digits at a resolution fine
enough for the interval (about 1/50 of its width). Plain decimal notation only
(json.dump writes floats this way between 1e-4 and 1e16).
"""
import math
import re


def _trie_regex(strings):
    trie = {}
    for s in strings:
        node = trie
        for ch in s:
            node = node.setdefault(ch, {})
        node[""] = {}

    def build(node):
        alts = []
        end = "" in node
        for ch, child in sorted(node.items()):
            if ch == "":
                continue
            alts.append(re.escape(ch) + build(child))
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return f"(?:{body})?" if end else body
    return build(trie)


def number_in_range(lo, hi):
    width = hi - lo
    k = max(1, math.ceil(-math.log10(width / 50.0)))
    n_lo, n_hi = math.floor(lo * 10**k), math.floor(hi * 10**k)
    full = []
    for n in range(n_lo, n_hi + 1):
        s = f"{n / 10**k:.{k}f}"
        full.append(s)
    # a value printed with fewer decimals than k (trailing zeros dropped)
    short = set()
    for s in full:
        point = s.index(".")
        t = s
        while t.endswith("0") and len(t) > point + 2:
            t = t[:-1]
            short.add(t)
    long_part = _trie_regex(full) + r"\d*"
    short_part = _trie_regex(sorted(short)) + r"(?![0-9])" if short else None
    alt = long_part if not short_part else f"(?:{long_part}|{short_part})"
    return alt


if __name__ == "__main__":
    tests = [("ret_B_out", 0.33805, 0.34145, ["0.33980203739576087", "0.3398", "0.34145", "0.33805", "0.34"],
              ["0.3370", "0.3415", "0.34146", "0.33804", "0.35"]),
             ("p0", 0.016428287, 0.016593396, ["0.016510841514151824", "0.0165108", "0.01643"],
              ["0.01919207707988197", "0.0164", "0.0166"]),
             ("T_M", 656.518, 656.718, ["656.6183367689355", "656.62", "656.618"], ["656.4", "656.8", "65.6618"]),
             ("conversion", 0.62375, 0.62625, ["0.625", "0.6250005617", "0.62375"], ["0.6237", "0.63", "0.62"])]
    for key, lo, hi, good, bad in tests:
        pat = re.compile(rf'"{key}"\s*:\s*' + number_in_range(lo, hi) + r"(?![0-9])")
        g = [bool(pat.search(f'{{"{key}": {v}}}')) for v in good]
        b = [bool(pat.search(f'{{"{key}": {v}}}')) for v in bad]
        print(key, "good", g, "bad", b, "len", len(pat.pattern))
