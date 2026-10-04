"""海缆连续缆段整数微应变联合反演核心（仅使用 Python 标准库）。

输入
----
- 6..12 段按顺序排列的缆段长度（正整数）
- 统一应变闭区间 [strain_min, strain_max]（整数微应变）
- 8..20 个观测窗，每窗给出连续起止段（1 基，含端点）与累计伸长量闭区间

模型（所有比较与运算均为 Python 任意精度精确整数）
--------------------------------------------------
待求逐段应变为 x_0..x_{n-1}（整数微应变）。

观测窗 [s,e] 约束：Σ_{i=s..e} L_i·x_i ∈ [lo, hi]。
每段还须满足 strain_min ≤ x_i ≤ strain_max。

引入前缀和 P_0=0, P_i = Σ_{j<i} L_j·x_j，则
    P_{e+1} - P_s ∈ [lo, hi]
    P_{i+1} - P_i ∈ [L_i·strain_min, L_i·strain_max]
全部是 P 上的**差分约束**，其最长路闭包（Floyd-Warshall）：
- 出现正环 ⇒ 整体不可行（可靠的充分必要判定，针对松弛后的实值域；
  整数可行性仍以下层整数搜索为准）；
- 给出每个 P_i 的精确上下界，进而导出每段 x_i 的紧整数域：
      x_i ≥ ceil( D[i][i+1] / L_i )
      x_i ≤ floor( -D[i+1][i] / L_i )

优选准则（字典序三级，逐级不可放宽）
1. 最小化相邻段最大应变差 M = max_k |x_k - x_{k-1}|
   —— 在 x 空间就是差分约束 x_k - x_{k-1} ∈ [-M, M]，同样并入最长路闭包。
2. 在 1 的最优解中最小化 S = Σ_k |x_k - x_{k-1}|
   —— M 较小时在差分空间 (x0,d_k) 直接以 Σ|d_k| 的一元域收紧精确判定；
      M 很大时在 x 空间由相邻段一元域给出各边 |Δ| 的紧包络，把 Σ|Δ|
      预算折算为逐边差分约束，并用一次分支定界直接求最小 S。
3. 在 1、2 的最优解中取应变序列 (x_0,...,x_{n-1}) 字典序最小者。

求解：
- 最长路闭包缩域 + 整数界传播（含精确等式的丢番图余数类传播）+ MRV
  /折半回溯；
- 纯整数（奇偶/模）矛盾先在一组小素数有限域上做高斯消元预检截杀，
  覆盖差分闭包（仅实值域）无法识别的「系数偶、右端奇」一类冲突；
  精确等式还做整数行变换（行 HNF 风格，么模、解集不变），化出单位
  主元/小系数行供界传播；
- 前缀闭包对每个相邻段区间蕴含的紧伸长量界作为冗余短约束喂给传播；
- 界传播使用增量事件队列：每次分支只钉一个变量，只重放涉及它的约束，
  不再在每个回溯节点全量重扫所有长窗；
- 最小 M 以「见证解实际 M 为上界、相邻段域间隔为下界」收窄二分区间；
  最小 S（大 M 时）由单次分支定界钉死，字典序各值再逐段二分；
- 阶段 2/3 按域宽在「差分空间 / x 空间」间自适应选择；
- 重复观测窗在求解侧按 (段范围, 区间) 去重，结果仍逐窗回算。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

MIN_SEGMENTS = 6
MAX_SEGMENTS = 12
MIN_WINDOWS = 8
MAX_WINDOWS = 20

# 模 p 可行性预检所用素数：整数可行 ⇒ 对每个素数模可行。
# 差分闭包只覆盖实值域，无法识别「全部系数偶而右端奇」这类格/奇偶矛盾；
# 在这些有限域上做高斯消元即可在线性时间级内截杀纯整数矛盾。
_PRESOLVE_PRIMES = (2, 3, 5, 7, 11)


class ValidationErrors(ValueError):
    """输入字段错误（可一次性包含多个字段问题）。"""

    def __init__(self, fields: list[dict[str, str]]):
        self.fields = fields
        super().__init__("; ".join(f"{f['field']}: {f['message']}" for f in fields))


class InfeasibleError(ValueError):
    """全部输入合法，但观测窗彼此冲突（含统一应变界），无可行解。"""


@dataclass(frozen=True)
class Window:
    start: int  # 0 基，含端点
    end: int  # 0 基，含端点
    lo: int  # 累计伸长量（长度加权应变和）闭区间下端
    hi: int  # 闭区间上端


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _err(fields: list[dict[str, str]], field: str, message: str) -> None:
    fields.append({"field": field, "message": message})


def _validate_lengths(payload: dict, fields: list[dict[str, str]]) -> list[int] | None:
    raw = payload.get("segment_lengths")
    if raw is None:
        _err(fields, "segment_lengths", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "segment_lengths", "must be an array of positive integers")
        return None
    lengths: list[int] = []
    for i, item in enumerate(raw):
        if not _is_int(item) or item < 1:
            _err(fields, f"segment_lengths[{i}]", "must be a positive integer")
        else:
            lengths.append(item)
    if not (MIN_SEGMENTS <= len(raw) <= MAX_SEGMENTS):
        _err(
            fields,
            "segment_lengths",
            f"must contain between {MIN_SEGMENTS} and {MAX_SEGMENTS} segments"
            f" (got {len(raw)})",
        )
    return lengths if len(lengths) == len(raw) else None


def _validate_strain_bounds(
    payload: dict, fields: list[dict[str, str]]
) -> tuple[int, int] | None:
    raw = payload.get("strain_bounds")
    if raw is None:
        _err(fields, "strain_bounds", "field is required")
        return None
    if not isinstance(raw, dict):
        _err(fields, "strain_bounds", 'must be an object {"min": int, "max": int}')
        return None
    for key in raw:
        if key not in ("min", "max"):
            _err(fields, f"strain_bounds.{key}", "unknown field")
    lo = raw.get("min")
    hi = raw.get("max")
    ok = True
    if not _is_int(lo):
        _err(fields, "strain_bounds.min", "must be an integer")
        ok = False
    if not _is_int(hi):
        _err(fields, "strain_bounds.max", "must be an integer")
        ok = False
    if ok and lo > hi:
        _err(fields, "strain_bounds", "min must be less than or equal to max")
        return None
    return (lo, hi) if ok else None


def _validate_windows(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) ->list[Window] | None:
    raw = payload.get("windows")
    if raw is None:
        _err(fields, "windows", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "windows", "must be an array of observation windows")
        return None
    if not (MIN_WINDOWS <= len(raw) <= MAX_WINDOWS):
        _err(
            fields,
            "windows",
            f"must contain between {MIN_WINDOWS} and {MAX_WINDOWS} windows"
            f" (got {len(raw)})",
        )
    windows: list[Window] = []
    all_ok = True
    for i, item in enumerate(raw):
        prefix = f"windows[{i}]"
        if not isinstance(item, dict):
            _err(fields, prefix, "must be an object")
            all_ok = False
            continue
        ok = True
        for key in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
            if key not in item:
                _err(fields, f"{prefix}.{key}", "field is required")
                ok = False
            elif not _is_int(item[key]):
                _err(fields, f"{prefix}.{key}", "must be an integer")
                ok = False
        for key in item:
            if key not in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
                _err(fields, f"{prefix}.{key}", "unknown field")
                ok = False
        if not ok:
            all_ok = False
            continue
        s = item["start_segment"]
        e = item["end_segment"]
        lo = item["min_elongation"]
        hi = item["max_elongation"]
        if n is not None:
            if not (1 <= s <= n):
                _err(fields, f"{prefix}.start_segment", f"must be between 1 and {n}")
                ok = False
            if not (1 <= e <= n):
                _err(fields, f"{prefix}.end_segment", f"must be between 1 and {n}")
                ok = False
            if ok and s > e:
                _err(fields, f"{prefix}.start_segment", "must be <= end_segment")
                ok = False
        if lo > hi:
            _err(
                fields,
                f"{prefix}.min_elongation",
                "must be less than or equal to max_elongation",
            )
            ok = False
        if ok and n is not None:
            windows.append(Window(start=s - 1, end=e - 1, lo=lo, hi=hi))
        else:
            all_ok = False
    return windows if all_ok else None


_ALLOWED_TOP_LEVEL = {"segment_lengths", "strain_bounds", "windows"}


def validate(payload: object) -> tuple[list[int], int, int, list[Window]]:
    """校验并归一化输入；非法时抛 ValidationErrors。"""
    fields: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        raise ValidationErrors(
            [{"field": ".", "message": "request body must be a JSON object"}]
        )
    for key in payload:
        if key not in _ALLOWED_TOP_LEVEL:
            _err(fields, key, "unknown field")
    lengths = _validate_lengths(payload, fields)
    bounds = _validate_strain_bounds(payload, fields)
    n = len(lengths) if lengths is not None else None
    windows = _validate_windows(payload, n, fields)
    if fields:
        raise ValidationErrors(fields)
    assert lengths is not None and bounds is not None and windows is not None
    return lengths, bounds[0], bounds[1], windows


# --------------------------------------------------------------------------- #
# 差分约束最长路闭包
# --------------------------------------------------------------------------- #
def _longest_path_closure(
    node_count: int, edges: list[tuple[int, int, int]]
) -> list[list[int | None]] | None:
    """约束 P_v >= P_u + w 的最长路闭包；存在正环返回 None。

    D[i][j] = P_j - P_i 的最强（最大）下界；None 表示尚无路径（-∞）。
    D[i][i] > 0 即正环，矛盾。以 None 为哨兵可兼容任意精度输入。
    """
    d: list[list[int | None]] = [[None] * node_count for _ in range(node_count)]
    for i in range(node_count):
        d[i][i] = 0
    for u, v, w in edges:
        if d[u][v] is None or w > d[u][v]:
            d[u][v] = w
    for k in range(node_count):
        dk = d[k]
        for i in range(node_count):
            dik = d[i][k]
            if dik is None:
                continue
            di = d[i]
            for j in range(node_count):
                if dk[j] is None:
                    continue
                cand = dik + dk[j]
                if di[j] is None or cand > di[j]:
                    di[j] = cand
    for i in range(node_count):
        if d[i][i] is not None and d[i][i] > 0:
            return None
    return d


def _ceil_div(a: int, b: int) -> int:
    """数学上的 ceil(a / b)（b>0），纯整数。"""
    return -((-a) // b)


def _prefix_closure(
    n: int, lengths: list[int], strain_min: int, strain_max: int,
    windows: list[Window],
) -> list[list[int]] | None:
    """前缀和 P 上的差分约束闭包（窗 + 每段应变界）。"""
    edges: list[tuple[int, int, int]] = []
    for i, length in enumerate(lengths):  # P_{i+1} - P_i ∈ [L·smin, L·smax]
        edges.append((i, i + 1, length * strain_min))
        edges.append((i + 1, i, -length * strain_max))
    for win in windows:  # P_{e+1} - P_s ∈ [lo, hi]
        edges.append((win.start, win.end + 1, win.lo))
        edges.append((win.end + 1, win.start, -win.hi))
    return _longest_path_closure(n + 1, edges)


def _x_bounds_from_prefix(
    n: int,
    lengths: list[int],
    strain_min: int,
    strain_max: int,
    p_closure: list[list[int]],
) -> tuple[list[int], list[int]]:
    """由 P 的最强界导出每段 x_i 的紧整数域。"""
    los: list[int] = []
    his: list[int] = []
    for i, length in enumerate(lengths):
        forward = p_closure[i][i + 1]
        backward = p_closure[i + 1][i]
        # i -> i+1 与 i+1 -> i 均为直接边，闭包后必非 None。
        assert forward is not None and backward is not None
        lo = max(strain_min, _ceil_div(forward, length))
        hi = min(strain_max, (-backward) // length)
        los.append(lo)
        his.append(hi)
    return los, his


# --------------------------------------------------------------------------- #
# 有限域（模素数）可行性预检
# --------------------------------------------------------------------------- #
def _mod_feasible(
    los: list[int],
    his: list[int],
    constraints: list[Constraint],
    prime: int,
) -> bool:
    """线性等式约束组在 F_p 上是否有解。

    精确窗（lo==hi）与单点域变量（x_i 被钉为常量）都是精确整数等式。
    任何整数解模 p 后必为 F_p 解，故 F_p 无解 ⇒ 整数不可行（可靠的剪枝）。
    不等式约束无法直接搬上有限域，予以略去——有限域系统是整数问题的
    松弛，故其不可行仍是整数不可行的充分条件；反向情形交给下层搜索兜底。
    """
    p = prime
    var_count = len(los)
    # 每个等式用一行系数（常数项放在最后一列）：Σ a_j x_j ≡ c (mod p)。
    rows: list[list[int]] = []
    for terms, a, b in constraints:
        if a != b:
            continue  # 仅等式可直接搬上有限域
        row = [0] * (var_count + 1)
        for j, w in terms:
            row[j] = (row[j] + w) % p
        row[var_count] = a % p
        rows.append(row)
    for i in range(var_count):
        if los[i] == his[i]:
            row = [0] * (var_count + 1)
            row[i] = 1
            row[var_count] = los[i] % p
            rows.append(row)
    # 高斯-若尔当消元（按列选主元）。
    pivot_col = 0
    r = 0
    row_count = len(rows)
    while r < row_count and pivot_col < var_count:
        pivot = -1
        for i in range(r, row_count):
            if rows[i][pivot_col] % p != 0:
                pivot = i
                break
        if pivot == -1:
            pivot_col += 1
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = pow(rows[r][pivot_col], p - 2, p)
        rows[r] = [(v * inv) % p for v in rows[r]]
        for i in range(row_count):
            if i != r and rows[i][pivot_col] % p != 0:
                factor = rows[i][pivot_col]
                rows[i] = [
                    (a - factor * b) % p for a, b in zip(rows[i], rows[r])
                ]
        r += 1
        pivot_col += 1
    for i in range(r, row_count):
        coef_nonzero = any(rows[i][j] % p != 0 for j in range(var_count))
        if not coef_nonzero and rows[i][var_count] % p != 0:
            return False  # 0 ≡ 非零常数
    return True


def _mod_presolve(
    los: list[int], his: list[int], constraints: list[Constraint],
    primes: tuple[int, ...] = _PRESOLVE_PRIMES,
) -> bool:
    """在给定素数上做有限域预检；任一素数无解即整数不可行。"""
    return all(
        _mod_feasible(los, his, constraints, p) for p in primes
    )


def _exact_row_hnf(
    var_count: int, constraints: list[Constraint]
) -> list[Constraint]:
    """对精确等式组做整数行变换（行 HNF 风格），得到更强的等价精确等式。

    精确观测窗之间只差少数几段时，原始行在公共变量上系数巨大（如全长窗
    系数 25、内含窗系数 9），一元界传播几乎无法收紧公共变量；而这些精确
    等式的**整数线性组合仍是精确等式**。用扩展欧几里得对逐列做整数消元
    （行运算么模，解集逐字不变），尽量把某变量系数化到 1，从而把该变量
    直接钉成其余变量与常数的线性式，传播强度大幅提高。

    仅返回「系数变小或出现单位主元」的新行；它们与原等式联立，可行集
    完全一致（新行是原行的整数线性组合，且消元可逆）。为控制规模，
    只对本就精确（lo==hi）的约束做处理，并限制迭代步数。
    """
    rows: list[list[int]] = []
    rhs: list[int] = []
    for terms, a, b in constraints:
        if a != b:
            continue
        row = [0] * var_count
        for j, w in terms:
            row[j] += w
        if any(v != 0 for v in row):
            rows.append(row)
            rhs.append(a)
    if len(rows) <= 1:
        return []

    r_count = len(rows)
    # 逐列整数消元：在该列找系数绝对值最小的行作主元，用扩展 gcd 把它
    # 化到 gcd，其余行减去整倍数清零该列（经典整数高斯消元 / 行 HNF）。
    pivot_row = 0
    for col in range(var_count):
        if pivot_row >= r_count:
            break
        # 选主元：该列非零且绝对值最小的行（化 1 最快）。
        cand = [r for r in range(pivot_row, r_count) if rows[r][col] != 0]
        if not cand:
            continue
        cand.sort(key=lambda r: abs(rows[r][col]))
        pr = cand[0]
        rows[pivot_row], rows[pr] = rows[pr], rows[pivot_row]
        rhs[pivot_row], rhs[pr] = rhs[pr], rhs[pivot_row]
        # 用其它非零行与主元行做扩展 gcd，把主元系数化到 g = gcd(全体)。
        changed = True
        guard = 0
        while changed and guard < 64:
            changed = False
            guard += 1
            piv = rows[pivot_row][col]
            for r in range(pivot_row + 1, r_count):
                v = rows[r][col]
                if v == 0:
                    continue
                g, x, y = _extended_gcd(abs(piv), abs(v))
                # x*piv + y*v = g（注意符号由系数符号修正）
                sp = 1 if piv >= 0 else -1
                sv_ = 1 if v >= 0 else -1
                xp = x * sp
                yv = y * sv_
                new_piv_row = [xp * rows[pivot_row][j] + yv * rows[r][j]
                               for j in range(var_count)]
                new_piv_rhs = xp * rhs[pivot_row] + yv * rhs[r]
                # 消去 r 行该列：系数 (-v/g) 与 (piv/g)。
                qp = -v // g
                qv = piv // g
                new_r_row = [qp * rows[pivot_row][j] + qv * rows[r][j]
                             for j in range(var_count)]
                new_r_rhs = qp * rhs[pivot_row] + qv * rhs[r]
                rows[pivot_row] = new_piv_row
                rhs[pivot_row] = new_piv_rhs
                rows[r] = new_r_row
                rhs[r] = new_r_rhs
                changed = True
        # 此时主元行该列 = gcd，其余行该列 = 0。
        pivot_row += 1

    out: list[Constraint] = []
    for row, c in zip(rows, rhs):
        terms = tuple((j, w) for j, w in enumerate(row) if w != 0)
        if terms:
            out.append((terms, c, c))
    return out


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """扩展欧几里得：返回 (g, x, y) 使 a*x + b*y = g = gcd(a,b)。"""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


# 搜索不动点处只跑最小的两个素数：每节点都要调用，需保持极廉价；
# 完整素数集仅用于搜索前的一次性预检。
_SEARCH_PRIMES = (2, 3)


def _x_diff_closure(
    n: int, los: list[int], his: list[int], m: int
) -> tuple[list[int], list[int]] | None:
    """并入一元域与 |x_k-x_{k-1}|<=M 后的 x 最长路闭包（锚点节点 n）。"""
    anchor = n
    edges: list[tuple[int, int, int]] = []
    for i in range(n):  # x_i >= anchor + lo_i；anchor >= x_i - hi_i
        edges.append((anchor, i, los[i]))
        edges.append((i, anchor, -his[i]))
    for k in range(1, n):
        edges.append((k - 1, k, -m))  # x_k >= x_{k-1} - M
        edges.append((k, k - 1, -m))  # x_{k-1} >= x_k - M
    closure = _longest_path_closure(n + 1, edges)
    if closure is None:
        return None
    out_lo = [closure[anchor][i] if closure[anchor][i] is not None else los[i] for i in range(n)]
    out_hi = [-closure[i][anchor] if closure[i][anchor] is not None else his[i] for i in range(n)]
    for i in range(n):
        if out_lo[i] > out_hi[i]:
            return None
    return out_lo, out_hi


# --------------------------------------------------------------------------- #
# 通用整数界传播 + 回溯（x 空间）
# --------------------------------------------------------------------------- #
# 约束：sum_j w_j * var_j ∈ [lo, hi]，端点可为 None。
Constraint = tuple[tuple[tuple[int, int], ...], int | None, int | None]


def _propagate(
    los: list[int], his: list[int], constraints: list[Constraint],
    mod_check: bool = True,
) -> tuple[list[int], list[int]] | None:
    """对所有线性区间约束反复做一元界传播；矛盾返回 None。

    每轮对每条约束只扫描其项两遍（O(约束数·项数)）：第一遍求
    和的整体包络 [smin,smax]、自由变量包络与系数 gcd；第二遍用
    「总和减去本项」做一元界收紧。

    mod_check 控制不动点处是否再跑模素数高斯消元。它只是剪枝（绝不影响
    正确性：叶子点由各约束包络直接精确核对）。差分空间搜索的原始精确窗
    方程组已在进入搜索前做过完整素数集预检，且搜索中新增的只是差分/绝对值
    不等式，因此该路径关闭它以省去每节点的高斯消元开销。
    """
    los = list(los)
    his = list(his)
    while True:
        changed = False
        for terms, a, b in constraints:
            # 本约束在本轮开始时的变量划分快照：即使本轮把某变量收紧成单点，
            # 也要等下一轮再重算包络，避免 fixed/free 口径中途不一致。
            fixed = 0
            free_terms: list[tuple[int, int]] = []
            free_min = 0
            free_max = 0
            g = 0
            for j, w in terms:
                if los[j] == his[j]:
                    fixed += w * los[j]
                else:
                    free_terms.append((j, w))
                    g = gcd(g, abs(w))
                    if w > 0:
                        free_min += w * los[j]
                        free_max += w * his[j]
                    else:
                        free_min += w * his[j]
                        free_max += w * los[j]
            smin = free_min + fixed
            smax = free_max + fixed
            # gcd/包络快判：自由变量之和必为系数 gcd 的倍数，且必落在
            # 其当前一元界包络内；与残差区间不交即矛盾（单点变量也在此
            # 被核对，故后加入的精确约束与其冲突时不会漏检）。
            rlo = None if a is None else a - fixed
            rhi = None if b is None else b - fixed
            lo_bound = free_min if rlo is None else max(free_min, rlo)
            hi_bound = free_max if rhi is None else min(free_max, rhi)
            if lo_bound > hi_bound:
                return None
            if g > 1:
                first = lo_bound + ((-lo_bound) % g)
                if first > hi_bound:
                    return None
            # 精确等式（lo==hi）的丢番图余数传播：其余自由变量系数的
            # gcd g_j 决定 w_j·x_j 的余数类，据此把 x_j 的一元域收紧到
            # 某个同余类（区间约束的右端不唯一时一般无此强结论，故仅对
            # 等式做），显著压缩「精确观测窗」的回溯空间。
            # 用前缀/后缀 gcd 令每个 g_j 只花 O(1)（整体 O(自由项数)）。
            exact = a is not None and a == b
            nf = len(free_terms)
            if exact and nf >= 2:
                pref = [0] * (nf + 1)
                for t in range(nf):
                    pref[t + 1] = gcd(pref[t], abs(free_terms[t][1]))
                suffix_g = 0
                gjs: list[int] = [0] * nf
                for t in range(nf - 1, -1, -1):
                    gjs[t] = gcd(pref[t], suffix_g)
                    suffix_g = gcd(suffix_g, abs(free_terms[t][1]))
            else:
                gjs = [0] * nf
            for t, (j, w) in enumerate(free_terms):
                gj = gjs[t]
                if exact and gj > 1:
                    r = a - fixed  # w_j·x_j ≡ r (mod gj)
                    g0 = gcd(abs(w), gj)
                    if r % g0 != 0:
                        return None
                    modulus = gj // g0
                    if modulus > 1:
                        ww = (abs(w) // g0) % modulus
                        rr = ((r // g0) % modulus) * pow(ww, -1, modulus)
                        if w < 0:
                            rr = (-rr) % modulus
                        rr %= modulus
                        low_v = los[j] + (rr - los[j]) % modulus
                        high_v = his[j] - (his[j] - rr) % modulus
                        if low_v > high_v:
                            return None
                        if low_v > los[j]:
                            los[j] = low_v
                            changed = True
                        if high_v < his[j]:
                            his[j] = high_v
                            changed = True
                if w > 0:
                    others_min = smin - w * los[j]
                    others_max = smax - w * his[j]
                else:
                    others_min = smin - w * his[j]
                    others_max = smax - w * los[j]
                if b is not None:  # w*x_j <= b - others_min
                    c = b - others_min
                    if w > 0:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                    else:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                if a is not None:  # w*x_j >= a - others_max
                    c = a - others_max
                    if w > 0:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                    else:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                if los[j] > his[j]:
                    return None
        if not changed:
            # 传播到不动点后做有限域预检：纯整数（奇偶/模）矛盾在一元界
            # 传播中不可见，但模素数高斯消元可立刻识别，避免回溯爆炸。
            # 每个搜索节点都到这里，故只跑最小的两个素数以保持廉价；
            # 已在上层做过完整素数集预检的路径可显式关闭。
            if mod_check and not _mod_presolve(
                los, his, constraints, _SEARCH_PRIMES
            ):
                return None
            return los, his


def _static_priority(
    var_count: int, constraints: list[Constraint]
) -> list[tuple[int, int]]:
    """每个变量的静态 MRV 平局优先级（只取决于约束结构，搜索中不变，算一次）。

    取包含该变量的约束中 (项数-等式加成, -该变量系数绝对值) 的最小值：
    优先处在短/精确约束且系数大的变量。默认值让不属于任何约束的变量排最后。
    """
    prio: list[tuple[int, int]] = [(10**9, 0)] * var_count
    for terms, a, b in constraints:
        exact_bonus = 1 if a == b else 0
        span = len(terms) - exact_bonus
        for j, w in terms:
            key = (span, -abs(w))
            if key < prio[j]:
                prio[j] = key
    return prio


def _choose_variable(
    los: list[int], his: list[int], priority: list[tuple[int, int]]
) -> int:
    """MRV：选当前域最小的自由变量；平局时按静态优先级。"""
    best = -1
    best_key: tuple[int, tuple[int, int]] | None = None
    for j in range(len(los)):
        if los[j] != his[j]:
            key = (his[j] - los[j], priority[j])
            if best_key is None or key < best_key:
                best = j
                best_key = key
    return best


# --------------------------------------------------------------------------- #
# x 空间带 Σ|Δ| 阈值的搜索（阶段 2/3，最优 M 较大时）
# --------------------------------------------------------------------------- #
class _IncStore:
    """约束的预处理直存（避免每节点重复解包 tuple、重复 gcd）。"""

    __slots__ = ("vars", "weights", "a", "b", "exact")

    def __init__(self, con: Constraint):
        terms, a, b = con
        self.vars = tuple(j for j, _ in terms)
        self.weights = tuple(w for _, w in terms)
        self.a = a
        self.b = b
        self.exact = a is not None and a == b


def _propagate_inc(
    los: list[int],
    his: list[int],
    store: list[_IncStore],
    occurred: list[list[int]],
    pending: list[int] | None = None,
) -> bool:
    """增量界传播：只重算涉及「自上次以来被收紧变量」的约束。

    与 _propagate 同样的一元界/丢番图余数规则（只是剪枝：叶子仍由各约束
    包络精确核对），但搜索每次分支只钉一个变量，全量重扫所有长窗约束的
    绝大部分工作是浪费；事件队列只重放受影响约束直至不动点。返回 False
    表示矛盾。

    约束内沿用「本轮快照」口径：进入约束时记录各变量域快照，包络与
    「其余项」均基于快照计算；同约束内更早变量本轮的收紧留到下一轮
    （可能稍松，绝不误杀），与 _propagate 的逐约束两遍口径一致。
    """
    from collections import deque

    if pending is None:
        queue = deque(range(len(store)))
    else:
        queue = deque(pending)
    queued = [False] * len(store)
    while queue:
        ci = queue.popleft()
        queued[ci] = False
        st = store[ci]
        vs, ws, a, b, exact = st.vars, st.weights, st.a, st.b, st.exact
        m = len(vs)
        snap_lo = [0] * m
        snap_hi = [0] * m
        fixed = 0
        free_idx: list[int] = []
        free_min = 0
        free_max = 0
        g = 0
        for t in range(m):
            j = vs[t]
            lj, hj = los[j], his[j]
            snap_lo[t], snap_hi[t] = lj, hj
            if lj == hj:
                fixed += ws[t] * lj
            else:
                free_idx.append(t)
                w = ws[t]
                g = gcd(g, abs(w))
                if w > 0:
                    free_min += w * lj
                    free_max += w * hj
                else:
                    free_min += w * hj
                    free_max += w * lj
        rlo = None if a is None else a - fixed
        rhi = None if b is None else b - fixed
        lo_bound = free_min if rlo is None else max(free_min, rlo)
        hi_bound = free_max if rhi is None else min(free_max, rhi)
        if lo_bound > hi_bound:
            return False
        if g > 1 and lo_bound + ((-lo_bound) % g) > hi_bound:
            return False
        nf = len(free_idx)
        if exact and nf >= 2:
            pref = [0] * (nf + 1)
            for t in range(nf):
                pref[t + 1] = gcd(pref[t], abs(ws[free_idx[t]]))
            suffix_g = 0
            gjs = [0] * nf
            for t in range(nf - 1, -1, -1):
                gjs[t] = gcd(pref[t], suffix_g)
                suffix_g = gcd(suffix_g, abs(ws[free_idx[t]]))
        else:
            gjs = [0] * nf
        changed_vars: list[int] = []
        for pos in range(nf):
            t = free_idx[pos]
            j, w = vs[t], ws[t]
            gj = gjs[pos]
            if exact and gj > 1:
                r = a - fixed  # w·x_j ≡ r (mod gj)
                g0 = gcd(abs(w), gj)
                if r % g0 != 0:
                    return False
                modulus = gj // g0
                if modulus > 1:
                    ww = (abs(w) // g0) % modulus
                    rr = ((r // g0) % modulus) * pow(ww, -1, modulus)
                    if w < 0:
                        rr = -rr
                    rr %= modulus
                    low_v = los[j] + (rr - los[j]) % modulus
                    high_v = his[j] - (his[j] - rr) % modulus
                    if low_v > high_v:
                        return False
                    if low_v > los[j]:
                        los[j] = low_v
                        changed_vars.append(j)
                    if high_v < his[j]:
                        his[j] = high_v
                        changed_vars.append(j)
            lj0, hj0 = snap_lo[t], snap_hi[t]
            if w > 0:
                others_min = free_min - w * lj0
                others_max = free_max - w * hj0
            else:
                others_min = free_min - w * hj0
                others_max = free_max - w * lj0
            # 其余自由项包络 + 已固定项 = 其余项总包络。
            rest_min = others_min + fixed
            rest_max = others_max + fixed
            if b is not None:  # w*x_j <= b - rest_min
                c = b - rest_min
                if w > 0:
                    v = c // w
                    if v < his[j]:
                        his[j] = v
                        changed_vars.append(j)
                else:
                    v = _ceil_div(c, w)
                    if v > los[j]:
                        los[j] = v
                        changed_vars.append(j)
            if a is not None:  # w*x_j >= a - rest_max
                c = a - rest_max
                if w > 0:
                    v = _ceil_div(c, w)
                    if v > los[j]:
                        los[j] = v
                        changed_vars.append(j)
                else:
                    v = c // w
                    if v < his[j]:
                        his[j] = v
                        changed_vars.append(j)
            if los[j] > his[j]:
                return False
        for j in changed_vars:
            for cj in occurred[j]:
                if not queued[cj]:
                    queued[cj] = True
                    queue.append(cj)
    return True


def _edge_abs_bounds_x(
    los: list[int], his: list[int], k: int
) -> tuple[int, int]:
    """由相邻段 x 的一元域给出 |x_k-x_{k-1}| 的紧上下界。"""
    d_lo = los[k] - his[k - 1]
    d_hi = his[k] - los[k - 1]
    if d_lo >= 0:
        return d_lo, d_hi
    if d_hi <= 0:
        return -d_hi, -d_lo
    return 0, max(d_hi, -d_lo)


class _XState:
    """阶段 2/3 x 空间搜索的一次性预处理状态（所有试解共享）。"""

    __slots__ = ("n", "store", "occurred", "priority", "exact_cons",
                 "s_lo", "s_hi", "active")

    def __init__(self, n: int, base_constraints: list[Constraint],
                 s_lo: int | None, s_hi: int | None):
        self.n = n
        self.s_lo = s_lo
        self.s_hi = s_hi
        self.active = s_lo is not None or s_hi is not None
        self.store = [_IncStore(c) for c in base_constraints]
        occurred: list[list[int]] = [[] for _ in range(n)]
        exact_cons: list[Constraint] = []
        for ci, st in enumerate(self.store):
            for j in st.vars:
                occurred[j].append(ci)
            if st.exact:
                terms = tuple(zip(st.vars, st.weights))
                exact_cons.append((terms, st.a, st.b))
        self.occurred = occurred
        self.exact_cons = exact_cons
        self.priority = _static_priority(n, base_constraints)


def _edge_round_x(
    los: list[int], his: list[int], st: _XState,
) -> list[int] | None:
    """按相邻段一元域把 Σ|Δ| 阈值折算为逐边收紧；返回受影响变量表。

    - s_hi：|Δ_k| <= c_k = s_hi - Σ_{j≠k}|Δ_j|_lb，直接收紧两端一元域
      （精确等价于 Σ|Δ|≤s_hi 的逐边必要界，随钉死迭代趋紧）；
    - s_lo：仅当相邻段域已不相交（符号已定）才收紧，跨 0 交给分支。
    矛盾返回 None；无任何收紧返回 []。
    """
    n = st.n
    s_lo, s_hi = st.s_lo, st.s_hi
    edge_lb = [0] * n
    edge_ub = [0] * n
    for k in range(1, n):
        edge_lb[k], edge_ub[k] = _edge_abs_bounds_x(los, his, k)
    total_lb = sum(edge_lb)
    total_ub = sum(edge_ub)
    if s_hi is not None and total_lb > s_hi:
        return None
    if s_lo is not None and total_ub < s_lo:
        return None
    touched: list[int] = []
    if s_hi is not None:
        for k in range(1, n):
            cap = s_hi - (total_lb - edge_lb[k])
            if cap < 0:
                return None
            # x_k ∈ [x_{k-1}-cap, x_{k-1}+cap]，双向各收紧一次。
            nl = los[k - 1] - cap
            if nl > los[k]:
                los[k] = nl
                touched.append(k)
            nh = his[k - 1] + cap
            if nh < his[k]:
                his[k] = nh
                touched.append(k)
            nl = los[k] - cap
            if nl > los[k - 1]:
                los[k - 1] = nl
                touched.append(k - 1)
            nh = his[k] + cap
            if nh < his[k - 1]:
                his[k - 1] = nh
                touched.append(k - 1)
            if los[k] > his[k] or los[k - 1] > his[k - 1]:
                return None
    if s_lo is not None:
        for k in range(1, n):
            need = s_lo - (total_ub - edge_ub[k])
            if need <= 0:
                continue
            # |Δ_k| >= need；跨 0 域无法用一元界表达，仅在符号已定时收紧。
            if los[k] > his[k - 1]:  # Δ_k = x_k-x_{k-1} 恒正
                nl = los[k - 1] + need  # x_k >= x_{k-1}+need
                nh = his[k] - need      # x_{k-1} <= x_k-need
                if nl > los[k]:
                    los[k] = nl
                    touched.append(k)
                if nh < his[k - 1]:
                    his[k - 1] = nh
                    touched.append(k - 1)
            elif his[k] < los[k - 1]:  # Δ_k 恒负
                nh = his[k - 1] - need  # x_k <= x_{k-1}-need
                nl = los[k] + need      # x_{k-1} >= x_k+need
                if nh < his[k]:
                    his[k] = nh
                    touched.append(k)
                if nl > los[k - 1]:
                    los[k - 1] = nl
                    touched.append(k - 1)
            if los[k] > his[k] or los[k - 1] > his[k - 1]:
                return None
    return touched


def _xspace_close(
    los: list[int], his: list[int], st: _XState, pending: list[int] | None,
) -> bool:
    """传播到不动点：增量线性传播 + Σ|Δ| 逐边折算交替收紧。"""
    if not _propagate_inc(los, his, st.store, st.occurred, pending):
        return False
    if st.active:
        while True:
            touched = _edge_round_x(los, his, st)
            if touched is None:
                return False
            if not touched:
                break
            pend: list[int] = []
            seen = set()
            for j in touched:
                for ci in st.occurred[j]:
                    if ci not in seen:
                        seen.add(ci)
                        pend.append(ci)
            if not _propagate_inc(los, his, st.store, st.occurred, pend):
                return False
    # 精确等式（含逐段钉死后的单点域）的模素数矛盾剪枝；叶子仍由
    # 各约束包络精确核对，这里只是避免奇偶/模矛盾导致的回溯爆炸。
    if st.exact_cons and not _mod_presolve(
        los, his, st.exact_cons, _SEARCH_PRIMES
    ):
        return False
    return True


def _xspace_search(
    los: list[int],
    his: list[int],
    st: _XState,
    pending: list[int] | None = None,
) -> tuple[int, ...] | None:
    """x 空间 MRV/折半搜索；Σ|Δ| 阈值由 st 的逐边折算精确处理。

    不引入 e_k≥|Δ| 辅助变量（会把 n 个窄域变量扩成 2n-1 个且新增宽域，
    MRV 折半大量空耗）；传播用增量事件队列：每次分支只钉一个变量，
    只需重放涉及该变量的约束，而非全量重扫全部长窗。
    """
    if not _xspace_close(los, his, st, pending):
        return None
    j = _choose_variable(los, his, st.priority)
    if j == -1:
        return tuple(los)
    mid = (los[j] + his[j]) // 2
    lo2, hi2 = list(los), list(his)
    hi2[j] = mid
    result = _xspace_search(lo2, hi2, st, st.occurred[j])
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _xspace_search(lo3, hi3, st, st.occurred[j])


def _xspace_minimize_s(
    los: list[int], his: list[int], st: _XState, cap0: int,
) -> tuple[int, ...] | None:
    """在满足全部线性约束与 |Δ|≤M 的前提下，用分支定界最小化 Σ|Δ|。

    一次搜索即给出最小 S（及其见证解），替代「对 S 反复二分、每次重开
    可行性回溯」：后者在紧接最优值之下的不可行探针会各花一整棵树。
    cap 随当前最优解单调收紧；每节点先由相邻段一元域算 Σ|Δ| 的下界
    （= 各边 |Δ| 紧下界之和），下界 ≥ cap 即剪枝，再把 cap 折算成逐边
    差分约束并入增量传播。优先走低值半边（MRV + 折半）以便尽早拿到紧
    见证解。无可行解（理论上阶段 1 已排除）返回 None。
    """
    cap = cap0
    best: tuple[int, ...] | None = None

    def edge_lb_sum(los, his) -> int:
        return sum(
            _edge_abs_bounds_x(los, his, k)[0] for k in range(1, st.n)
        )

    def rec(los, his, pending) -> None:
        nonlocal cap, best
        # 1) 线性约束增量传播到不动点。
        if not _propagate_inc(los, his, st.store, st.occurred, pending):
            return
        while True:
            # 2) Σ|Δ| 下界剪枝 + 逐边 cap 折算。
            lb = edge_lb_sum(los, his)
            if lb > cap:
                return
            touched: list[int] = []
            for k in range(1, st.n):
                elb = _edge_abs_bounds_x(los, his, k)[0]
                c = cap - (lb - elb)  # |Δ_k| <= c
                if c < 0:
                    return
                nl = los[k - 1] - c
                if nl > los[k]:
                    los[k] = nl
                    touched.append(k)
                nh = his[k - 1] + c
                if nh < his[k]:
                    his[k] = nh
                    touched.append(k)
                nl = los[k] - c
                if nl > los[k - 1]:
                    los[k - 1] = nl
                    touched.append(k - 1)
                nh = his[k] + c
                if nh < his[k - 1]:
                    his[k - 1] = nh
                    touched.append(k - 1)
                if los[k] > his[k] or los[k - 1] > his[k - 1]:
                    return
            if not touched:
                break
            pend: list[int] = []
            seen = set()
            for j in touched:
                for ci in st.occurred[j]:
                    if ci not in seen:
                        seen.add(ci)
                        pend.append(ci)
            if not _propagate_inc(los, his, st.store, st.occurred, pend):
                return

        j = _choose_variable(los, his, st.priority)
        if j == -1:
            s = sum(abs(los[k] - los[k - 1]) for k in range(1, st.n))
            if s < cap or (s == cap and (best is None or tuple(los) < best)):
                cap = s
                best = tuple(los)
            return
        mid = (los[j] + his[j]) // 2
        lo2, hi2 = list(los), list(his)
        hi2[j] = mid
        rec(lo2, hi2, st.occurred[j])
        lo3, hi3 = list(los), list(his)
        lo3[j] = mid + 1
        rec(lo3, hi3, st.occurred[j])

    rec(los, his, None)
    return best


# --------------------------------------------------------------------------- #
# 反演
# --------------------------------------------------------------------------- #
def _window_constraints(n: int, lengths: list[int], windows: list[Window]) -> list[Constraint]:
    cons: list[Constraint] = []
    for win in windows:
        terms = tuple(
            (i, lengths[i]) for i in range(win.start, win.end + 1)
        )
        cons.append((terms, win.lo, win.hi))
    return cons


def _implied_range_windows(
    n: int, lengths: list[int], p_closure: list[list[int]],
    max_span: int = 1,
) -> list[Window]:
    """前缀闭包对短连续段区间 [s,e]（长度 2..max_span+1）蕴含的伸长量界。

    闭包 D[s][e+1]、D[e+1][s] 综合了全部观测窗与逐段应变界（经任意长
    路径组合），而整数搜索只直接持有提交窗。短区间蕴含界把「两个大长窗
    之差」这类紧关系直接喂给界传播，以极低代价（约 n·max_span 条短约束）
    显著压缩回溯。它们是原约束系统的必然推论（实值最紧界，ceil/floor
    后对整数赋值仍成立），加入绝不改变可行集或最优解。不把全部长区间
    都加入，是为避免事件队列在每次分支后重放过多冗余约束反而变慢。
    """
    out: list[Window] = []
    seen: set[Window] = set()
    for s in range(n):
        for e in range(s + 1, min(n, s + 1 + max_span)):
            fwd = p_closure[s][e + 1]
            bwd = p_closure[e + 1][s]
            if fwd is None or bwd is None:
                continue
            win = Window(start=s, end=e, lo=fwd, hi=-bwd)
            if win not in seen:
                seen.add(win)
                out.append(win)
    return out


def _diff_constraints(n: int, m: int) -> list[Constraint]:
    """|x_k - x_{k-1}| <= M（亦供通用传播，与差分闭包一致）。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        cons.append((((k, 1), (k - 1, -1)), -m, m))
    return cons


def _abs_lb(low: int, high: int) -> int:
    """给定一元域 [low,high]，|d| 的紧下界（0 若域跨过 0）。"""
    if low == high:
        return abs(low)
    if low >= 0:
        return low
    if high <= 0:
        return -high
    return 0


def _dspace_constraints(
    n: int, lengths: list[int], windows: list[Window],
    x_lo: list[int], x_hi: list[int],
) -> list[Constraint]:
    """差分空间 (x_0, d_1,...,d_{n-1}) 上的基础线性约束。

    x_k = x_0 + Σ_{j=1..k} d_j；观测窗以
        W_len·x_0 + Σ_j (Σ_{i∈窗,i≥j} L_i)·d_j ∈ [lo,hi]
    表达。此变量下各 d_k 域仅为 [-M,M]（M 通常很小），
    传播与回溯都远比 x 空间紧致。
    """
    cons: list[Constraint] = []
    for k in range(1, n):  # x_k 的一元界翻译到 (x0, d1..dk)
        terms = tuple([(0, 1)] + [(j, 1) for j in range(1, k + 1)])
        cons.append((terms, x_lo[k], x_hi[k]))
    for win in windows:
        total_len = sum(lengths[i] for i in range(win.start, win.end + 1))
        terms: list[tuple[int, int]] = [(0, total_len)]
        for j in range(1, n):
            coef = sum(
                lengths[i]
                for i in range(win.start, win.end + 1)
                if i >= j
            )
            if coef:
                terms.append((j, coef))
        cons.append((tuple(terms), win.lo, win.hi))
    return cons


def _dspace_search(
    los: list[int],
    his: list[int],
    constraints: list[Constraint],
    s_lo: int | None,
    s_hi: int | None,
    priority: list[tuple[int, int]] | None = None,
) -> tuple[int, ...] | None:
    """差分空间搜索；s_lo/s_hi 给出 Σ|d_k| 的闭区间（可缺省一端）。

    线性界传播与「Σ|d| 对每个 d 域的收紧」联立求不动点：
        |d_k| <= (s_hi - Σ_{j≠k}|d_j|_lb)   （上界收紧）
        |d_k| >= (s_lo - Σ_{j≠k}|d_j|_ub)   （符号已定时下界收紧）
    它会随逐段钉死不断变紧，无需求助 e_k 辅助变量，保持传播紧致。
    """
    if priority is None:
        priority = _static_priority(len(los), constraints)
    active = s_lo is not None or s_hi is not None
    again = True
    while again:
        # 差分空间只是对 x 做了可逆整数线性变换，精确窗方程的格一致性已由
        # 进入阶段 2 前的完整素数集预检覆盖；此处新增的仅为差分/绝对值
        # 不等式。故关闭每节点的模高斯消元（叶子仍由各约束包络精确核对）。
        narrowed = _propagate(los, his, constraints, mod_check=False)
        if narrowed is None:
            return None
        los, his = narrowed
        again = False
        if active:
            edge_lb = [0] * len(los)
            edge_ub = [0] * len(los)
            for k in range(1, len(los)):
                edge_lb[k] = _abs_lb(los[k], his[k])
                edge_ub[k] = max(abs(los[k]), abs(his[k]))
            total_lb = sum(edge_lb)
            total_ub = sum(edge_ub)
            if s_hi is not None and total_lb > s_hi:
                return None
            if s_lo is not None and total_ub < s_lo:
                return None
            for k in range(1, len(los)):
                if s_hi is not None:
                    cap = s_hi - (total_lb - edge_lb[k])
                    if cap < 0:
                        return None
                    new_lo = max(los[k], -cap)
                    new_hi = min(his[k], cap)
                    if new_lo > new_hi:
                        return None
                    if new_lo > los[k]:
                        los[k] = new_lo
                        again = True
                    if new_hi < his[k]:
                        his[k] = new_hi
                        again = True
                if s_lo is not None:
                    need = s_lo - (total_ub - edge_ub[k])
                    if need > 0:
                        # |d_k| >= need；跨 0 域无法用一元界表达，仅在符号
                        # 已定时收紧，否则交给后续分支。
                        if los[k] >= 0:
                            if los[k] < need:
                                los[k] = need
                                again = True
                        elif his[k] <= 0:
                            if his[k] > -need:
                                his[k] = -need
                                again = True
                        if los[k] > his[k]:
                            return None
    j = _choose_variable(los, his, priority)
    if j == -1:
        return tuple(los)
    mid = (los[j] + his[j]) // 2
    lo2, hi2 = list(los), list(his)
    hi2[j] = mid
    result = _dspace_search(lo2, hi2, constraints, s_lo, s_hi, priority)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _dspace_search(lo3, hi3, constraints, s_lo, s_hi, priority)


def invert_payload(payload: object) -> dict:
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。"""
    lengths, strain_min, strain_max, windows = validate(payload)
    n = len(lengths)
    # 重复观测窗是合法输入，但产生逐字相同的约束；求解侧去重可避免把
    # 传播/消元开销成倍放大。结果回算仍用原始 windows（含每个重复窗）。
    unique_windows: list[Window] = []
    seen_windows: set[Window] = set()
    for win in windows:
        if win not in seen_windows:
            seen_windows.add(win)
            unique_windows.append(win)
    window_cons = _window_constraints(n, lengths, unique_windows)

    # 前缀和差分闭包：与 M 无关，只算一次。
    p_closure = _prefix_closure(n, lengths, strain_min, strain_max, unique_windows)
    if p_closure is None:
        raise InfeasibleError("observation windows are mutually inconsistent")
    base_x_lo, base_x_hi = _x_bounds_from_prefix(
        n, lengths, strain_min, strain_max, p_closure
    )
    # 闭包还蕴含每个连续段区间的最紧累计伸长量界：它们综合了全部观测窗
    # （经任意长路径组合），作为额外冗余约束喂给界传播可大幅压缩回溯，
    # 且不改变可行集与三级最优解。
    implied_windows = _implied_range_windows(n, lengths, p_closure)
    implied_cons = _window_constraints(n, lengths, implied_windows)
    search_cons = window_cons + implied_cons
    # 精确等式（精确窗 + 闭包的紧蕴含区间）经整数行变换化出小系数/单位
    # 主元：等式的整数线性组合仍是等式（么模，解集不变），却让界传播
    # 强得多。作为额外冗余约束加入，供 M 二分与 x 空间阶段共享。
    exact_rows = _exact_row_hnf(n, search_cons)
    if exact_rows:
        have_cons = set(search_cons)
        search_cons = search_cons + [c for c in exact_rows if c not in have_cons]
    # 有限域预检：在任何搜索之前截杀纯整数矛盾（如精确窗模 2 奇偶冲突）。
    if not _mod_presolve(base_x_lo, base_x_hi, search_cons):
        raise InfeasibleError("observation windows are mutually inconsistent")
    full_range = strain_max - strain_min

    def solve_m(m: int) -> tuple[int, ...] | None:
        """|Δ|<=m 下找一个整数解（x 空间增量搜索）；无解返回 None。"""
        bounds = _x_diff_closure(n, base_x_lo, base_x_hi, m)
        if bounds is None:
            return None
        x_lo, x_hi = bounds
        constraints = list(search_cons)
        constraints += _diff_constraints(n, m)
        state = _XState(n, constraints, None, None)
        return _xspace_search(list(x_lo), list(x_hi), state)

    # 阶段 0：无平滑约束（M = 全量程）下的整数可行性，并取一个见证解。
    seed_sol = solve_m(full_range)
    if seed_sol is None:
        raise InfeasibleError("observation windows are mutually inconsistent")

    # M 的初始上界取见证解的实际最大相邻差；下界由无 M 约束的紧一元域给出
    # （相邻段域已不相交时 M 至少为其间隔）。把二分区间从 [0, 全量程]
    # 压到这个窄带，显著减少阶段 1 的可行性探针数。
    m_hi = max(
        abs(seed_sol[k] - seed_sol[k - 1]) for k in range(1, n)
    )
    m_lo = 0
    for k in range(1, n):
        if base_x_lo[k] > base_x_hi[k - 1]:
            m_lo = max(m_lo, base_x_lo[k] - base_x_hi[k - 1])
        if base_x_hi[k] < base_x_lo[k - 1]:
            m_lo = max(m_lo, base_x_lo[k - 1] - base_x_hi[k])
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        sol = solve_m(mid)
        if sol is not None:
            m_hi = min(
                mid,
                max(abs(sol[k] - sol[k - 1]) for k in range(1, n)),
            )
        else:
            m_lo = mid + 1
    best_m = m_lo

    # 阶段 2/3 取最优 M 下的差分闭包域，随后自适应选择求解变量空间。
    bounds = _x_diff_closure(n, base_x_lo, base_x_hi, best_m)
    assert bounds is not None
    x_lo, x_hi = bounds

    # 阶段 2/3 的变量空间自适应选择：
    #  - 最优 M 较小时，差分 d_k∈[-M,M] 极窄，用差分空间（无 e 辅助变量）；
    #  - 最优 M 很大时，差分会把本已被闭包压窄的 x 域重新放宽到 2M+1，
    #    则在 x 空间直接以相邻域包络约束 Σ|Δ|。判据为 d 域宽不超过最宽 x 域。
    max_x_width = max(x_hi[i] - x_lo[i] for i in range(n))
    use_dspace = (2 * best_m + 1) <= max_x_width

    if use_dspace:
        # 该路径最优 M 很小，d_k∈[-M,M] 极窄，S 的天然上界 (n-1)·M 已足够
        # 紧，无需热启动；沿用全量传播的差分空间搜索。
        strains = _optimize_dspace(
            n, lengths, unique_windows, x_lo, x_hi, best_m,
        )
    else:
        # 最优 M 很大：先取最优 M 下的一个可行见证解，以其实际 Σ|Δ| 作为
        # 阶段 2 的紧上界热启动，避免从 (n-1)·M 这种过宽上界二分
        # （宽上界下 Σ|Δ| 剪枝几乎失效）。
        witness = solve_m(best_m)
        assert witness is not None
        s_warm = sum(abs(witness[k] - witness[k - 1]) for k in range(1, n))
        strains = _optimize_xspace(
            n, search_cons, x_lo, x_hi, best_m, s_warm,
        )

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    best_s = sum(abs(d) for d in diffs)
    return _build_result(lengths, windows, strains, diffs, best_m, best_s)


def _optimize_dspace(
    n: int, lengths: list[int], unique_windows: list[Window],
    x_lo: list[int], x_hi: list[int], best_m: int,
) -> list[int]:
    """阶段 2/3：在差分空间 (x0,d1,..) 内最小化 Σ|d| 并钉死字典序。"""
    d_base = _dspace_constraints(n, lengths, unique_windows, x_lo, x_hi)

    def d_domains() -> tuple[list[int], list[int]]:
        d_lo = [x_lo[0]] + [-best_m] * (n - 1)
        d_hi = [x_hi[0]] + [best_m] * (n - 1)
        return d_lo, d_hi

    # 一次性的完整素数集预检：阶段 1 可行 ⇒ 此处必有模解，断言仅作防御。
    d0_lo, d0_hi = d_domains()
    if not _mod_presolve(d0_lo, d0_hi, d_base):
        raise InfeasibleError("observation windows are mutually inconsistent")

    def feasible_s(
        s_lo: int | None, s_hi: int | None,
        d_lo: list[int], d_hi: list[int],
    ) -> tuple[int, ...] | None:
        return _dspace_search(list(d_lo), list(d_hi), d_base, s_lo, s_hi)

    # 阶段 2：二分最小可行 S = Σ|Δ_k|；可行试解带回的实际 Σ|d| 可作更紧上界。
    s_lo, s_hi = 0, (n - 1) * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        d_lo0, d_hi0 = d_domains()
        sol = feasible_s(None, mid, d_lo0, d_hi0)
        if sol is not None:
            achieved = sum(abs(sol[k]) for k in range(1, n))
            s_hi = min(mid, achieved)
        else:
            s_lo = mid + 1
    best_s = s_lo

    # 阶段 3：逐段钉死字典序最小值。前缀固定后「x_k<=t」即「d_k<=t-x_{k-1}」。
    strains: list[int] = []
    d_lo, d_hi = d_domains()
    for var in range(n):
        if var == 0:
            t_lo, t_hi = x_lo[0], x_hi[0]
        else:
            prev = strains[var - 1]
            t_lo = max(x_lo[var], prev + d_lo[var])
            t_hi = min(x_hi[var], prev + d_hi[var])
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            trial_lo, trial_hi = list(d_lo), list(d_hi)
            if var == 0:
                trial_hi[0] = mid
            else:
                trial_hi[var] = mid - strains[var - 1]
            sol = feasible_s(best_s, best_s, trial_lo, trial_hi)
            if sol is not None:
                achieved = sol[0] if var == 0 else strains[var - 1] + sol[var]
                t_hi = min(mid, achieved)
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        if var == 0:
            d_lo[0] = d_hi[0] = t_lo
        else:
            d_lo[var] = d_hi[var] = t_lo - strains[var - 1]
    return strains


def _optimize_xspace(
    n: int, window_cons: list[Constraint],
    x_lo: list[int], x_hi: list[int], best_m: int,
    s_warm: int,
) -> list[int]:
    """阶段 2/3：最优 M 很大时在 x 空间直接以相邻域包络约束 Σ|Δ|。

    不使用 e_k≥|Δ| 辅助变量（会把变量数从 n 扩到 2n-1，且新增宽域变量，
    MRV 折半大量空耗），改为在每个搜索节点由相邻段一元域计算各边 |Δ|
    的紧包络，把 Σ|Δ| 预算折算成逐边收紧并经增量事件队列传播
    （见 _xspace_search/_xspace_close）。
    """
    base_constraints = list(window_cons)
    base_constraints += _diff_constraints(n, best_m)

    # 阶段 2：单次分支定界直接求出最小 S = Σ|Δ_k|（及其见证解）。
    # 旧法对 S 反复二分，而紧接最优值之下的每个不可行探针都要各跑一整棵
    # 回溯树；分支定界只维护一棵随见证解单调收紧的 cap 树（以上游见证解
    # 的实际 S=s_warm 作初始 cap），逐节点 Σ|Δ| 下界剪枝在整棵树内共享。
    st_min = _XState(n, base_constraints, None, None)
    witness_min = _xspace_minimize_s(list(x_lo), list(x_hi), st_min, s_warm)
    assert witness_min is not None
    best_s = sum(
        abs(witness_min[k] - witness_min[k - 1]) for k in range(1, n)
    )

    # 阶段 3：逐段钉死字典序最小值（Σ|Δ| 恰为 best_s）。
    st = _XState(n, base_constraints, best_s, best_s)
    lo, hi = list(x_lo), list(x_hi)
    assert _xspace_close(lo, hi, st, None)  # 已由阶段 2 保证可行
    strains: list[int] = []
    for var in range(n):
        t_lo, t_hi = lo[var], hi[var]
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            trial_lo, trial_hi = list(lo), list(hi)
            trial_hi[var] = mid  # 试探 x_var <= mid
            sol = _xspace_search(trial_lo, trial_hi, st, st.occurred[var])
            if sol is not None:
                t_hi = min(mid, sol[var])
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        lo[var] = hi[var] = t_lo
        assert _xspace_close(lo, hi, st, st.occurred[var])  # 已选最优值必可行
    return strains


def _build_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
) -> dict:
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    window_checks = []
    for idx, win in enumerate(windows):
        weighted_sum = 0
        for i in range(win.start, win.end + 1):
            weighted_sum += lengths[i] * strains[i]
        total_length = prefix[win.end + 1] - prefix[win.start]
        window_checks.append(
            {
                "index": idx,
                "start_segment": win.start + 1,
                "end_segment": win.end + 1,
                "total_length": total_length,
                "min_elongation": win.lo,
                "max_elongation": win.hi,
                "weighted_strain_sum": weighted_sum,
                "satisfied": win.lo <= weighted_sum <= win.hi,
            }
        )
    # 两级平滑指标均由相邻差直接复核（重算以自证，不直接采用搜索内部值）。
    recomputed_m = max(abs(d) for d in diffs)
    recomputed_s = sum(abs(d) for d in diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    return {
        "segment_count": len(lengths),
        "strains": strains,
        "adjacent_diffs": diffs,
        "objectives": {
            "max_adjacent_diff": recomputed_m,
            "sum_adjacent_abs_diff": recomputed_s,
        },
        "window_checks": window_checks,
        "criteria_order": [
            "max_adjacent_diff",
            "sum_adjacent_abs_diff",
            "lexicographic",
        ],
    }
