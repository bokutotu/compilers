from __future__ import annotations

import islpy as isl

from ir_types import (
    Access,
    BinaryOp,
    Block,
    Call,
    Compare,
    Compute,
    Constraint,
    Expr,
    FloatConst,
    IntConst,
    Load,
    Logical,
    PrimFunc,
    ReduceStore,
    Stmt,
    Store,
    UnaryOp,
    Var,
)

# ==========================================
# 1. AST -> ISL文字列 変換 (Visitor)
# ==========================================


def expr_to_isl(expr: Expr) -> str:
    """式ASTをISL形式の文字列に変換"""
    if isinstance(expr, IntConst):
        return str(expr.value)

    if isinstance(expr, FloatConst):
        # ISLは整数/有理数セットのため、浮動小数は原則として整数化するか、
        # あるいはコンテキストに応じて文字列化します。
        return str(int(expr.value))

    if isinstance(expr, Var):
        return expr.name

    if isinstance(expr, BinaryOp):
        lhs = expr_to_isl(expr.lhs)
        rhs = expr_to_isl(expr.rhs)

        match expr.op:
            case "Add":
                return f"({lhs} + {rhs})"
            case "Sub":
                return f"({lhs} - {rhs})"
            case "Mul":
                return f"({lhs} * {rhs})"
            # ISLの除算は通常 floor(x/y) ですが、明示的にfloorを使います
            case "Div":
                return f"floor({lhs} / {rhs})"
            case "FloorDiv":
                return f"floor({lhs} / {rhs})"
            case "Mod":
                return f"({lhs} % {rhs})"
            case "Max":
                return f"max({lhs}, {rhs})"
            case "Min":
                return f"min({lhs}, {rhs})"
            case _:
                raise ValueError(f"Unsupported binary op: {expr.op}")

    if isinstance(expr, UnaryOp):
        operand = expr_to_isl(expr.operand)
        if expr.op == "Neg":
            return f"-{operand}"
        if expr.op == "Not":
            return f"not {operand}"

    if isinstance(expr, Call):
        args = ", ".join(expr_to_isl(arg) for arg in expr.args)
        return f"{expr.name}({args})"

    if isinstance(expr, Load):
        # 注意: ISLの制約式内にメモリロードを含めることはできません。
        # ここではアクセス解析用にインデックスを生成する目的でのみ使用を想定し、
        # 文字列表現を返しますが、Setの定義に使うとエラーになる可能性があります。
        idx = ", ".join(expr_to_isl(i) for i in expr.access.index)
        return f"{expr.access.tensor.name}[{idx}]"

    raise TypeError(f"Unknown expression type: {type(expr)}")


def constraint_to_isl(constraint: Constraint) -> str:
    """制約ASTをISL形式の文字列に変換"""
    if isinstance(constraint, Compare):
        lhs = expr_to_isl(constraint.lhs)
        rhs = expr_to_isl(constraint.rhs)
        op_map = {"LT": "<", "LE": "<=", "GT": ">", "GE": ">=", "EQ": "=", "NE": "!="}
        return f"{lhs} {op_map[constraint.op]} {rhs}"

    if isinstance(constraint, Logical):
        lhs = constraint_to_isl(constraint.lhs)
        rhs = constraint_to_isl(constraint.rhs)
        match constraint.op:
            case "And":
                return f"({lhs} and {rhs})"
            case "Or":
                return f"({lhs} or {rhs})"

    raise TypeError(f"Unknown constraint type: {type(constraint)}")


def _build_header(compute: Compute) -> tuple[str, str, str]:
    """[Params] -> { Name[Iters] : Constraints } の各パーツを生成"""
    domain = compute.domain

    # パラメータ: [N, M]
    param_str = f"[{', '.join(domain.params)}]" if domain.params else "[]"

    # タプル: S[i, j]
    iter_names = [it.name for it in domain.iterators]
    tuple_str = f"{compute.name}[{', '.join(iter_names)}]"

    # 制約
    if domain.constraints:
        const_parts = [constraint_to_isl(c) for c in domain.constraints]
        const_str = " and ".join(const_parts)
    else:
        const_str = "1 = 1"

    return param_str, tuple_str, const_str


def build_domain(func: PrimFunc, ctx: isl.Context | None = None) -> isl.UnionSet:
    """
    計算領域 (Iteration Domain) を構築
    ISL Set: [Params] -> { Stmt[iters] : constraints }
    """
    ctx = ctx or isl.Context()
    u_set = isl.UnionSet("{ }", ctx)

    for compute in func.computes:
        param_str, tuple_str, const_str = _build_header(compute)
        isl_str = f"{param_str} -> {{ {tuple_str} : {const_str} }}"
        try:
            dom = isl.UnionSet(isl_str, ctx)
            u_set = u_set.union(dom)
        except isl.Error as e:
            raise RuntimeError(f"Failed to build ISL domain: {isl_str}") from e

    return u_set


def build_schedule(func: PrimFunc, ctx: isl.Context | None = None) -> isl.UnionMap:
    """
    スケジュールマップを構築
    ISL Map: [Params] -> { Stmt[iters] -> [time_dims] : constraints }

    複数のComputeがある場合、末尾にstatement IDを追加して実行順序を明示:
    - 1つのCompute: S[i] -> [i]
    - 複数のCompute: S1[i] -> [i, 0], S2[j] -> [j, 1]

    stmt_idを末尾に置くことで、同じイテレータを持つループは融合可能になる。
    """
    ctx = ctx or isl.Context()
    u_map = isl.UnionMap("{ }", ctx)

    global_loop_order = func.schedule.loop_order
    add_stmt_id = len(func.computes) >= 2

    for stmt_id, compute in enumerate(func.computes):
        param_str, src_tuple_str, const_str = _build_header(compute)

        # このドメインに含まれるイテレータのみを、global_loop_orderの順序で抽出
        domain_iters = {it.name for it in compute.domain.iterators}
        sched_dims = [v for v in global_loop_order if v in domain_iters]

        # 複数Computeの場合、末尾にstmt_idを追加（ループ融合を可能にするため）
        if add_stmt_id:
            dst_tuple_str = f"[{', '.join(sched_dims)}, {stmt_id}]"
        else:
            dst_tuple_str = f"[{', '.join(sched_dims)}]"
        isl_str = (
            f"{param_str} -> {{ {src_tuple_str} -> {dst_tuple_str} : {const_str} }}"
        )

        try:
            m = isl.UnionMap(isl_str, ctx)
            u_map = u_map.union(m)
        except isl.Error as e:
            raise RuntimeError(f"Failed to build ISL schedule: {isl_str}") from e

    return u_map


def _collect_accesses(stmt: Stmt) -> list[tuple[Access, Constraint | None, bool]]:
    """
    文をトラバースしてアクセス情報を収集
    Returns: list of (Access, Predicate, is_write)
    """
    results = []

    def visit(s: Stmt, preds: list[Constraint]):
        # 現在の述語(条件)を構築
        curr_pred = None
        if preds:
            curr_pred = preds[0]
            for p in preds[1:]:
                curr_pred = Logical("And", curr_pred, p)

        if isinstance(s, Block):
            for child in s.stmts:
                visit(child, preds)

        elif isinstance(s, Store):
            # Write Access
            # Store固有のpredicateがあれば追加
            write_preds = list(preds)
            if s.predicate:
                write_preds.append(s.predicate)

            # Write Predicate結合
            w_pred_obj = curr_pred
            if s.predicate:
                w_pred_obj = (
                    Logical("And", curr_pred, s.predicate) if curr_pred else s.predicate
                )

            results.append((s.access, w_pred_obj, True))

            # Read Access (RHS & Index)
            # Readは「文が実行される条件」下で発生
            _collect_expr_reads(s.value, results, w_pred_obj)
            for idx in s.access.index:
                _collect_expr_reads(idx, results, w_pred_obj)

        elif isinstance(s, ReduceStore):
            # Reduceは Read-Modify-Write
            results.append((s.access, curr_pred, True))  # Write
            results.append((s.access, curr_pred, False))  # Read (self)

            _collect_expr_reads(s.value, results, curr_pred)
            if s.init:
                _collect_expr_reads(s.init, results, curr_pred)
            for idx in s.access.index:
                _collect_expr_reads(idx, results, curr_pred)

    visit(stmt, [])
    return results


def _collect_expr_reads(expr: Expr, acc_list: list, pred: Constraint | None):
    """式中のLoadを再帰的に収集"""
    if isinstance(expr, Load):
        acc_list.append((expr.access, pred, False))
        for idx in expr.access.index:
            _collect_expr_reads(idx, acc_list, pred)
    elif isinstance(expr, BinaryOp):
        _collect_expr_reads(expr.lhs, acc_list, pred)
        _collect_expr_reads(expr.rhs, acc_list, pred)
    elif isinstance(expr, UnaryOp):
        _collect_expr_reads(expr.operand, acc_list, pred)
    elif isinstance(expr, Call):
        for arg in expr.args:
            _collect_expr_reads(arg, acc_list, pred)


def _build_access_map_generic(
    func: PrimFunc, want_write: bool, ctx: isl.Context
) -> isl.UnionMap:
    u_map = isl.UnionMap("{ }", ctx)

    for compute in func.computes:
        param_str, src_tuple_str, domain_const_str = _build_header(compute)
        accesses = _collect_accesses(compute.body)

        for access, pred, is_write in accesses:
            if is_write != want_write:
                continue

            try:
                # インデックス式文字列化
                idx_str = ", ".join(expr_to_isl(i) for i in access.index)
                tensor_acc = f"{access.tensor.name}[{idx_str}]"

                # 制約結合
                full_const = domain_const_str
                if pred:
                    pred_str = constraint_to_isl(pred)
                    full_const = f"({full_const}) and ({pred_str})"

                isl_str = (
                    f"{param_str} -> {{ {src_tuple_str} -> {tensor_acc} : "
                    f"{full_const} }}"
                )
                m = isl.UnionMap(isl_str, ctx)
                u_map = u_map.union(m)
            except (ValueError, isl.Error):
                # 非アフィンなインデックス等で生成できない場合はスキップ
                continue

    return u_map


def build_write_access(func: PrimFunc, ctx: isl.Context | None = None) -> isl.UnionMap:
    return _build_access_map_generic(func, True, ctx or isl.Context())


def build_read_access(func: PrimFunc, ctx: isl.Context | None = None) -> isl.UnionMap:
    return _build_access_map_generic(func, False, ctx or isl.Context())


# ==========================================
# 3. 依存関係解析 (Dependence Analysis)
# ==========================================


def compute_raw_dependence(
    schedule: isl.UnionMap,
    write_access: isl.UnionMap,
    read_access: isl.UnionMap,
) -> isl.UnionMap:
    """
    RAW (Read After Write) 依存関係を計算
    フロー依存: 書き込み → 読み込み (同じ配列要素、書き込みが先)

    Returns: { S_write[...] -> S_read[...] }
    """
    # 同じ配列要素へのアクセスペア: S_write -> S_read
    same_access = write_access.apply_range(read_access.reverse())

    # 時間順序: 書き込みが読み込みより前
    before = schedule.lex_lt_union_map(schedule)

    return same_access.intersect(before)


def compute_war_dependence(
    schedule: isl.UnionMap,
    write_access: isl.UnionMap,
    read_access: isl.UnionMap,
) -> isl.UnionMap:
    """
    WAR (Write After Read) 依存関係を計算
    反依存: 読み込み → 書き込み (同じ配列要素、読み込みが先)

    Returns: { S_read[...] -> S_write[...] }
    """
    # 同じ配列要素へのアクセスペア: S_read -> S_write
    same_access = read_access.apply_range(write_access.reverse())

    # 時間順序: 読み込みが書き込みより前
    before = schedule.lex_lt_union_map(schedule)

    return same_access.intersect(before)


def compute_waw_dependence(
    schedule: isl.UnionMap,
    write_access: isl.UnionMap,
) -> isl.UnionMap:
    """
    WAW (Write After Write) 依存関係を計算
    出力依存: 書き込み → 書き込み (同じ配列要素)

    Returns: { S_write1[...] -> S_write2[...] }
    """
    # 同じ配列要素への書き込みペア
    same_access = write_access.apply_range(write_access.reverse())

    # 時間順序: 最初の書き込みが後の書き込みより前
    # lex_lt は厳密な「より小さい」なので、自己ループ (S[i] -> S[i]) は
    # schedule(S[i]) < schedule(S[i]) が偽となり自動的に除外される
    before = schedule.lex_lt_union_map(schedule)

    return same_access.intersect(before)


def compute_all_dependences(
    func: PrimFunc,
    ctx: isl.Context | None = None,
) -> dict[str, isl.UnionMap]:
    """
    すべての依存関係を計算

    Returns: {"RAW": ..., "WAR": ..., "WAW": ...}
    """
    ctx = ctx or isl.Context()
    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)
    read_access = build_read_access(func, ctx)

    return {
        "RAW": compute_raw_dependence(schedule, write_access, read_access),
        "WAR": compute_war_dependence(schedule, write_access, read_access),
        "WAW": compute_waw_dependence(schedule, write_access),
    }
