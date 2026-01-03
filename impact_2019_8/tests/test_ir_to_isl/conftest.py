"""ir_to_islテスト用の共通ヘルパー."""

from ir_types import (
    Access,
    BinaryOp,
    Compare,
    Compute,
    Domain,
    IntConst,
    Iterator,
    Load,
    PrimFunc,
    Schedule,
    Store,
    Tensor,
    Var,
)


def make_simple_func(
    iterators: tuple[Iterator, ...],
    params: tuple[str, ...],
    constraints: tuple[Compare, ...] = (),
) -> PrimFunc:
    """テスト用のシンプルなPrimFuncを生成する."""
    if params:
        shape = tuple(Var(p) for p in params)
    else:
        shape = tuple(IntConst(4) for _ in iterators)

    a = Tensor(name="A", shape=shape)
    b = Tensor(name="B", shape=shape)
    out = Tensor(name="C", shape=shape)

    index = tuple(Var(it.name) for it in iterators)

    return PrimFunc(
        name="kernel",
        params=(a, b, out),
        computes=(
            Compute(
                name="S",
                domain=Domain(
                    params=params,
                    iterators=iterators,
                    constraints=constraints,
                ),
                body=Store(
                    access=Access(tensor=out, index=index),
                    value=BinaryOp(
                        op="Add",
                        lhs=Load(access=Access(tensor=a, index=index)),
                        rhs=Load(access=Access(tensor=b, index=index)),
                    ),
                ),
            ),
        ),
        schedule=Schedule(loop_order=tuple(it.name for it in iterators)),
    )
