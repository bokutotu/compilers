from ast_parser import parse_isl_ast
from ir_to_isl_new import build_domain_and_schedule
from ir_types_new import Axis, Compute, Domain, PrimFunc, Schedule, Tensor
from isl_ast import build_ast_from_domain_and_schedule
from codegen import isl_ast_to_c


def main():
    # 1. IRを定義
    a = Tensor("A", (10,))
    b = Tensor("B", (10,))
    c = Tensor("C", (10,))

    domain = Domain((Axis("i", 10),))
    schedule = Schedule(("i",))

    compute = Compute(
        name="S",
        op="add",
        a=a,
        b=b,
        out=c,
        domain=domain,
    )

    func = PrimFunc(
        name="add_func",
        compute=compute,
        schedule=schedule,
        params=(a, b, c),
    )

    # 2. ドメインとスケジュールからISL ASTを生成
    isl_domain, isl_schedule = build_domain_and_schedule(func)
    ast = build_ast_from_domain_and_schedule(isl_domain, isl_schedule)

    # 3. AST文字列をパースしてForLoopオブジェクトに変換
    parsed_ast = parse_isl_ast(str(ast))

    # 4. ForLoopからCコードを生成
    c_code = isl_ast_to_c(parsed_ast, domain_exprs={compute.name: compute})

    # 5. Cコードを出力
    print(c_code)


if __name__ == "__main__":
    main()
