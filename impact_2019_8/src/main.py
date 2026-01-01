from llvmlite import ir

from jit import JITCompiler


def create_return_one_function():
    """1を返すだけの超簡単なLLVM IR関数を作成する"""
    module = ir.Module(name="simple_module")
    func_type = ir.FunctionType(ir.IntType(32), [])
    func = ir.Function(module, func_type, name="return_one")
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    one = ir.Constant(ir.IntType(32), 1)
    builder.ret(one)
    return module


def main():
    module = create_return_one_function()

    compiler = JITCompiler()
    compiler.print_ir(module)
    compiler.compile(module)
    result = compiler.run("return_one")

    print(f"return_one() の実行結果: {result}")


if __name__ == "__main__":
    main()
