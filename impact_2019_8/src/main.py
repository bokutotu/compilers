from llvmlite import binding, ir


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


def compile_and_run(module: ir.Module) -> int:
    """LLVM IRをコンパイルして実行する"""
    binding.initialize_all_targets()
    binding.initialize_native_asmprinter()

    llvm_ir = str(module)
    print("=== 生成されたLLVM IR ===")
    print(llvm_ir)
    print("=" * 30)

    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = binding.parse_assembly("")
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)

    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address("return_one")

    import ctypes

    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
    result = cfunc()

    return result


def main():
    module = create_return_one_function()
    result = compile_and_run(module)
    print(f"return_one() の実行結果: {result}")


if __name__ == "__main__":
    main()
