from llvmlite import binding, ir


class JITCompiler:
    """LLVM IRをJITコンパイルして実行するクラス"""

    def __init__(self):
        self._engine = None
        self._initialize_llvm()

    def _initialize_llvm(self):
        """LLVMバックエンドを初期化する"""
        binding.initialize_all_targets()
        binding.initialize_native_asmprinter()

    def compile(self, module: ir.Module):
        """LLVM IRをコンパイルしてJITエンジンを作成する"""
        llvm_ir = str(module)
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()

        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        backing_mod = binding.parse_assembly("")
        self._engine = binding.create_mcjit_compiler(backing_mod, target_machine)

        self._engine.add_module(mod)
        self._engine.finalize_object()
        self._engine.run_static_constructors()

    def run(self, func_name: str, restype=None, argtypes=None):
        """コンパイル済み関数を実行する"""
        import ctypes

        if restype is None:
            restype = ctypes.c_int
        if argtypes is None:
            argtypes = []

        func_ptr = self._engine.get_function_address(func_name)
        cfunc = ctypes.CFUNCTYPE(restype, *argtypes)(func_ptr)
        return cfunc()

    @staticmethod
    def print_ir(module: ir.Module):
        """LLVM IRを表示する（デバッグ用）"""
        print("=== 生成されたLLVM IR ===")
        print(module)
        print("=" * 30)
