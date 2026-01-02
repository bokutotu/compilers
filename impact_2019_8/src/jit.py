import ctypes
import os
import subprocess
import tempfile
from typing import Any


class JITCompiler:
    """CコードをJITコンパイルして実行するクラス"""

    def __init__(self):
        self._lib = None
        self._lib_path = None
        self._c_code = None

    def compile(self, c_code: str):
        """Cコードをclangでコンパイルして共有ライブラリを作成する"""
        self._c_code = c_code

        # 一時ディレクトリを作成
        self._tmpdir = tempfile.mkdtemp()
        c_file = os.path.join(self._tmpdir, "code.c")

        # プラットフォームに応じた共有ライブラリの拡張子
        lib_ext = ".dylib" if os.uname().sysname == "Darwin" else ".so"

        self._lib_path = os.path.join(self._tmpdir, f"code{lib_ext}")

        # Cコードを一時ファイルに書き込む
        with open(c_file, "w") as f:
            f.write(c_code)

        try:
            # clangで共有ライブラリをコンパイル
            subprocess.run(
                [
                    "clang",
                    "-shared",
                    "-fPIC",
                    "-O3",
                    "-o",
                    self._lib_path,
                    c_file,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            self._cleanup()
            raise RuntimeError(f"Compilation failed: {e.stderr}") from e

        # 共有ライブラリをロード
        self._lib = ctypes.CDLL(self._lib_path)

    def get_function(
        self,
        func_name: str,
        restype: type[ctypes._SimpleCData] | None = None,
        argtypes: list[type[ctypes._SimpleCData]] | None = None,
    ) -> ctypes._CFuncPtr:
        """コンパイル済み関数を取得する"""
        if self._lib is None:
            raise RuntimeError("No code has been compiled yet")

        if restype is None:
            restype = ctypes.c_int
        if argtypes is None:
            argtypes = []

        func = getattr(self._lib, func_name)
        func.restype = restype
        func.argtypes = argtypes
        return func

    def run(
        self,
        func_name: str,
        restype: type[ctypes._SimpleCData] | None = None,
        argtypes: list[type[ctypes._SimpleCData]] | None = None,
        args: list[Any] | None = None,
    ) -> Any:
        """コンパイル済み関数を実行する"""
        func = self.get_function(func_name, restype, argtypes)
        if args is None:
            args = []
        return func(*args)

    def _cleanup(self):
        """一時ファイルを削除する"""
        if self._lib_path and os.path.exists(self._lib_path):
            os.unlink(self._lib_path)
        if hasattr(self, "_tmpdir") and os.path.exists(self._tmpdir):
            c_file = os.path.join(self._tmpdir, "code.c")
            if os.path.exists(c_file):
                os.unlink(c_file)
            os.rmdir(self._tmpdir)

    def __del__(self):
        """デストラクタで一時ファイルをクリーンアップ"""
        self._cleanup()
