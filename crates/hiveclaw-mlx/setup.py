from setuptools import setup
from mlx.extension import CMakeExtension, CMakeBuild

setup(
    name="hiveclaw_mlx_ext",
    version="0.1.0",
    ext_modules=[CMakeExtension("hiveclaw_mlx_ext", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
)
