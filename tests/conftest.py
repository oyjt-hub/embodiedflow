"""测试必须完全隔离：强制使用临时目录，绝不碰生产库。"""
import os
import pathlib
import tempfile

_tmp = pathlib.Path(tempfile.mkdtemp(prefix="ef_test_"))
# 注意：这里用赋值而不是 setdefault —— 否则外部导出的 EF_DATA 会让测试写进生产库
os.environ["EF_ROOT"] = str(_tmp / "proj")
os.environ["EF_DATA"] = str(_tmp / "data")
os.environ["EF_DB"] = str(_tmp / "data" / "metadata" / "platform.db")
