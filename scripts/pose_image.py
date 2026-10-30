"""识别单图或目录，并将完整人体姿态导出为 JSON。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gem.inference.pose_image_cli import main

if __name__ == "__main__":
    sys.exit(main())
