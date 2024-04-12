from __future__ import annotations

from os import getenv

import PyInstaller.__main__
from gooey import Gooey, GooeyParser

from python.main import anonymize_excel


@Gooey  # type: ignore[misc]
def main_gui() -> None:
    parser = GooeyParser(description="Choose Folder")
    parser.add_argument(
        "--input_folder_name",
        widget="DirChooser",
        gooey_options={
            "wildcard": "Excel files (*.xls;*.xlsx)|*.xls;*.xlsx|"
            "JSON files (*.json)|*.json",
            "message": "Pick folder",
            "default_path": "c:/batch/stuff",
        },
    )

    args = parser.parse_args()
    anonymize_excel(
        args.input_folder_name,
    )


def main() -> None:
    if getenv("STAGING"):
        main_gui()
    else:
        PyInstaller.__main__.run(
            [
                "main.py",
                "--distpath",
                "tmp/dist",
                "--name",
                "anonymize-excel",
                "--runtime-hook",
                "prm/runtime_hook.py",
                "--onefile",
                "--specpath",
                "tmp/spec",
                "--windowed",
                "--workpath",
                "tmp/build",
            ],
        )


if __name__ == "__main__":
    main()
