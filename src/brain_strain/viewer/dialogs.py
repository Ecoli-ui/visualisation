"""Cross-platform native file dialogs used by the interactive viewer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_macos_dialog(script: str, *arguments: str, description: str) -> str:
    result = subprocess.run(
        [
            "/usr/bin/osascript",
            "-l",
            "JavaScript",
            "-e",
            script,
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown macOS scripting error"
        raise RuntimeError(f"Could not open the {description}: {detail}")
    return result.stdout.strip()


def choose_open_file(
    initial_directory: str | Path,
    *,
    title: str,
    pattern: str,
) -> str:
    """Choose an existing file, returning an empty string when cancelled."""
    directory = Path(initial_directory).expanduser().resolve()
    if sys.platform == "darwin":
        script = """
function run(argv) {
    const app = Application.currentApplication();
    app.includeStandardAdditions = true;
    app.activate();
    try {
        const selectedFile = app.chooseFile({
            withPrompt: argv[1],
            defaultLocation: Path(argv[0])
        });
        return selectedFile.toString();
    } catch (error) {
        if (error.errorNumber === -128) {
            return "";
        }
        throw error;
    }
}
"""
        return _run_macos_dialog(
            script,
            str(directory),
            title,
            description="macOS file chooser",
        )

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "The local Python installation does not provide Tk"
        ) from exc

    root: Any | None = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
            root.update_idletasks()
        except tk.TclError:
            pass
        return str(
            filedialog.askopenfilename(
                parent=root,
                title=title,
                initialdir=str(directory),
                filetypes=(("Supported mesh files", pattern),),
            )
        )
    except tk.TclError as exc:
        raise RuntimeError(f"Could not open the local file chooser: {exc}") from exc
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


def choose_save_file(
    initial_directory: str | Path,
    *,
    title: str,
    default_name: str,
    extension: str,
    file_type_name: str,
) -> str:
    """Choose an output filename, returning an empty string when cancelled."""
    directory = Path(initial_directory).expanduser().resolve()
    if sys.platform == "darwin":
        script = """
function run(argv) {
    const app = Application.currentApplication();
    app.includeStandardAdditions = true;
    app.activate();
    try {
        const selectedFile = app.chooseFileName({
            withPrompt: argv[2],
            defaultLocation: Path(argv[0]),
            defaultName: argv[1]
        });
        return selectedFile.toString();
    } catch (error) {
        if (error.errorNumber === -128) {
            return "";
        }
        throw error;
    }
}
"""
        return _run_macos_dialog(
            script,
            str(directory),
            default_name,
            title,
            description="workbook save dialog",
        )

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "The local Python installation does not provide Tk"
        ) from exc

    root: Any | None = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
            root.update_idletasks()
        except tk.TclError:
            pass
        return str(
            filedialog.asksaveasfilename(
                parent=root,
                title=title,
                initialdir=str(directory),
                initialfile=default_name,
                defaultextension=extension,
                filetypes=((file_type_name, f"*{extension}"),),
            )
        )
    except tk.TclError as exc:
        raise RuntimeError(f"Could not open the workbook save dialog: {exc}") from exc
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


__all__ = ["choose_open_file", "choose_save_file"]

