#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to clean up orphaned temporary directories created by PoliwhiRL.
This script helps resolve the temporary file accumulation issue on macOS.
"""

import os
import shutil
import tempfile
import time
import argparse
import psutil


def find_poliwhirl_temp_dirs():
    """Find all PoliwhiRL temporary directories"""
    temp_base = tempfile.gettempdir()
    poliwhirl_dirs = []

    try:
        for item in os.listdir(temp_base):
            if item.startswith(
                ("poliwhirl_", "tmp", "poliwhirl_shared_", "poliwhirl_env_")
            ) and os.path.isdir(os.path.join(temp_base, item)):
                full_path = os.path.join(temp_base, item)
                poliwhirl_dirs.append(full_path)
    except PermissionError:
        print(f"Permission denied accessing {temp_base}")

    return poliwhirl_dirs


def is_directory_in_use(dir_path):
    """Check if a directory is currently being used by any process"""
    try:
        for proc in psutil.process_iter(["pid", "name", "open_files"]):
            try:
                if proc.info["open_files"]:
                    for file_info in proc.info["open_files"]:
                        if file_info.path.startswith(dir_path):
                            return True, proc.info["name"], proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"Error checking process usage: {e}")

    return False, None, None


def get_directory_age(dir_path):
    """Get the age of a directory in hours"""
    try:
        stat = os.stat(dir_path)
        age_seconds = time.time() - stat.st_mtime
        return age_seconds / 3600  # Convert to hours
    except OSError:
        return 0


def get_directory_size(dir_path):
    """Get the size of a directory in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass
    except OSError:
        pass

    return total_size / (1024 * 1024)  # Convert to MB


def cleanup_temp_directories(dry_run=True, min_age_hours=1, force=False):
    """Clean up PoliwhiRL temporary directories"""
    temp_dirs = find_poliwhirl_temp_dirs()

    if not temp_dirs:
        print("No PoliwhiRL temporary directories found.")
        return

    print(f"Found {len(temp_dirs)} potential PoliwhiRL temporary directories:")
    print("-" * 80)

    total_size = 0
    cleaned_count = 0

    # Also check for stale PyBoy processes
    pyboy_processes = []
    for proc in psutil.process_iter(["pid", "name", "create_time"]):
        try:
            if (
                "pyboy" in proc.info["name"].lower()
                or "python" in proc.info["name"].lower()
            ):
                age_hours = (time.time() - proc.info["create_time"]) / 3600
                if age_hours > min_age_hours:
                    pyboy_processes.append(
                        (proc.info["pid"], proc.info["name"], age_hours)
                    )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if pyboy_processes:
        print(
            "\nFound {} potentially stale PyBoy/Python processes:".format(
                len(pyboy_processes)
            )
        )
        for pid, name, age in pyboy_processes:
            print(f"  PID {pid}: {name} (age: {age:.1f} hours)")
        print()

    for dir_path in temp_dirs:
        age_hours = get_directory_age(dir_path)
        size_mb = get_directory_size(dir_path)
        total_size += size_mb

        in_use, proc_name, proc_pid = is_directory_in_use(dir_path)

        print(f"Directory: {dir_path}")
        print(f"  Age: {age_hours:.1f} hours")
        print(f"  Size: {size_mb:.1f} MB")

        if in_use:
            print(f"  Status: IN USE by {proc_name} (PID: {proc_pid})")
        elif age_hours < min_age_hours and not force:
            print(f"  Status: TOO RECENT (< {min_age_hours} hours)")
        else:
            if dry_run:
                print("  Status: WOULD BE DELETED")
            else:
                try:
                    shutil.rmtree(dir_path)
                    print("  Status: DELETED")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  Status: ERROR DELETING - {e}")

        print()

    print("-" * 80)
    print(f"Total size of temporary directories: {total_size:.1f} MB")

    if dry_run:
        eligible_count = sum(
            1
            for dir_path in temp_dirs
            if get_directory_age(dir_path) >= min_age_hours
            and not is_directory_in_use(dir_path)[0]
        )
        print(f"Directories that would be cleaned: {eligible_count}")
        print("\nThis was a dry run. Use --execute to actually delete directories.")
    else:
        print(f"Directories cleaned: {cleaned_count}")

    # Clean up lock files
    lock_files_cleaned = 0
    for dir_path in [tempfile.gettempdir(), os.getcwd()]:
        try:
            for item in os.listdir(dir_path):
                if item.endswith(".lock"):
                    lock_path = os.path.join(dir_path, item)
                    if get_directory_age(lock_path) > min_age_hours:
                        if not dry_run:
                            try:
                                os.remove(lock_path)
                                lock_files_cleaned += 1
                            except OSError:
                                pass
                        else:
                            lock_files_cleaned += 1
        except OSError:
            pass

    if lock_files_cleaned > 0:
        action = "would be cleaned" if dry_run else "cleaned"
        print("\nLock files {}: {}".format(action, lock_files_cleaned))


def main():
    parser = argparse.ArgumentParser(
        description="Clean up orphaned PoliwhiRL temporary directories"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete directories (default is dry run)",
    )
    parser.add_argument(
        "--min-age",
        type=float,
        default=1.0,
        help="Minimum age in hours before deletion (default: 1.0)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force deletion regardless of age"
    )

    args = parser.parse_args()

    print("PoliwhiRL Temporary Directory Cleanup Utility")
    print("=" * 50)

    if args.execute:
        print("WARNING: This will permanently delete temporary directories!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Cleanup cancelled.")
            return

    cleanup_temp_directories(
        dry_run=not args.execute, min_age_hours=args.min_age, force=args.force
    )


if __name__ == "__main__":
    main()
