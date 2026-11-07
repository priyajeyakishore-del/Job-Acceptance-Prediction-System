
# ============================================================
# MAIN EDA PIPELINE
# Project: Job Acceptance Prediction System
# Purpose: Execute all EDA scripts sequentially
# ============================================================

from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EDA_DIR = Path(__file__).resolve().parent

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_data_analysis"
    / "reports"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = REPORT_DIR / "eda_execution_log.txt"


# ------------------------------------------------------------
# 2. EDA SCRIPT EXECUTION ORDER
# ------------------------------------------------------------

EDA_SCRIPTS = [

    "dataset_overview.py",

    "data_quality_analysis.py",

    "target_analysis.py",

    "univariate_numerical_analysis.py",

    "univariate_categorical_analysis.py",

    "bivariate_numerical_analysis.py",

    "bivariate_categorical_analysis.py",

    "multivariate_analysis.py",

    "key_insights.py"

]


# ------------------------------------------------------------
# 3. EXECUTE INDIVIDUAL EDA SCRIPT
# ------------------------------------------------------------

def execute_script(script_name):

    script_path = EDA_DIR / script_name

    print("\n" + "=" * 75)
    print(f"RUNNING: {script_name}")
    print("=" * 75)

    if not script_path.exists():

        print(f"ERROR: Script not found: {script_path}")

        return {
            "script": script_name,
            "status": "FAILED",
            "duration": 0,
            "message": "Script file not found"
        }

    start_time = time.time()

    try:

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        duration = time.time() - start_time

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("\nERROR / WARNING OUTPUT:")
            print(result.stderr)

        if result.returncode == 0:

            print(f"\nSUCCESS: {script_name}")
            print(f"Execution time: {duration:.2f} seconds")

            return {
                "script": script_name,
                "status": "SUCCESS",
                "duration": duration,
                "message": "Executed successfully"
            }

        else:

            print(f"\nFAILED: {script_name}")
            print(f"Return code: {result.returncode}")

            return {
                "script": script_name,
                "status": "FAILED",
                "duration": duration,
                "message": f"Return code: {result.returncode}"
            }

    except Exception as error:

        duration = time.time() - start_time

        print(f"\nEXCEPTION: {error}")

        return {
            "script": script_name,
            "status": "FAILED",
            "duration": duration,
            "message": str(error)
        }


# ------------------------------------------------------------
# 4. SAVE EXECUTION LOG
# ------------------------------------------------------------

def save_execution_log(results, total_duration):

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("JOB ACCEPTANCE PREDICTION SYSTEM\n")
        file.write("EDA PIPELINE EXECUTION REPORT\n")
        file.write("=" * 75 + "\n\n")

        file.write(
            f"Execution date: {datetime.now()}\n"
        )

        file.write(
            f"Total scripts: {len(results)}\n"
        )

        successful = sum(
            result["status"] == "SUCCESS"
            for result in results
        )

        failed = len(results) - successful

        file.write(f"Successful scripts: {successful}\n")
        file.write(f"Failed scripts: {failed}\n")

        file.write(
            f"Total execution time: {total_duration:.2f} seconds\n\n"
        )

        file.write("SCRIPT EXECUTION DETAILS\n")
        file.write("-" * 75 + "\n")

        for result in results:

            file.write(
                f"\nScript: {result['script']}\n"
            )

            file.write(
                f"Status: {result['status']}\n"
            )

            file.write(
                f"Duration: {result['duration']:.2f} seconds\n"
            )

            file.write(
                f"Message: {result['message']}\n"
            )

    print(f"\nExecution log saved at:\n{LOG_PATH}")


# ------------------------------------------------------------
# 5. MAIN EDA PIPELINE
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 75)
    print("JOB ACCEPTANCE PREDICTION SYSTEM")
    print("COMPLETE EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 75)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"EDA directory: {EDA_DIR}")

    start_time = time.time()

    results = []

    for index, script_name in enumerate(
        EDA_SCRIPTS,
        start=1
    ):

        print(
            f"\nPIPELINE PROGRESS: "
            f"{index}/{len(EDA_SCRIPTS)}"
        )

        result = execute_script(script_name)

        results.append(result)

        # Stop pipeline if an individual script fails.
        if result["status"] == "FAILED":

            print("\nEDA PIPELINE STOPPED.")
            print(
                f"Please fix the error in: {script_name}"
            )

            break

    total_duration = time.time() - start_time

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    successful = sum(
        result["status"] == "SUCCESS"
        for result in results
    )

    failed = sum(
        result["status"] == "FAILED"
        for result in results
    )

    print("\n" + "=" * 75)
    print("EDA PIPELINE EXECUTION SUMMARY")
    print("=" * 75)

    for result in results:

        print(
            f"{result['script']:<40} "
            f"{result['status']:<10} "
            f"{result['duration']:.2f} sec"
        )

    print("-" * 75)

    print(f"Successful scripts: {successful}")
    print(f"Failed scripts: {failed}")
    print(f"Total execution time: {total_duration:.2f} seconds")

    if failed == 0 and successful == len(EDA_SCRIPTS):

        print("\nALL EDA SCRIPTS COMPLETED SUCCESSFULLY!")

    else:

        print("\nEDA PIPELINE INCOMPLETE. CHECK THE ERRORS ABOVE.")

    save_execution_log(results, total_duration)


# ------------------------------------------------------------
# 6. SCRIPT ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
