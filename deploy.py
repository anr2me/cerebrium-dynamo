import os
import shutil
import subprocess
import sys
import uuid


def cerebrium_deploy(source_toml: str) -> bool:
    """
    Deploy to Cerebrium using the given toml file as cerebrium.toml.

    Returns True on success, False on deploy failure.
    """
    cerebrium_toml = "cerebrium.toml"
    backup_toml = None
    deploy_success = True

    # Guard — source toml must exist before touching anything
    if not os.path.exists(source_toml):
        print(f"Error: source file '{source_toml}' not found.")
        return False

    # Step 1 — back up existing cerebrium.toml if present (skip entirely if
    # source_toml IS cerebrium.toml — e.g. `python deploy.py cerebrium.toml`,
    # or the no-argument default — since there's nothing to back up from or
    # copy into; it's already in place)
    if cerebrium_toml != source_toml and os.path.exists(cerebrium_toml):
        backup_toml = f"{uuid.uuid4()}.toml"
        os.rename(cerebrium_toml, backup_toml)
        print(f"Existing cerebrium.toml renamed to: {backup_toml}")

    try:
        # Step 2 — copy source toml into place (skipped if source_toml is
        # already cerebrium.toml)
        if cerebrium_toml != source_toml:
            shutil.copy2(source_toml, cerebrium_toml)
            print(f"Copied '{source_toml}' → cerebrium.toml")

        # Step 3 — run the deploy
        print("Running: cerebrium deploy …")
        try:
            result = subprocess.run(
                ["cerebrium", "deploy", "-y"],
                capture_output=False,
            )
        except Exception as e:
            print(f"Deploy Failed: subprocess error — {e}")
            deploy_success = False
        else:
            if result.returncode != 0:
                print("Deploy Failed")
                deploy_success = False
            else:
                print("Deploy succeeded.")

    finally:
        # Step 4 — remove the temporary cerebrium.toml (only if we actually
        # created one as a stand-in; if source_toml WAS cerebrium.toml,
        # there's no temporary file — it's the real one, leave it alone)
        if cerebrium_toml != source_toml and os.path.exists(cerebrium_toml):
            os.remove(cerebrium_toml)
            print("Removed temporary cerebrium.toml")

        # Step 5 — restore the original cerebrium.toml if one was backed up
        if backup_toml and os.path.exists(backup_toml):
            os.rename(backup_toml, cerebrium_toml)
            print(f"Restored {backup_toml} → cerebrium.toml")

    return deploy_success


if __name__ == "__main__":
    first_argument = "cerebrium.toml"
    if len(sys.argv) > 1:
        first_argument = sys.argv[1]

    success = cerebrium_deploy(first_argument)
    print(f"\nDeploy result: {'SUCCESS' if success else 'FAILED'}")
