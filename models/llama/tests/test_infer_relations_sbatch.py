"""Sanity checks on the Llama sbatch wrapper. No SLURM needed."""
from pathlib import Path
import subprocess

SBATCH = Path(__file__).resolve().parent.parent / "infer_relations.sbatch"


def test_sbatch_file_exists():
    assert SBATCH.is_file(), f"{SBATCH} not found"


def test_sbatch_has_required_directives():
    text = SBATCH.read_text()
    assert "#SBATCH --job-name=" in text
    assert "#SBATCH --time=" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --output=" in text
    assert "#SBATCH --error=" in text
    # Partition is intentionally NOT in the header (env-var driven, passed at submit).
    assert "#SBATCH --partition=" not in text, (
        "Per UNI-66 spec §3.3, partition is passed on the sbatch command line, not in the header."
    )


def test_sbatch_guards_partition_env_var():
    text = SBATCH.read_text()
    assert "SLURM_PARTITION_GPU" in text, "must reference SLURM_PARTITION_GPU"
    assert ":?" in text, "must use bash :? guard so missing env var fails fast"


def test_sbatch_activates_main_venv():
    text = SBATCH.read_text()
    assert ".venv/bin/activate" in text
    assert ".venv-maven-train" not in text, "Llama wrapper must NOT activate the BERT venv"


def test_sbatch_invokes_infer_relations_module():
    text = SBATCH.read_text()
    assert "python -m models.llama.infer_relations" in text


def test_sbatch_passes_through_cli_args():
    text = SBATCH.read_text()
    assert '"$@"' in text


def test_sbatch_bash_syntax():
    result = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"
