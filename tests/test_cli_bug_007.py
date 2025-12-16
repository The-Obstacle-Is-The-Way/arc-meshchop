import unittest.mock

from typer.testing import CliRunner

from arc_meshchop.cli import app

runner = CliRunner()


def test_experiment_invalid_variant_validation() -> None:
    """Test that providing an invalid variant fails with correct error (BUG-007)."""
    with unittest.mock.patch("arc_meshchop.experiment.runner.run_experiment") as mock_run:
        result = runner.invoke(
            app,
            ["experiment", "--variant", "invalid_variant", "--output", "tmp_out"],
        )

        # Should exit with error
        assert result.exit_code != 0, "Should fail with invalid variant"

        # Should print error message to stderr
        assert "Invalid variant" in result.stderr

        # Should NOT call run_experiment
        mock_run.assert_not_called()


def test_experiment_valid_variant_passes() -> None:
    """Test that providing a valid variant passes validation."""
    with unittest.mock.patch("arc_meshchop.experiment.runner.run_experiment") as mock_run:
        # Configure mock return value to support formatting
        mock_result = unittest.mock.MagicMock()
        mock_result.test_mean_dice = 0.876
        mock_result.test_std_dice = 0.016
        mock_result.test_mean_avd = 1.5
        mock_result.test_std_avd = 0.5
        mock_result.test_mean_mcc = 0.85
        mock_result.test_std_mcc = 0.02
        mock_result.total_duration_hours = 2.5
        mock_run.return_value = mock_result

        result = runner.invoke(
            app,
            ["experiment", "--variant", "meshnet_26", "--output", "tmp_out"],
        )

        assert result.exit_code == 0
        mock_run.assert_called_once()
