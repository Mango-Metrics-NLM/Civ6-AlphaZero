"""Unit tests for alpha_zero.training.training_report module."""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from alpha_zero.training.training_report import (
    TrainingReport,
    TrainingReportConfig,
    TrainingReportData,
)


class TestTrainingReportConfig(unittest.TestCase):
    def test_config_defaults(self):
        config = TrainingReportConfig(report_interval=100)
        self.assertEqual(config.report_interval, 100)
        self.assertTrue(config.include_safeguard_events)
        self.assertTrue(config.include_hyperparameters)
        self.assertEqual(config.wandb_prefix, "training_report/")

    def test_config_validation_too_small(self):
        with self.assertRaises(ValueError) as ctx:
            TrainingReportConfig(report_interval=0)
        self.assertIn("must be >= 1", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            TrainingReportConfig(report_interval=-5)
        self.assertIn("must be >= 1", str(ctx.exception))

    def test_config_serialization(self):
        original = TrainingReportConfig(
            report_interval=50,
            include_safeguard_events=False,
            include_hyperparameters=True,
            wandb_prefix="custom/",
        )

        data = original.to_dict()
        restored = TrainingReportConfig.from_dict(data)

        self.assertEqual(restored.report_interval, 50)
        self.assertFalse(restored.include_safeguard_events)
        self.assertTrue(restored.include_hyperparameters)
        self.assertEqual(restored.wandb_prefix, "custom/")

    @patch.dict(
        os.environ,
        {
            "CIV6AZ_REPORT_INTERVAL": "200",
            "CIV6AZ_REPORT_SAFEGUARD_EVENTS": "false",
            "CIV6AZ_REPORT_HYPERPARAMETERS": "true",
            "CIV6AZ_REPORT_WANDB_PREFIX": "metrics/",
        },
    )
    def test_config_from_env(self):
        config = TrainingReportConfig.from_env()

        self.assertEqual(config.report_interval, 200)
        self.assertFalse(config.include_safeguard_events)
        self.assertTrue(config.include_hyperparameters)
        self.assertEqual(config.wandb_prefix, "metrics/")

    @patch.dict(os.environ, {}, clear=True)
    def test_config_from_env_defaults(self):
        config = TrainingReportConfig.from_env()

        self.assertEqual(config.report_interval, 100)
        self.assertTrue(config.include_safeguard_events)
        self.assertTrue(config.include_hyperparameters)
        self.assertEqual(config.wandb_prefix, "training_report/")


class TestTrainingReportData(unittest.TestCase):
    def test_to_dict_serialization(self):
        report = TrainingReportData(
            step_range=(0, 99),
            num_steps=100,
            mean_loss=0.5,
            min_loss=0.1,
            max_loss=0.9,
            mean_grad_norm=1.2,
            max_grad_norm=2.5,
            learning_rate=0.001,
            throughput_steps_per_sec=10.5,
            safeguard_events=[{"event": "grad_clip", "step": 50}],
            extra_metrics={"policy_loss": 0.3, "value_loss": 0.2},
            timestamp="2026-02-09T12:00:00Z",
        )

        data = report.to_dict()

        self.assertEqual(data["step_range"], (0, 99))
        self.assertEqual(data["num_steps"], 100)
        self.assertEqual(data["mean_loss"], 0.5)
        self.assertEqual(data["min_loss"], 0.1)
        self.assertEqual(data["max_loss"], 0.9)
        self.assertEqual(data["mean_grad_norm"], 1.2)
        self.assertEqual(data["max_grad_norm"], 2.5)
        self.assertEqual(data["learning_rate"], 0.001)
        self.assertEqual(data["throughput_steps_per_sec"], 10.5)
        self.assertEqual(len(data["safeguard_events"]), 1)
        self.assertEqual(data["extra_metrics"]["policy_loss"], 0.3)
        self.assertEqual(data["timestamp"], "2026-02-09T12:00:00Z")

    def test_to_wandb_dict_namespacing(self):
        report = TrainingReportData(
            step_range=(100, 199),
            num_steps=100,
            mean_loss=0.4,
            min_loss=0.05,
            max_loss=0.8,
            mean_grad_norm=1.0,
            max_grad_norm=2.0,
            learning_rate=0.0001,
            throughput_steps_per_sec=15.0,
            safeguard_events=[{"event": "test1"}, {"event": "test2"}],
            extra_metrics={"custom_metric": 0.7},
        )

        wandb_dict = report.to_wandb_dict(prefix="training_report/")

        self.assertEqual(wandb_dict["training_report/step_start"], 100)
        self.assertEqual(wandb_dict["training_report/step_end"], 199)
        self.assertEqual(wandb_dict["training_report/num_steps"], 100)
        self.assertEqual(wandb_dict["training_report/mean_loss"], 0.4)
        self.assertEqual(wandb_dict["training_report/min_loss"], 0.05)
        self.assertEqual(wandb_dict["training_report/max_loss"], 0.8)
        self.assertEqual(wandb_dict["training_report/mean_grad_norm"], 1.0)
        self.assertEqual(wandb_dict["training_report/max_grad_norm"], 2.0)
        self.assertEqual(wandb_dict["training_report/learning_rate"], 0.0001)
        self.assertEqual(wandb_dict["training_report/throughput_steps_per_sec"], 15.0)
        self.assertEqual(wandb_dict["training_report/num_safeguard_events"], 2)
        self.assertEqual(wandb_dict["training_report/extra/custom_metric"], 0.7)

    def test_to_wandb_dict_custom_prefix(self):
        report = TrainingReportData(
            step_range=(0, 9),
            num_steps=10,
            mean_loss=0.5,
            min_loss=0.1,
            max_loss=0.9,
            mean_grad_norm=1.0,
            max_grad_norm=2.0,
            learning_rate=0.001,
            throughput_steps_per_sec=5.0,
        )

        wandb_dict = report.to_wandb_dict(prefix="custom/metrics/")

        self.assertEqual(wandb_dict["custom/metrics/step_start"], 0)
        self.assertEqual(wandb_dict["custom/metrics/step_end"], 9)
        self.assertEqual(wandb_dict["custom/metrics/mean_loss"], 0.5)


class TestTrainingReport(unittest.TestCase):
    def test_report_interval(self):
        config = TrainingReportConfig(report_interval=10)
        report = TrainingReport(config)

        for i in range(9):
            report.record_step(step=i, loss=0.5, grad_norm=1.0, lr=0.001)
            self.assertFalse(report.should_report(i))

        report.record_step(step=9, loss=0.5, grad_norm=1.0, lr=0.001)
        self.assertTrue(report.should_report(9))

    def test_report_stats_accuracy(self):
        config = TrainingReportConfig(report_interval=5)
        report = TrainingReport(config)

        losses = [0.1, 0.2, 0.3, 0.4, 0.5]
        for i, loss in enumerate(losses):
            report.record_step(step=i + 1, loss=loss, grad_norm=1.0, lr=0.001)

        time.sleep(0.01)
        report_data = report.generate_report(step=5)

        self.assertIsNotNone(report_data)
        self.assertAlmostEqual(report_data.mean_loss, 0.3, places=5)
        self.assertAlmostEqual(report_data.min_loss, 0.1, places=5)
        self.assertAlmostEqual(report_data.max_loss, 0.5, places=5)
        self.assertEqual(report_data.num_steps, 5)
        self.assertEqual(report_data.step_range, (1, 5))

    def test_throughput_calculation(self):
        config = TrainingReportConfig(report_interval=5)
        report = TrainingReport(config)

        for i in range(5):
            report.record_step(step=i, loss=0.5, grad_norm=1.0, lr=0.001)
            time.sleep(0.01)

        report_data = report.generate_report(step=4)

        self.assertIsNotNone(report_data)
        self.assertGreater(report_data.throughput_steps_per_sec, 0)

    def test_safeguard_events_included(self):
        config = TrainingReportConfig(report_interval=3, include_safeguard_events=True)
        report = TrainingReport(config)

        report.record_step(step=0, loss=0.5, grad_norm=1.0, lr=0.001)
        report.record_step(
            step=1,
            loss=0.5,
            grad_norm=1.0,
            lr=0.001,
            safeguard_event={"event": "grad_clip", "step": 1},
        )
        report.record_step(step=2, loss=0.5, grad_norm=1.0, lr=0.001)

        report_data = report.generate_report(step=2)

        self.assertIsNotNone(report_data)
        self.assertEqual(len(report_data.safeguard_events), 1)
        self.assertEqual(report_data.safeguard_events[0]["event"], "grad_clip")
        self.assertEqual(report_data.safeguard_events[0]["step"], 1)

    def test_safeguard_events_disabled(self):
        config = TrainingReportConfig(report_interval=3, include_safeguard_events=False)
        report = TrainingReport(config)

        report.record_step(
            step=0,
            loss=0.5,
            grad_norm=1.0,
            lr=0.001,
            safeguard_event={"event": "grad_clip"},
        )
        report.record_step(step=1, loss=0.5, grad_norm=1.0, lr=0.001)
        report.record_step(step=2, loss=0.5, grad_norm=1.0, lr=0.001)

        report_data = report.generate_report(step=2)

        self.assertIsNotNone(report_data)
        self.assertEqual(len(report_data.safeguard_events), 0)

    def test_extra_metrics_averaged(self):
        config = TrainingReportConfig(report_interval=3)
        report = TrainingReport(config)

        report.record_step(
            step=0, loss=0.5, grad_norm=1.0, lr=0.001, policy_loss=0.3, value_loss=0.2
        )
        report.record_step(
            step=1, loss=0.5, grad_norm=1.0, lr=0.001, policy_loss=0.4, value_loss=0.3
        )
        report.record_step(
            step=2, loss=0.5, grad_norm=1.0, lr=0.001, policy_loss=0.5, value_loss=0.4
        )

        report_data = report.generate_report(step=2)

        self.assertIsNotNone(report_data)
        self.assertAlmostEqual(report_data.extra_metrics["policy_loss"], 0.4, places=5)
        self.assertAlmostEqual(report_data.extra_metrics["value_loss"], 0.3, places=5)

    def test_empty_interval_report(self):
        config = TrainingReportConfig(report_interval=10)
        report = TrainingReport(config)

        report_data = report.generate_report(step=0)

        self.assertIsNone(report_data)

    def test_get_latest_report(self):
        config = TrainingReportConfig(report_interval=2)
        report = TrainingReport(config)

        self.assertIsNone(report.get_latest_report())

        report.record_step(step=1, loss=0.1, grad_norm=1.0, lr=0.001)
        report.record_step(step=2, loss=0.2, grad_norm=1.0, lr=0.001)
        time.sleep(0.01)
        report.generate_report(step=2)

        latest = report.get_latest_report()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.step_range, (1, 2))
        self.assertAlmostEqual(latest.mean_loss, 0.15, places=5)

        report.record_step(step=3, loss=0.3, grad_norm=1.0, lr=0.001)
        report.record_step(step=4, loss=0.4, grad_norm=1.0, lr=0.001)
        time.sleep(0.01)
        report.generate_report(step=4)

        latest = report.get_latest_report()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.step_range, (3, 4))
        self.assertAlmostEqual(latest.mean_loss, 0.35, places=5)

    def test_get_all_reports(self):
        config = TrainingReportConfig(report_interval=2)
        report = TrainingReport(config)

        self.assertEqual(len(report.get_all_reports()), 0)

        for i in range(6):
            report.record_step(step=i + 1, loss=float(i) * 0.1, grad_norm=1.0, lr=0.001)
            if (i + 1) % 2 == 0:
                time.sleep(0.01)
                report.generate_report(step=i + 1)

        all_reports = report.get_all_reports()
        self.assertEqual(len(all_reports), 3)
        self.assertEqual(all_reports[0].step_range, (1, 2))
        self.assertEqual(all_reports[1].step_range, (3, 4))
        self.assertEqual(all_reports[2].step_range, (5, 6))

    def test_reset_clears_state(self):
        config = TrainingReportConfig(report_interval=3)
        report = TrainingReport(config)

        report.record_step(step=0, loss=0.5, grad_norm=1.0, lr=0.001)
        report.record_step(step=1, loss=0.5, grad_norm=1.0, lr=0.001)
        report.record_step(step=2, loss=0.5, grad_norm=1.0, lr=0.001)
        time.sleep(0.01)
        report.generate_report(step=2)

        self.assertEqual(report.report_count, 1)

        report.reset()

        self.assertEqual(report.report_count, 0)
        self.assertIsNone(report.get_latest_report())
        self.assertEqual(len(report.get_all_reports()), 0)

        report.record_step(step=0, loss=0.1, grad_norm=1.0, lr=0.001)
        report_data = report.generate_report(step=0)
        self.assertIsNotNone(report_data)
        self.assertEqual(report_data.step_range, (0, 0))

    def test_report_count_property(self):
        config = TrainingReportConfig(report_interval=2)
        report = TrainingReport(config)

        self.assertEqual(report.report_count, 0)

        report.record_step(step=0, loss=0.1, grad_norm=1.0, lr=0.001)
        report.record_step(step=1, loss=0.2, grad_norm=1.0, lr=0.001)
        time.sleep(0.01)
        report.generate_report(step=1)
        self.assertEqual(report.report_count, 1)

        report.record_step(step=2, loss=0.3, grad_norm=1.0, lr=0.001)
        report.record_step(step=3, loss=0.4, grad_norm=1.0, lr=0.001)
        time.sleep(0.01)
        report.generate_report(step=3)
        self.assertEqual(report.report_count, 2)

        report.record_step(step=4, loss=0.5, grad_norm=1.0, lr=0.001)
        report.record_step(step=5, loss=0.6, grad_norm=1.0, lr=0.001)
        time.sleep(0.01)
        report.generate_report(step=5)
        self.assertEqual(report.report_count, 3)

    def test_gradient_norm_statistics(self):
        config = TrainingReportConfig(report_interval=4)
        report = TrainingReport(config)

        grad_norms = [0.5, 1.0, 1.5, 2.0]
        for i, grad_norm in enumerate(grad_norms):
            report.record_step(step=i, loss=0.5, grad_norm=grad_norm, lr=0.001)

        time.sleep(0.01)
        report_data = report.generate_report(step=3)

        self.assertIsNotNone(report_data)
        self.assertAlmostEqual(report_data.mean_grad_norm, 1.25, places=5)
        self.assertAlmostEqual(report_data.max_grad_norm, 2.0, places=5)

    def test_learning_rate_averaging(self):
        config = TrainingReportConfig(report_interval=3)
        report = TrainingReport(config)

        learning_rates = [0.001, 0.0005, 0.00025]
        for i, lr in enumerate(learning_rates):
            report.record_step(step=i, loss=0.5, grad_norm=1.0, lr=lr)

        time.sleep(0.01)
        report_data = report.generate_report(step=2)

        self.assertIsNotNone(report_data)
        expected_lr = sum(learning_rates) / len(learning_rates)
        self.assertAlmostEqual(report_data.learning_rate, expected_lr, places=6)

    def test_multiple_extra_metrics(self):
        config = TrainingReportConfig(report_interval=2)
        report = TrainingReport(config)

        report.record_step(
            step=0, loss=0.5, grad_norm=1.0, lr=0.001, metric_a=1.0, metric_b=2.0, metric_c=3.0
        )
        report.record_step(
            step=1, loss=0.5, grad_norm=1.0, lr=0.001, metric_a=2.0, metric_b=3.0, metric_c=4.0
        )

        time.sleep(0.01)
        report_data = report.generate_report(step=1)

        self.assertIsNotNone(report_data)
        self.assertAlmostEqual(report_data.extra_metrics["metric_a"], 1.5, places=5)
        self.assertAlmostEqual(report_data.extra_metrics["metric_b"], 2.5, places=5)
        self.assertAlmostEqual(report_data.extra_metrics["metric_c"], 3.5, places=5)


if __name__ == "__main__":
    unittest.main()
