import os
import sys
import json
import yaml

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.vehicle.logger.logger import logger
from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.entity.config_entity import ModelEvaluationConfig

from src.vehicle.entity.artifacts_entity import (ModelTrainerArtifact,
                                                 DataIngestionArtifact,
                                                 ModelEvaluationArtifact)


class ModelEvaluation:
    """
    End-to-end model evaluation for YOLO26 foggy vehicle detection.

    Pipeline (mirrors notebook):
      1. Standard vs TTA comparison
      2. IoU threshold sweep  → best NMS IoU via mAP50-95
      3. Conf threshold sweep → best conf via F1 score
      4. Final per-class evaluation on test set (best IoU + best conf + TTA)
      5. Save plots + JSON report
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(),
    ):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.cfg = model_evaluation_config

            self.data_yaml = os.path.join(
                data_ingestion_artifact.feature_store_file_path, "data.yaml"
            )
            self.model = YOLO(model_trainer_artifact.trained_model_path)

            os.makedirs(self.cfg.model_evaluation_dir, exist_ok=True)
            os.makedirs(self.cfg.plots_dir, exist_ok=True)

        except Exception as e:
            raise VehicleException(e, sys)

    # ──────────────────────────────────────────────────────────
    # Step 1: Standard vs TTA
    # ──────────────────────────────────────────────────────────
    def _run_standard_vs_tta(self) -> tuple[dict, dict]:
        """Compare standard inference vs Test Time Augmentation."""
        logger.info("Running standard evaluation...")
        standard = self.model.val(
            data=self.data_yaml,
            split=self.cfg.split,
            imgsz=640,
            conf=self.cfg.conf_default,
            iou=self.cfg.iou_default,
            verbose=False,
        )

        logger.info("Running TTA evaluation...")
        tta = self.model.val(
            data=self.data_yaml,
            split=self.cfg.split,
            imgsz=640,
            conf=self.cfg.conf_default,
            iou=self.cfg.iou_default,
            augment=True,   # enables TTA
            verbose=False,
        )

        standard_results = {
            "mAP50": round(float(standard.box.map50), 4),
            "mAP50_95": round(float(standard.box.map), 4),
            "precision": round(float(standard.box.p.mean()), 4),
            "recall": round(float(standard.box.r.mean()), 4),
        }
        tta_results = {
            "mAP50": round(float(tta.box.map50), 4),
            "mAP50_95": round(float(tta.box.map), 4),
            "precision": round(float(tta.box.p.mean()), 4),
            "recall": round(float(tta.box.r.mean()), 4),
        }

        logger.info(f"Standard: {standard_results}")
        logger.info(f"TTA:      {tta_results}")

        self._print_comparison_table(standard_results, tta_results)
        return standard_results, tta_results

    def _print_comparison_table(self, standard: dict, tta: dict):
        print(f"\n{'─'*46}")
        print(f"{'Metric':<20} {'Standard':>12} {'With TTA':>12}")
        print(f"{'─'*46}")
        for key in standard:
            print(f"{key:<20} {standard[key]:>12.4f} {tta[key]:>12.4f}")
        print(f"{'─'*46}\n")

    # ──────────────────────────────────────────────────────────
    # Step 2: IoU Threshold Sweep
    # ──────────────────────────────────────────────────────────
    def _run_iou_sweep(self) -> dict:
        """Sweep NMS IoU thresholds and return best by mAP50-95."""
        logger.info(f"Sweeping IoU thresholds: {self.cfg.iou_sweep}")
        results = []

        for iou_thresh in self.cfg.iou_sweep:
            m = self.model.val(
                data=self.data_yaml,
                split="val",        # use val set for hyperparameter tuning
                imgsz=640,
                conf=self.cfg.conf_default,
                iou=iou_thresh,
                verbose=False,
            )
            entry = {
                "iou": iou_thresh,
                "mAP50": round(float(m.box.map50), 4),
                "mAP50_95": round(float(m.box.map), 4),
                "precision": round(float(m.box.p.mean()), 4),
                "recall": round(float(m.box.r.mean()), 4),
            }
            results.append(entry)
            logger.info(
                f"IoU={iou_thresh:.2f} : mAP50={entry['mAP50']:.4f} | "
                f"mAP50-95={entry['mAP50_95']:.4f} | "
                f"P={entry['precision']:.4f} | R={entry['recall']:.4f}"
            )

        self._plot_iou_sweep(results)

        best = max(results, key=lambda x: x["mAP50_95"])
        logger.info(f"Best NMS IoU: {best['iou']} : mAP50-95: {best['mAP50_95']:.4f}")
        return best

    def _plot_iou_sweep(self, results: list):
        ious = [r["iou"] for r in results]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(ious, [r["mAP50"] for r in results], "b-o", label="mAP50")
        axes[0].plot(ious, [r["mAP50_95"] for r in results], "r-o", label="mAP50-95")
        axes[0].set_xlabel("NMS IoU Threshold")
        axes[0].set_ylabel("mAP")
        axes[0].set_title("mAP vs NMS IoU")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(ious, [r["precision"] for r in results], "g-o", label="Precision")
        axes[1].plot(ious, [r["recall"] for r in results], "m-o", label="Recall")
        axes[1].set_xlabel("NMS IoU Threshold")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Precision/Recall vs NMS IoU")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.cfg.plots_dir, "iou_sweep.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"IoU sweep plot saved: {save_path}")

    # ──────────────────────────────────────────────────────────
    # Step 3: Confidence Threshold Sweep
    # ──────────────────────────────────────────────────────────
    def _run_conf_sweep(self, best_iou: float) -> dict:
        """Sweep confidence thresholds and return best by F1 score."""
        logger.info(f"Sweeping confidence thresholds: {self.cfg.conf_sweep}")
        results = []

        for conf in self.cfg.conf_sweep:
            m = self.model.val(
                data=self.data_yaml,
                split="val",
                imgsz=640,
                conf=conf,
                iou=best_iou,
                verbose=False,
            )
            p = float(m.box.p.mean())
            r = float(m.box.r.mean())
            f1 = 2 * p * r / (p + r + 1e-6)

            entry = {
                "conf": conf,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "mAP50_95": round(float(m.box.map), 4),
            }
            results.append(entry)
            logger.info(
                f"conf={conf:.2f} → P={p:.3f} | R={r:.3f} | "
                f"F1={f1:.3f} | mAP50-95={entry['mAP50_95']:.4f}"
            )

        self._plot_conf_sweep(results)

        best = max(results, key=lambda x: x["f1"])
        logger.info(f"Best confidence: {best['conf']} (F1={best['f1']:.4f})")
        return best

    def _plot_conf_sweep(self, results: list):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        confs = [r["conf"] for r in results]
        axes[0].plot(
            [r["recall"] for r in results],
            [r["precision"] for r in results],
            "b-o"
        )
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].set_title("Precision-Recall Tradeoff")
        axes[0].grid(True)

        axes[1].plot(confs, [r["f1"] for r in results], "r-o", label="F1")
        axes[1].set_xlabel("Confidence Threshold")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("F1 vs Confidence Threshold")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.cfg.plots_dir, "conf_sweep.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Confidence sweep plot saved: {save_path}")

    # ──────────────────────────────────────────────────────────
    # Step 4: Final Per-Class Evaluation
    # ──────────────────────────────────────────────────────────
    def _run_final_evaluation(self, best_iou: float, best_conf: float) -> dict:
        """Run final evaluation on test set with best thresholds + TTA."""
        logger.info("Running final per-class evaluation (TTA + best thresholds)...")

        with open(self.data_yaml) as f:
            data = yaml.safe_load(f)
        class_names = data["names"]

        m = self.model.val(
            data=self.data_yaml,
            split=self.cfg.split,
            imgsz=640,
            conf=best_conf,
            iou=best_iou,
            augment=True,   # TTA
            verbose=False,
        )

        # Print per-class table
        print(f"\n{'─'*60}")
        print(f"{'Class':<20} {'AP50':>8} {'AP50-95':>10} {'Precision':>10} {'Recall':>8}")
        print(f"{'─'*60}")

        per_class = {}
        for i, name in enumerate(class_names):
            try:
                ap50 = float(m.box.ap50[i])
                ap = float(m.box.ap[i])
                p = float(m.box.p[i])
                r = float(m.box.r[i])
                per_class[name] = {
                    "AP50": round(ap50, 4),
                    "AP50_95": round(ap, 4),
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                }
                print(f"{name:<20} {ap50:>8.4f} {ap:>10.4f} {p:>10.4f} {r:>8.4f}")
            except Exception:
                pass

        print(f"{'─'*60}")
        print(f"{'OVERALL':<20} {m.box.map50:>8.4f} {m.box.map:>10.4f}")
        print(f"{'─'*60}\n")

        self._plot_per_class(per_class)

        return {
            "final_map50": round(float(m.box.map50), 4),
            "final_map50_95": round(float(m.box.map), 4),
            "per_class": per_class,
        }

    def _plot_per_class(self, per_class: dict):
        names = list(per_class.keys())
        ap50_vals = [per_class[n]["AP50"] for n in names]
        ap_vals = [per_class[n]["AP50_95"] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(names, ap50_vals, color="steelblue")
        axes[0].set_title("Per-Class AP50")
        axes[0].set_ylabel("AP50")
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].bar(names, ap_vals, color="coral")
        axes[1].set_title("Per-Class AP50-95")
        axes[1].set_ylabel("AP50-95")
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        save_path = os.path.join(self.cfg.plots_dir, "per_class_ap.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Per-class AP plot saved: {save_path}")

    # ──────────────────────────────────────────────────────────
    # Step 5: Save Report
    # ──────────────────────────────────────────────────────────
    def _save_report(self, report: dict):
        with open(self.cfg.evaluation_report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Evaluation report saved: {self.cfg.evaluation_report_path}")

    # ──────────────────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────────────────
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logger.info("Initiating model evaluation component")
        try:
            if os.path.exists(self.cfg.evaluation_report_path):
                logger.info(f"Evaluation report already exists at: {self.cfg.evaluation_report_path} — skipping evaluation.")
                with open(self.cfg.evaluation_report_path, "r") as f:
                    report = json.load(f)

                return ModelEvaluationArtifact(
                    best_iou_threshold=report["best_iou_threshold"],
                    best_conf_threshold=report["best_conf_threshold"],
                    best_f1_score=report["best_f1_score"],
                    standard_map50=report["standard"]["mAP50"],
                    standard_map50_95=report["standard"]["mAP50_95"],
                    tta_map50=report["tta"]["mAP50"],
                    tta_map50_95=report["tta"]["mAP50_95"],
                    final_map50=report["final"]["final_map50"],
                    final_map50_95=report["final"]["final_map50_95"],
                    evaluation_report_path=self.cfg.evaluation_report_path,
                )

            logger.info("No evaluation report found — proceeding with model evaluation.")

            # Step 1
            standard_results, tta_results = self._run_standard_vs_tta()

            # Step 2
            best_iou_entry = self._run_iou_sweep()

            # Step 3
            best_conf_entry = self._run_conf_sweep(best_iou=best_iou_entry["iou"])

            # Step 4
            final_results = self._run_final_evaluation(
                best_iou=best_iou_entry["iou"],
                best_conf=best_conf_entry["conf"],
            )

            # Step 5
            report = {
                "standard": standard_results,
                "tta": tta_results,
                "best_iou_threshold": best_iou_entry["iou"],
                "best_conf_threshold": best_conf_entry["conf"],
                "best_f1_score": best_conf_entry["f1"],
                "final": final_results,
            }
            self._save_report(report)

            artifact = ModelEvaluationArtifact(
                best_iou_threshold=best_iou_entry["iou"],
                best_conf_threshold=best_conf_entry["conf"],
                best_f1_score=best_conf_entry["f1"],
                standard_map50=standard_results["mAP50"],
                standard_map50_95=standard_results["mAP50_95"],
                tta_map50=tta_results["mAP50"],
                tta_map50_95=tta_results["mAP50_95"],
                final_map50=final_results["final_map50"],
                final_map50_95=final_results["final_map50_95"],
                evaluation_report_path=self.cfg.evaluation_report_path,
            )

            logger.info(f"Model evaluation artifact: {artifact}")
            return artifact

        except Exception as e:
            raise VehicleException(e, sys)