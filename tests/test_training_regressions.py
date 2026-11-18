"""Regression tests for the local trainers (run with python -m unittest discover)."""

import unittest

import torch

import simple_trainer_low_vram as low
from gsplat.scene import GaussianScene
from gsplat.strategy.ops import duplicate, remove, split


def make_splats(count=4, device="cpu", sh_degree=0):
    means = torch.zeros(count, 3, device=device)
    means[:, 2] = 3.0
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.full((count, 3), -4.0, device=device)),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                .expand(count, 4)
                .clone()
            ),
            "opacities": torch.nn.Parameter(torch.zeros(count, device=device)),
            "sh0": torch.nn.Parameter(
                torch.arange(count, device=device, dtype=torch.float32)
                .reshape(count, 1, 1)
                .expand(count, 1, 3)
                .clone()
            ),
        }
    )
    if sh_degree:
        splats["shN"] = torch.nn.Parameter(
            torch.zeros(count, (sh_degree + 1) ** 2 - 1, 3, device=device)
        )
    return splats


class ImportanceAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.splats = make_splats()
        self.scene = GaussianScene.from_splats(self.splats, id="test")
        self.optimizers = {
            name: torch.optim.Adam([param])
            for name, param in self.splats.items()
        }
        import inspect
        options = {"scene": self.scene} if "scene" in inspect.signature(
            low.ImportanceBudgetController
        ).parameters else {}
        self.controller = low.ImportanceBudgetController(
            4, max_cap=2, device="cpu", **options
        )
        self.controller.importance_score.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
        self.controller.visibility_count.copy_(torch.tensor([1, 2, 3, 4]))

    def test_strategy_remove_keeps_scores_with_surviving_gaussians(self):
        remove(
            self.splats,
            self.optimizers,
            {},
            torch.tensor([False, True, False, False]),
            scene=self.scene,
        )
        self.controller.sync_size(3)
        self.assertEqual(self.controller.importance_score.tolist(), [10, 30, 40])
        self.assertEqual(self.controller.visibility_count.tolist(), [1, 3, 4])
        removed = self.controller.enforce_hard_cap(
            self.splats, self.optimizers, {}, self.scene
        )
        self.assertEqual(removed, 1)
        self.assertEqual(self.splats["sh0"][:, 0, 0].tolist(), [2, 3])
        self.assertEqual(self.controller.importance_score.tolist(), [30, 40])

    def test_duplicate_and_split_preserve_parent_identity(self):
        duplicate(
            self.splats,
            self.optimizers,
            {},
            torch.tensor([False, True, False, False]),
            scene=self.scene,
        )
        self.assertEqual(self.controller.importance_score.tolist(), [10, 20, 30, 40, 20])
        split(
            self.splats,
            self.optimizers,
            {},
            torch.tensor([True, False, False, True, False]),
            scene=self.scene,
        )
        self.assertEqual(self.splats["sh0"][:, 0, 0].tolist(), [1, 2, 1, 0, 3, 0, 3])
        self.assertEqual(
            self.controller.importance_score.tolist(), [20, 30, 20, 10, 40, 10, 40]
        )
        self.assertEqual(self.controller.visibility_count.tolist(), [2, 3, 2, 1, 4, 1, 4])

    def test_controller_prune_applies_its_mask_once(self):
        self.controller.enforce_hard_cap(
            self.splats, self.optimizers, {}, self.scene
        )
        self.assertEqual(self.splats["sh0"][:, 0, 0].tolist(), [2, 3])
        self.assertEqual(self.controller.importance_score.tolist(), [30, 40])
        self.assertEqual(self.controller.visibility_count.tolist(), [3, 4])
        self.scene.validate()

    def test_relocation_and_permutation_keep_scores_aligned(self):
        # The same hooks are called by the MCMC strategy for relocated/new rows.
        self.scene.on_relocate(torch.tensor([0]), torch.tensor([3]))
        self.scene.on_sample_add(torch.tensor([1]))
        self.scene.on_permute(torch.tensor([4, 0, 2, 1, 3]))
        self.assertEqual(self.controller.importance_score.tolist(), [20, 40, 30, 20, 40])
        self.assertEqual(self.controller.visibility_count.tolist(), [2, 4, 3, 2, 4])


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_mock_runner(self, count=4, sh_degree=0):
        from gsplat.stage import Stage
        runner = object.__new__(low.Runner)
        runner.device = self.device
        runner.world_rank = 0
        runner.world_size = 1
        runner.cfg = low.Config(disable_viewer=True)
        runner.splats = make_splats(count=count, device=self.device, sh_degree=sh_degree)
        runner.scene = GaussianScene.from_splats(runner.splats, id="test")
        runner.splats = runner.scene.splats
        runner.stage = Stage()
        runner.stage.add_scene(runner.scene, runner.rasterize_splats)
        runner.budget_controller = None
        runner.is_compressed = False
        runner.sh_codebook = None
        runner.sh_indices = None
        runner.active_sh_degree = sh_degree
        return runner

    def test_load_compressed_checkpoint_with_shards_and_changed_count(self):
        import tempfile
        import os
        # Create a saved model with 6 Gaussians, compressed codebook (8 centroids, degree 3)
        runner_saved = self._create_mock_runner(count=6, sh_degree=0)
        runner_saved.is_compressed = True
        runner_saved.active_sh_degree = 3
        runner_saved.sh_codebook = torch.nn.Parameter(
            torch.randn(8, 15, 3, device=self.device)
        )
        runner_saved.sh_indices = torch.tensor([0, 1, 2, 3, 4, 5], device=self.device, dtype=torch.int16)

        c2w = torch.eye(4, device=self.device).unsqueeze(0)
        K = torch.tensor([[[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]]], device=self.device)
        colors_saved, _, _ = runner_saved.rasterize_splats(camtoworlds=c2w, Ks=K, width=256, height=256)

        # Save into 2 shards (3 Gaussians each)
        with tempfile.TemporaryDirectory() as tmpdir:
            shard0_path = os.path.join(tmpdir, "ckpt_rank0.pt")
            shard1_path = os.path.join(tmpdir, "ckpt_rank1.pt")

            splats_dict = {k: v.detach().cpu() for k, v in runner_saved.splats.items()}
            shard0_splats = {k: v[:3] for k, v in splats_dict.items()}
            shard1_splats = {k: v[3:] for k, v in splats_dict.items()}

            data0 = {
                "step": 15000,
                "scene_id": "test",
                "splats": shard0_splats,
                "is_compressed": True,
                "active_sh_degree": 3,
                "sh_codebook": runner_saved.sh_codebook.detach().cpu(),
                "sh_indices": runner_saved.sh_indices[:3].cpu(),
            }
            data1 = {
                "step": 15000,
                "scene_id": "test",
                "splats": shard1_splats,
                "is_compressed": True,
                "active_sh_degree": 3,
                "sh_codebook": runner_saved.sh_codebook.detach().cpu(),
                "sh_indices": runner_saved.sh_indices[3:].cpu(),
            }
            torch.save(data0, shard0_path)
            torch.save(data1, shard1_path)

            # Target runner starts with 4 Gaussians, uncompressed, degree 0
            runner_target = self._create_mock_runner(count=4, sh_degree=0)
            step = runner_target.load_checkpoint([shard0_path, shard1_path])

            self.assertEqual(step, 15000)
            self.assertTrue(runner_target.is_compressed)
            self.assertEqual(runner_target.active_sh_degree, 3)
            self.assertEqual(runner_target.splats["means"].shape[0], 6)
            self.assertEqual(runner_target.sh_indices.shape[0], 6)

            colors_loaded, _, _ = runner_target.rasterize_splats(camtoworlds=c2w, Ks=K, width=256, height=256)
            self.assertTrue(torch.allclose(colors_saved, colors_loaded, atol=1e-5))

    def test_load_uncompressed_checkpoint_with_shN(self):
        import tempfile
        import os
        runner_saved = self._create_mock_runner(count=5, sh_degree=1)
        runner_saved.splats["shN"].data.copy_(torch.randn_like(runner_saved.splats["shN"]))
        c2w = torch.eye(4, device=self.device).unsqueeze(0)
        K = torch.tensor([[[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]]], device=self.device)
        colors_saved, _, _ = runner_saved.rasterize_splats(camtoworlds=c2w, Ks=K, width=256, height=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt_uncompressed.pt")
            data = {
                "step": 5000,
                "scene_id": "test",
                "splats": {k: v.detach().cpu() for k, v in runner_saved.splats.items()},
                "is_compressed": False,
                "active_sh_degree": 1,
            }
            torch.save(data, ckpt_path)

            runner_target = self._create_mock_runner(count=2, sh_degree=0)
            step = runner_target.load_checkpoint(ckpt_path)

            self.assertEqual(step, 5000)
            self.assertFalse(runner_target.is_compressed)
            self.assertEqual(runner_target.active_sh_degree, 1)
            self.assertEqual(runner_target.splats["means"].shape[0], 5)

            colors_loaded, _, _ = runner_target.rasterize_splats(camtoworlds=c2w, Ks=K, width=256, height=256)
            self.assertTrue(torch.allclose(colors_saved, colors_loaded, atol=1e-5))


class ExportParameterizationTests(unittest.TestCase):
    def test_convert_module_importable_without_side_effects(self):
        import convert
        self.assertTrue(hasattr(convert, "convert_checkpoint"))

    def test_convert_preserves_raw_log_scales_and_logit_opacities(self):
        import tempfile
        import os
        import convert
        from gsplat.exporter import load_ply_to_splats

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            orig_scales = torch.tensor([[-2.0, -3.0, -4.0], [-1.0, -1.5, -2.5]], dtype=torch.float32)
            orig_opacities = torch.tensor([1.5, -0.5], dtype=torch.float32)
            data = {
                "step": 30000,
                "splats": {
                    "means": torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=torch.float32),
                    "scales": orig_scales.clone(),
                    "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                    "opacities": orig_opacities.clone(),
                    "sh0": torch.ones(2, 1, 3, dtype=torch.float32) * 0.5,
                },
                "is_compressed": False,
            }
            torch.save(data, ckpt_path)

            outputs = convert.convert_checkpoint(
                ckpt_path,
                out_dir=tmpdir,
                formats=["ply"],
                output_name="test_model",
            )
            self.assertIn("ply", outputs)
            ply_path = outputs["ply"]
            self.assertTrue(os.path.exists(ply_path))

            loaded = load_ply_to_splats(ply_path)
            self.assertTrue(
                torch.allclose(loaded["scales"], orig_scales, atol=1e-4),
                f"Scales were double-transformed: expected {orig_scales}, got {loaded['scales']}",
            )
            self.assertTrue(
                torch.allclose(loaded["opacities"], orig_opacities, atol=1e-4),
                f"Opacities were double-transformed: expected {orig_opacities}, got {loaded['opacities']}",
            )

    def test_convert_compressed_checkpoint_with_codebook(self):
        import tempfile
        import os
        import convert

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_compressed.pt")
            codebook = torch.randn(4, 15, 3, dtype=torch.float32)
            indices = torch.tensor([0, 1, 3], dtype=torch.int16)
            data = {
                "step": 30000,
                "splats": {
                    "means": torch.zeros(3, 3, dtype=torch.float32),
                    "scales": torch.full((3, 3), -3.0, dtype=torch.float32),
                    "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float32),
                    "opacities": torch.zeros(3, dtype=torch.float32),
                    "sh0": torch.zeros(3, 1, 3, dtype=torch.float32),
                },
                "is_compressed": True,
                "sh_codebook": codebook,
                "sh_indices": indices,
            }
            torch.save(data, ckpt_path)

            outputs = convert.convert_checkpoint(
                ckpt_path,
                out_dir=tmpdir,
                formats=["splat", "ply_compressed"],
                output_name="compressed_model",
            )
            self.assertIn("splat", outputs)
            self.assertIn("ply_compressed", outputs)
            self.assertGreater(os.path.getsize(outputs["splat"]), 0)
            self.assertGreater(os.path.getsize(outputs["ply_compressed"]), 0)


class EvaluationLogsTests(unittest.TestCase):
    def test_eval_only_mode_preserves_training_logs_and_summaries(self):
        import tempfile
        import os
        from train_logger import TrainLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            train_csv = os.path.join(tmpdir, "train_metrics.csv")
            eval_csv = os.path.join(tmpdir, "eval_metrics.csv")
            summary_json = os.path.join(tmpdir, "benchmark_summary.json")

            dummy_train_content = "step,time_elapsed_s,loss\n1,0.1,0.5\n2,0.2,0.4\n"
            dummy_eval_content = "step,time_elapsed_s,psnr\n100,5.0,25.4\n"
            dummy_summary_content = '{"experiment_name": "train_run", "total_steps": 30000}\n'

            with open(train_csv, "w", encoding="utf-8") as f:
                f.write(dummy_train_content)
            with open(eval_csv, "w", encoding="utf-8") as f:
                f.write(dummy_eval_content)
            with open(summary_json, "w", encoding="utf-8") as f:
                f.write(dummy_summary_content)

            # In eval_only mode, existing files must NOT be truncated or overwritten
            logger = TrainLogger(
                result_dir=tmpdir,
                exp_name="eval_run",
                eval_only=True,
                is_rank_zero=True,
            )
            self.assertTrue(logger.eval_only)

            # Log an eval step
            logger.log_eval_step(step=30000, metrics={"psnr": 26.5, "ssim": 0.85, "lpips": 0.15}, num_GS=1000)
            logger.finish(final_step=30000, final_num_gs=1000)

            # Check training files are untouched
            with open(train_csv, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), dummy_train_content)
            with open(eval_csv, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), dummy_eval_content)
            with open(summary_json, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), dummy_summary_content)

            # Check separate eval-only files exist
            eval_only_csv = os.path.join(tmpdir, "eval_only_metrics.csv")
            eval_only_summary = os.path.join(tmpdir, "eval_only_summary.json")
            self.assertTrue(os.path.exists(eval_only_csv))
            self.assertTrue(os.path.exists(eval_only_summary))


class AppearanceRenderingTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_rasterize_with_appearance_optimization_forward_and_backward(self):
        if self.device != "cuda":
            self.skipTest("CUDA required for gsplat rasterization")

        from gsplat.strategy import DefaultStrategy
        from utils import AppearanceOptModule

        count = 8
        cfg = low.Config(
            disable_viewer=True,
            app_opt=True,
            app_embed_dim=16,
            strategy=DefaultStrategy(verbose=False),
        )

        runner = object.__new__(low.Runner)
        runner.device = self.device
        runner.world_rank = 0
        runner.world_size = 1
        runner.cfg = cfg
        runner.active_sh_degree = 3
        runner.is_compressed = False

        splats = make_splats(count=count, device=self.device, sh_degree=0)
        splats["features"] = torch.nn.Parameter(torch.randn(count, 32, device=self.device))
        splats["colors"] = torch.nn.Parameter(torch.zeros(count, 3, device=self.device))
        runner.splats = splats

        runner.app_module = AppearanceOptModule(
            n=2,
            feature_dim=32,
            embed_dim=cfg.app_embed_dim,
            sh_degree=cfg.sh_degree,
        ).to(self.device)

        c2w = torch.eye(4, device=self.device).unsqueeze(0)
        K = torch.tensor([[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]], device=self.device)

        render_colors, render_alphas, info = runner.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=64,
            height=64,
            image_ids=torch.tensor([0], device=self.device),
        )
        self.assertEqual(render_colors.shape, (1, 64, 64, 3))
        loss = render_colors.sum()
        loss.backward()
        self.assertIsNotNone(splats["features"].grad)
        self.assertIsNotNone(splats["colors"].grad)


class SmallKMeansInputTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_chunked_gpu_kmeans_rejects_empty_input(self):
        empty_data = torch.empty((0, 45), device=self.device)
        with self.assertRaises(ValueError):
            low.chunked_gpu_kmeans(empty_data, num_clusters=1024)

    def test_chunked_gpu_kmeans_supports_small_input_n_less_than_k(self):
        import train_low_vram

        data = torch.randn(10, 45, device=self.device)
        centroids, indices = low.chunked_gpu_kmeans(data, num_clusters=1024, num_iters=3)
        self.assertEqual(centroids.shape, (10, 45))
        self.assertEqual(indices.shape, (10,))
        self.assertTrue((indices >= 0).all() and (indices < 10).all())

        # Test train_low_vram's implementation as well
        centroids_tlv, indices_tlv = train_low_vram.chunked_gpu_kmeans(data, num_clusters=1024, num_iters=3)
        self.assertEqual(centroids_tlv.shape, (10, 45))
        self.assertEqual(indices_tlv.shape, (10,))
        self.assertTrue((indices_tlv >= 0).all() and (indices_tlv < 10).all())

    def test_compress_sh_into_codebook_with_n_less_than_codebook_size(self):
        from gsplat.strategy import DefaultStrategy

        count = 10
        cfg = low.Config(
            disable_viewer=True,
            codebook_size=1024,
            sh_degree=3,
            strategy=DefaultStrategy(verbose=False),
        )

        runner = object.__new__(low.Runner)
        runner.device = self.device
        runner.world_rank = 0
        runner.world_size = 1
        runner.cfg = cfg
        runner.active_sh_degree = 1
        runner.is_compressed = False
        runner.sh_codebook = None
        runner.sh_indices = None
        runner.budget_controller = None

        splats = make_splats(count=count, device=self.device, sh_degree=1)
        runner.splats = splats
        runner.optimizers = {
            name: torch.optim.Adam([param]) for name, param in splats.items()
        }

        # Call compression when count (10) < codebook_size (1024)
        runner.compress_sh_into_codebook(step=7000)

        self.assertTrue(runner.is_compressed)
        self.assertEqual(runner.sh_codebook.shape, (1024, 15, 3))
        self.assertEqual(runner.sh_indices.shape, (10,))
        self.assertTrue((runner.sh_indices >= 0).all() and (runner.sh_indices < 10).all())

        if self.device == "cuda":
            c2w = torch.eye(4, device=self.device).unsqueeze(0)
            K = torch.tensor([[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]], device=self.device)
            out, _, _ = runner.rasterize_splats(camtoworlds=c2w, Ks=K, width=64, height=64)
            self.assertEqual(out.shape, (1, 64, 64, 3))


if __name__ == "__main__":
    unittest.main()
