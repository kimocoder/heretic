# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from unittest.mock import patch

from heretic.system import (
    accelerator_version_lines,
    get_accelerator_info,
    parse_cuda_toolkit_version_json,
    parse_nvidia_smi_cuda_version,
)


class NvidiaSmiCudaVersionTests(unittest.TestCase):
    def test_prefers_umd_version_on_driver_620(self) -> None:
        header = (
            "| NVIDIA-SMI 620.70.06              KMD Version: 620.70.06"
            "     CUDA UMD Version: 13.5     |"
        )
        self.assertEqual(parse_nvidia_smi_cuda_version(header), "13.5")

    def test_reads_legacy_cuda_version_header(self) -> None:
        header = (
            "| NVIDIA-SMI 550.54.15              Driver Version: 550.54.14"
            "      CUDA Version: 12.4     |"
        )
        self.assertEqual(parse_nvidia_smi_cuda_version(header), "12.4")

    def test_ignores_query_deprecation_note_when_umd_is_present(self) -> None:
        output = (
            "CUDA Version                                           :"
            " 13.5 [Deprecated; will be removed in CUDA 14.0.]\n"
            "CUDA UMD Version                                       : 13.5\n"
        )
        self.assertEqual(parse_nvidia_smi_cuda_version(output), "13.5")


class CudaToolkitVersionTests(unittest.TestCase):
    def test_reads_version_json(self) -> None:
        text = '{"cuda": {"name": "CUDA SDK", "version": "13.5.0"}}'
        self.assertEqual(parse_cuda_toolkit_version_json(text), "13.5.0")

    def test_rejects_invalid_json(self) -> None:
        self.assertIsNone(parse_cuda_toolkit_version_json("{"))


class AcceleratorVersionLineTests(unittest.TestCase):
    def test_splits_driver_cuda_from_pytorch_build(self) -> None:
        lines = accelerator_version_lines(
            {
                "api_name": "CUDA Version",
                "api_version": "13.0",
                "system_cuda_version": "13.5",
                "cuda_toolkit_version": "13.5.0",
            }
        )
        self.assertEqual(
            lines,
            [
                ("CUDA Version", "13.5"),
                ("CUDA Toolkit", "13.5.0"),
                ("PyTorch CUDA", "13.0"),
            ],
        )

    def test_hides_pytorch_line_when_it_matches_the_driver(self) -> None:
        lines = accelerator_version_lines(
            {
                "api_name": "CUDA Version",
                "api_version": "13.5",
                "system_cuda_version": "13.5",
                "cuda_toolkit_version": None,
            }
        )
        self.assertEqual(lines, [("CUDA Version", "13.5")])

    def test_falls_back_to_pytorch_when_nvidia_smi_has_no_cuda(self) -> None:
        lines = accelerator_version_lines(
            {
                "api_name": "CUDA Version",
                "api_version": "13.0",
                "system_cuda_version": None,
                "cuda_toolkit_version": None,
            }
        )
        self.assertEqual(lines, [("CUDA Version", "13.0")])

    def test_report_uses_the_driver_cuda_version(self) -> None:
        info = {
            "type": "CUDA",
            "api_name": "CUDA Version",
            "api_version": "13.0",
            "system_cuda_version": "13.5",
            "cuda_toolkit_version": "13.5.0",
            "driver_version": "620.70.06",
            "devices": [{"name": "NVIDIA GeForce RTX 3090 Ti", "vram_gb": 23.57}],
        }
        with patch("heretic.system.get_accelerator_info_dict", return_value=info):
            report = get_accelerator_info(include_warnings=False)
        self.assertIn("CUDA Version: [bold]13.5[/]", report)
        self.assertIn("CUDA Toolkit: [bold]13.5.0[/]", report)
        self.assertIn("PyTorch CUDA: [bold]13.0[/]", report)
        self.assertNotIn("CUDA Version: [bold]13.0[/]", report)
