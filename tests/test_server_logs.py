from __future__ import annotations

from io import StringIO
import unittest

from llama_model_manager.server_logs import LlamaServerLogFormatter
from llama_model_manager.ui import TerminalUI


class ServerLogFormatterTests(unittest.TestCase):
    def formatter(self) -> tuple[LlamaServerLogFormatter, StringIO]:
        stream = StringIO()
        ui = TerminalUI(stream=stream, no_color=True, plain=True)
        return LlamaServerLogFormatter(ui), stream

    def test_request_summary_compacts_moe_diagnostics(self) -> None:
        formatter, stream = self.formatter()
        lines = [
            "0.00 I slot launch_slot_: id 0 | task 125 | processing task, is_child = 0\n",
            "moe-prepack: disabled reason=requires single-row independent or bounded speculative MAIN decode\n",
            "moe-prepack: disabled reason=requires single-row independent or bounded speculative MAIN decode\n",
            "0.51 I slot print_timing: id 0 | task 125 | prompt eval time = 4067.93 ms / 10521 tokens (0.39 ms per token, 2586.33 tokens per second)\n",
            "0.51 I slot print_timing: id 0 | task 125 | eval time = 11937.15 ms / 1035 tokens (11.54 ms per token, 86.62 tokens per second)\n",
            "0.51 I slot print_timing: id 0 | task 125 | total time = 16005.08 ms / 11556 tokens\n",
            "0.51 I slot print_timing: id 0 | task 125 | graphs reused = 554\n",
            "0.51 I slot print_timing: id 0 | task 125 | draft acceptance = 0.44294 (590 accepted / 1332 generated), mean len = 2.33\n",
            "moe-grouped-decode: fallback=0 rollback=0 required_unsupported=0 prepare_error=0 finish_error=0 populated_slots=2163 slot_capacity=2173 populated_payload_bytes=4127440896 payload_capacity_bytes=4145946624\n",
            "moe-grouped-paths: captures=213 replays=19282 cache_hits=316838 cache_misses=96936\n",
            "0.51 I slot release: id 0 | task 125 | stop processing: n_tokens = 33258, truncated = 0\n",
        ]
        for line in lines:
            formatter.handle(line)

        output = stream.getvalue()
        self.assertIn("REQUEST 125", output)
        self.assertIn("10,521 tok", output)
        self.assertIn("86.62 t/s", output)
        self.assertIn("44.29%", output)
        self.assertIn("2,163/2,173 slots", output)
        self.assertIn("cumulative · 316,838 hits · 96,936 misses · 76.6%", output)
        self.assertEqual(output.count("MoE prepack"), 1)
        self.assertNotIn("moe-grouped-decode:", output)

    def test_cuda_oom_is_never_hidden(self) -> None:
        formatter, stream = self.formatter()
        formatter.handle("0.12.000 E ggml-cuda CUDA error: out of memory\n")
        self.assertIn("CUDA error: out of memory", stream.getvalue())

    def test_raw_output_passes_through(self) -> None:
        stream = StringIO()
        ui = TerminalUI(stream=stream, no_color=True, plain=True, raw_output=True)
        formatter = LlamaServerLogFormatter(ui, raw_output=True)
        line = "moe-cache-phase: phase=decode ops=0\n"
        formatter.handle(line)
        self.assertEqual(stream.getvalue(), line)


if __name__ == "__main__":
    unittest.main()
