"""Model output widths and indexer candidate communication regressions."""

import itertools
import unittest

from srammachine.frontend.modules import (
    BMMOp, FlashAttentionOp, FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp,
)
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import MiniMaxM3Mapper, mapper_for_model
from srammachine.pipetree import TreeParser


class OutputPrecisionTests(unittest.TestCase):
    def test_model_bmm_outputs_and_indexer_candidate_traffic(self):
        models = ("deepseek-v3", "deepseek-v3.2", "glm-5.1", "minimax-m3")
        for model, fold, dtype, strategy in itertools.product(
            models, (True, False), ("fp8", "fp16"), MoEParallelStrategy,
        ):
            with self.subTest(model=model, fold=fold, dtype=dtype, strategy=strategy):
                mapper = mapper_for_model(model)
                inference = InferenceConfig(32, 256, 1, strategy, dtype)
                mapped = mapper.map(inference, xfold=fold)
                graph = TreeParser(mapper.hardware_config).parse(
                    mapped.root_node, mapped.operators, mapped.operator_mappings,
                    layer_count=1,
                )
                for op in mapped.op_list:
                    if isinstance(op, BMMOp):
                        self.assertEqual(op.input_dtype_bytes, 1, op.op_id)
                        self.assertEqual(op.output_dtype_bytes, 2, op.op_id)
                        if isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)):
                            self.assertEqual(op.index_dtype_bytes, 4)
                            self.assertEqual(op.weight_dtype_bytes,
                                             inference.kv_cache_bytes_per_element)
                            outputs = [command for command in graph.commands
                                       if command.op_id == op.op_id
                                       and command.cmd_id.endswith(".output")]
                            self.assertTrue(outputs)
                            for command in outputs:
                                self.assertEqual(command.size_bytes,
                                                 max(command.topk_candidates_per_row) * 6)
                        else:
                            self.assertEqual(op.weight_dtype_bytes, 1, op.op_id)
                    elif isinstance(op, FlashAttentionOp):
                        self.assertEqual(op.output_dtype_bytes, 1)
                        self.assertEqual(op.state_dtype_bytes, 4)
                        self.assertEqual(op.kv_cache_dtype_bytes,
                                         inference.kv_cache_bytes_per_element)

    def test_minimax_dense_bmm_outputs(self):
        ops = MiniMaxM3Mapper(layer_type="dense").build_op_list(InferenceConfig(32, 256, 1))
        bmms = [op for op in ops if isinstance(op, BMMOp)]
        self.assertTrue(bmms)
        for op in bmms:
            self.assertEqual(op.output_dtype_bytes, 2, op.op_id)


if __name__ == "__main__":
    unittest.main()
