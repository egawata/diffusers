# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for BREAK keyword support in lpw_stable_diffusion_xl.py
"""

import unittest

from lpw_stable_diffusion_xl import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    group_tokens_and_weights,
    parse_prompt_attention,
)


class TestParsePromptAttentionBreak(unittest.TestCase):
    """Tests for BREAK keyword detection in parse_prompt_attention"""

    def test_break_keyword_detected(self):
        """BREAK keyword is detected and returns weight=-1"""
        result = parse_prompt_attention("cat BREAK dog")
        self.assertIn(["BREAK", -1], result)

    def test_break_with_surrounding_spaces(self):
        """BREAK is detected even with extra surrounding spaces"""
        result = parse_prompt_attention("cat   BREAK   dog")
        self.assertIn(["BREAK", -1], result)

    def test_break_at_start(self):
        """BREAK at the start of prompt"""
        result = parse_prompt_attention("BREAK dog")
        self.assertIn(["BREAK", -1], result)

    def test_break_at_end(self):
        """BREAK at the end of prompt"""
        result = parse_prompt_attention("cat BREAK")
        self.assertIn(["BREAK", -1], result)

    def test_multiple_breaks(self):
        """Multiple BREAK keywords are all detected"""
        result = parse_prompt_attention("cat BREAK dog BREAK bird")
        break_count = sum(1 for item in result if item == ["BREAK", -1])
        self.assertEqual(break_count, 2)

    def test_break_not_in_word(self):
        """BREAK inside a word (like BREAKFAST) should not be detected as BREAK keyword"""
        result = parse_prompt_attention("BREAKFAST")
        break_entries = [r for r in result if r[0] == "BREAK" and r[1] == -1]
        self.assertEqual(len(break_entries), 0)

    def test_break_case_sensitive(self):
        """BREAK is case-sensitive - lowercase 'break' should not be detected"""
        result = parse_prompt_attention("cat break dog")
        break_entries = [r for r in result if r[0] == "BREAK" and r[1] == -1]
        self.assertEqual(len(break_entries), 0)

    def test_break_with_weights(self):
        """BREAK works alongside weighted tokens"""
        result = parse_prompt_attention("(cat:1.5) BREAK dog")
        self.assertIn(["BREAK", -1], result)
        # Check that weighted token is also present
        cat_entries = [r for r in result if "cat" in r[0]]
        self.assertTrue(len(cat_entries) > 0)


class TestGroupTokensAndWeightsBreak(unittest.TestCase):
    """Tests for BREAK marker handling in group_tokens_and_weights"""

    def test_break_creates_two_chunks(self):
        """BREAK marker (-1) splits tokens into separate chunks"""
        # 3 tokens + BREAK + 3 tokens
        token_ids = [100, 101, 102, -1, 200, 201, 202]
        weights = [1.0, 1.0, 1.0, -1, 1.0, 1.0, 1.0]

        result_tokens, result_weights = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # Should be split into 2 chunks
        self.assertEqual(len(result_tokens), 2)
        self.assertEqual(len(result_weights), 2)

    def test_chunk_structure_with_break(self):
        """Each chunk has correct structure: BOS + 75 tokens + EOS = 77 total"""
        token_ids = [100, 101, 102, -1, 200, 201, 202]
        weights = [1.0, 1.0, 1.0, -1, 1.0, 1.0, 1.0]

        result_tokens, result_weights = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # Each chunk should be 77 tokens
        for chunk in result_tokens:
            self.assertEqual(len(chunk), 77)
        for chunk in result_weights:
            self.assertEqual(len(chunk), 77)

        # First and last tokens should be BOS and EOS
        for chunk in result_tokens:
            self.assertEqual(chunk[0], BOS_TOKEN_ID)
            self.assertEqual(chunk[-1], EOS_TOKEN_ID)

    def test_tokens_in_correct_chunks(self):
        """Tokens appear in correct chunks after BREAK split"""
        token_ids = [100, 101, -1, 200, 201]
        weights = [1.0, 1.0, -1, 1.0, 1.0]

        result_tokens, _ = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # First chunk should contain 100, 101
        self.assertIn(100, result_tokens[0])
        self.assertIn(101, result_tokens[0])
        self.assertNotIn(200, result_tokens[0])

        # Second chunk should contain 200, 201
        self.assertIn(200, result_tokens[1])
        self.assertIn(201, result_tokens[1])
        self.assertNotIn(100, result_tokens[1])

    def test_break_at_start_creates_empty_chunk(self):
        """BREAK at start creates an empty (padded) chunk first"""
        token_ids = [-1, 100, 101]
        weights = [-1, 1.0, 1.0]

        result_tokens, _ = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # Should create 2 chunks: empty chunk + actual tokens
        self.assertEqual(len(result_tokens), 2)

        # First chunk should be all padding (EOS tokens after BOS)
        first_chunk_content = result_tokens[0][1:-1]  # exclude BOS and final EOS
        self.assertTrue(all(t == EOS_TOKEN_ID for t in first_chunk_content))

    def test_multiple_breaks(self):
        """Multiple BREAKs create multiple chunks"""
        token_ids = [100, -1, 200, -1, 300]
        weights = [1.0, -1, 1.0, -1, 1.0]

        result_tokens, _ = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # Should be 3 chunks
        self.assertEqual(len(result_tokens), 3)

        # Verify each token is in correct chunk
        self.assertIn(100, result_tokens[0])
        self.assertIn(200, result_tokens[1])
        self.assertIn(300, result_tokens[2])

    def test_consecutive_breaks(self):
        """Consecutive BREAKs create empty chunks between them"""
        token_ids = [100, -1, -1, 200]
        weights = [1.0, -1, -1, 1.0]

        result_tokens, _ = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # Should be 3 chunks: [100], [empty], [200]
        self.assertEqual(len(result_tokens), 3)

    def test_break_preserves_weights(self):
        """Token weights are preserved correctly after BREAK split"""
        token_ids = [100, 101, -1, 200]
        weights = [1.5, 2.0, -1, 0.5]

        result_tokens, result_weights = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # Find positions of actual tokens and check weights
        # First chunk: tokens at positions 1 and 2 (after BOS)
        self.assertEqual(result_weights[0][1], 1.5)
        self.assertEqual(result_weights[0][2], 2.0)

        # Second chunk: token at position 1 (after BOS)
        self.assertEqual(result_weights[1][1], 0.5)

    def test_pad_last_block_false(self):
        """With pad_last_block=False, last chunk is not padded to 75"""
        token_ids = [100, 101, -1, 200]
        weights = [1.0, 1.0, -1, 1.0]

        result_tokens, result_weights = group_tokens_and_weights(token_ids, weights, pad_last_block=False)

        # First chunk (before BREAK) should still be padded to 77
        self.assertEqual(len(result_tokens[0]), 77)

        # Last chunk should NOT be padded - just BOS + 1 token + EOS = 3
        self.assertEqual(len(result_tokens[1]), 3)

    def test_no_break_single_chunk(self):
        """Without BREAK, tokens stay in single chunk"""
        token_ids = [100, 101, 102]
        weights = [1.0, 1.0, 1.0]

        result_tokens, _ = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        self.assertEqual(len(result_tokens), 1)
        self.assertIn(100, result_tokens[0])
        self.assertIn(101, result_tokens[0])
        self.assertIn(102, result_tokens[0])

    def test_exactly_75_tokens_with_break(self):
        """Exactly 75 tokens before BREAK: fills one chunk, BREAK creates empty chunk"""
        token_ids = list(range(100, 175)) + [-1] + [200, 201]  # 75 tokens + BREAK + 2 tokens
        weights = [1.0] * 75 + [-1] + [1.0, 1.0]

        result_tokens, _ = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # Should be 3 chunks:
        # 1. First 75 tokens (auto-finalized when reaching 75)
        # 2. Empty chunk (created by BREAK)
        # 3. Remaining 2 tokens
        self.assertEqual(len(result_tokens), 3)

        # First chunk should have all 75 original tokens
        first_chunk_content = result_tokens[0][1:-1]  # exclude BOS and EOS
        for i, token in enumerate(range(100, 175)):
            self.assertEqual(first_chunk_content[i], token)

        # Second chunk should be empty (all padding)
        second_chunk_content = result_tokens[1][1:-1]
        self.assertTrue(all(t == EOS_TOKEN_ID for t in second_chunk_content))

        # Third chunk should have 200, 201
        self.assertIn(200, result_tokens[2])
        self.assertIn(201, result_tokens[2])


class TestGroupTokensAndWeightsGeneral(unittest.TestCase):
    """General tests for group_tokens_and_weights (non-BREAK related)"""

    def test_empty_input(self):
        """Empty input still produces one chunk"""
        result_tokens, result_weights = group_tokens_and_weights([], [], pad_last_block=True)

        # Should produce one empty (padded) chunk
        self.assertEqual(len(result_tokens), 1)
        self.assertEqual(len(result_tokens[0]), 77)

    def test_over_75_tokens_auto_split(self):
        """More than 75 tokens automatically splits into multiple chunks"""
        token_ids = list(range(100, 260))  # 160 tokens
        weights = [1.0] * 160

        result_tokens, _ = group_tokens_and_weights(token_ids, weights, pad_last_block=True)

        # 160 tokens = 2 full chunks (75 each) + 1 partial chunk (10 tokens)
        self.assertEqual(len(result_tokens), 3)


if __name__ == "__main__":
    unittest.main()
