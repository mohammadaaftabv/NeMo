# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class RangeFst(GraphFst):
    """
    time: time FST
    deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, time: GraphFst, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)
        self.graph = pynini.accep("")
        if not deterministic:
            self.graph |= time + delete_space + pynini.cross("-", " to ") + delete_space + time
            cardinal = cardinal.graph
            range_graph = cardinal + delete_space + pynini.cross("-", " minus ") + delete_space + cardinal

            for x in ["+", " + "]:
                range_graph |= cardinal + pynini.closure(pynini.cross(x, " plus ") + cardinal, 1)
            for x in ["/", " / "]:
                range_graph |= cardinal + pynini.closure(pynini.cross(x, " divided by ") + cardinal, 1)
            for x in [" x ", "x"]:
                range_graph |= cardinal + pynini.closure(
                    pynini.cross(x, pynini.union(" by ", " times ")) + cardinal, 1
                )
            for x in ["*", " * "]:
                range_graph |= cardinal + pynini.closure(pynini.cross(x, pynini.union(" times ")) + cardinal, 1)
            self.graph |= range_graph
            self.graph = self.graph.optimize()
        graph = pynutil.insert("value: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()