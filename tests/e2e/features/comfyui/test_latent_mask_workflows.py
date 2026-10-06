# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from graphlib import TopologicalSorter
from pathlib import Path

import pytest
from comfyui_vllm_omni import nodes as omni_nodes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW_DIR = Path(__file__).resolve().parents[4] / "apps/ComfyUI-vLLM-Omni/example_workflows"
IMAGE_MASK = "vLLM-Omni Latent Mask Editing - Image Mask.json"
TEMPORAL_MASK = "vLLM-Omni Latent Mask Editing - Temporal Mask.json"


def _load(name: str) -> dict:
    return json.loads((WORKFLOW_DIR / name).read_text())


def _source_of(workflow: dict, node: dict, input_name: str) -> dict:
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    link_id = next(i["link"] for i in node["inputs"] if i["name"] == input_name)
    assert link_id is not None, f"{node['type']}.{input_name} is not connected"
    return nodes[links[link_id][1]]


@pytest.mark.parametrize("name", [IMAGE_MASK, TEMPORAL_MASK])
def test_latent_mask_workflow_connections(name):
    workflow = _load(name)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert len(nodes) == len(workflow["nodes"])
    assert len(links) == len(workflow["links"])
    graph: dict[int, set[int]] = {node_id: set() for node_id in nodes}
    for link_id, source, output_slot, target, input_slot, kind in links.values():
        output = nodes[source]["outputs"][output_slot]
        input_ = nodes[target]["inputs"][input_slot]
        assert output["type"] == input_["type"] == kind
        assert link_id in output["links"]
        assert input_["link"] == link_id
        graph[target].add(source)
    assert len(tuple(TopologicalSorter(graph).static_order())) == len(nodes)


@pytest.mark.parametrize("name", [IMAGE_MASK, TEMPORAL_MASK])
def test_latent_mask_workflow_matches_omni_node_interfaces(name):
    for node in _load(name)["nodes"]:
        if not node["type"].startswith("VLLMOmni"):
            continue
        cls = getattr(omni_nodes, node["type"])
        schema = cls.INPUT_TYPES()
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        for input_ in node["inputs"]:
            kind = inputs[input_["name"]][0]
            assert input_["type"] == ("COMBO" if isinstance(kind, list) else kind)
        assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES


@pytest.mark.parametrize("name", [IMAGE_MASK, TEMPORAL_MASK])
def test_latent_mask_workflow_uses_h3_canvas(name):
    (generate,) = [n for n in _load(name)["nodes"] if n["type"] == "VLLMOmniGenerateVideo"]
    assert generate["widgets_values"][4:6] == [1344, 768]


def test_temporal_mask_workflow_builds_per_frame_mask():
    # The server pads a short [T, H, W] mask with its last slice, so the mask must
    # come from the Temporal Mask node (one slice per output frame), not a
    # hand-batched SolidMask stack.
    workflow = _load(TEMPORAL_MASK)
    (edit,) = [n for n in workflow["nodes"] if n["type"] == "VLLMOmniLatentMaskEditing"]
    (generate,) = [n for n in workflow["nodes"] if n["type"] == "VLLMOmniGenerateVideo"]
    temporal = _source_of(workflow, edit, "video_mask")
    assert temporal["type"] == "VLLMOmniMiniMaxH3TemporalMask"
    assert _source_of(workflow, temporal, "images")["type"] == "GetVideoComponents"
    assert _source_of(workflow, edit, "source_video")["type"] == "LoadVideo"
    source_fps, duration, mode, preserve_fraction = temporal["widgets_values"]
    assert duration == generate["widgets_values"][7]
    assert mode == "continuation" and 0 < preserve_fraction < 1
