"""Small native-model output schemas; semantic validation remains mandatory."""


def closed(properties):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


PATCH_SCHEMA = closed(
    {
        "operations": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "anyOf": [
                    closed(
                        {
                            "op": {"const": "move"},
                            "target_id": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        }
                    ),
                    closed(
                        {
                            "op": {"const": "resize"},
                            "target_id": {"type": "string"},
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                        }
                    ),
                    closed(
                        {
                            "op": {"const": "set_geometry"},
                            "target_id": {"type": "string"},
                            "geometry": closed(
                                {
                                    "shape": {
                                        "enum": [
                                            "rect",
                                            "circle",
                                            "ellipse",
                                            "diamond",
                                            "triangle",
                                        ]
                                    }
                                }
                            ),
                        }
                    ),
                    closed(
                        {
                            "op": {"const": "set_style"},
                            "target_id": {"type": "string"},
                            "style": closed(
                                {
                                    "font_size": {"type": ["number", "null"]},
                                    "font_weight": {"type": ["number", "null"]},
                                    "radius": {"type": ["number", "null"]},
                                    "stroke_width": {"type": ["number", "null"]},
                                }
                            ),
                        }
                    ),
                ]
            },
        },
        "concept": closed(
            {
                "title": {"type": "string"},
                "medium": {"type": "string"},
                "metaphor": {"type": "string"},
                "composition": {"type": "string"},
                "takeaway": {"type": "string"},
            }
        ),
    }
)

REPAIR_SCHEMA = closed(
    {
        "operations": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "anyOf": [
                    *[
                        operation
                        for operation in PATCH_SCHEMA["properties"]["operations"][
                            "items"
                        ]["anyOf"]
                        if operation["properties"]["op"]["const"]
                        in {"move", "resize", "set_style"}
                    ],
                    closed(
                        {
                            "op": {"const": "replace_text"},
                            "target_id": {"type": "string"},
                            "text": {"type": "string", "maxLength": 90},
                        }
                    ),
                    closed(
                        {
                            "op": {"const": "set_motion"},
                            "target_id": {"type": "string"},
                            "keyframes": {
                                "type": "array",
                                "maxItems": 8,
                                "items": closed(
                                    {
                                        "t": {"type": "number"},
                                        "value": {"type": "number"},
                                        "easing": {
                                            "enum": [
                                                "linear",
                                                "ease_in",
                                                "ease_out",
                                                "ease_in_out",
                                                "spring_gentle",
                                                "spring_snappy",
                                            ]
                                        },
                                    }
                                ),
                            },
                        }
                    ),
                ]
            },
        },
        "counterexamples": {
            "type": "array",
            "maxItems": 3,
            "items": closed(
                {
                    "frame_index": {"type": "integer", "minimum": 1, "maximum": 24},
                    "target_id": {"type": "string", "maxLength": 96},
                    "observed": {"type": "string", "maxLength": 120},
                    "requirement": {"type": "string", "maxLength": 120},
                }
            ),
        },
    }
)
