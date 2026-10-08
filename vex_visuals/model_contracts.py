"""Small native-model output schemas; semantic validation remains mandatory."""
def closed(properties):
    return {"type":"object","additionalProperties":False,"required":list(properties),"properties":properties}


PATCH_SCHEMA=closed({
    "operations":{"type":"array","maxItems":4,"items":{"anyOf":[
        closed({"op":{"const":"move"},"target_id":{"type":"string"},"x":{"type":"number"},"y":{"type":"number"}}),
        closed({"op":{"const":"resize"},"target_id":{"type":"string"},"width":{"type":"number"},"height":{"type":"number"}}),
        closed({"op":{"const":"set_geometry"},"target_id":{"type":"string"},"geometry":closed({"shape":{"enum":["rect","circle","ellipse","diamond","triangle"]}})}),
        closed({"op":{"const":"set_style"},"target_id":{"type":"string"},"style":closed({"font_size":{"type":["number","null"]},"font_weight":{"type":["number","null"]},"radius":{"type":["number","null"]},"stroke_width":{"type":["number","null"]}})}),
    ]}},
    "concept":closed({"title":{"type":"string"},"medium":{"type":"string"},"metaphor":{"type":"string"},"composition":{"type":"string"},"takeaway":{"type":"string"}}),
})
