from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from vex_manim.briefs import SceneBrief


@dataclass
class BlueprintElement:
    element_id: str
    role: str
    kind: str
    placement: str
    motion: str
    notes: str = ""
    copy_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MotionBeat:
    beat_id: str
    purpose: str
    focus: str
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneBlueprint:
    blueprint_id: str
    scene_family: str
    archetype: str
    rationale: str
    layout_thesis: str
    focal_system: str
    background_system: str
    camera_plan: str
    title_strategy: str
    copy_strategy: list[str] = field(default_factory=list)
    dynamic_devices: list[str] = field(default_factory=list)
    suggested_features: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    elements: list[BlueprintElement] = field(default_factory=list)
    motion_beats: list[MotionBeat] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "scene_family": self.scene_family,
            "archetype": self.archetype,
            "rationale": self.rationale,
            "layout_thesis": self.layout_thesis,
            "focal_system": self.focal_system,
            "background_system": self.background_system,
            "camera_plan": self.camera_plan,
            "title_strategy": self.title_strategy,
            "copy_strategy": list(self.copy_strategy),
            "dynamic_devices": list(self.dynamic_devices),
            "suggested_features": list(self.suggested_features),
            "anti_patterns": list(self.anti_patterns),
            "elements": [element.to_dict() for element in self.elements],
            "motion_beats": [beat.to_dict() for beat in self.motion_beats],
            "score": round(float(self.score or 0.0), 3),
        }

    def prompt_terms(self) -> set[str]:
        phrases = [
            self.scene_family,
            self.archetype,
            self.focal_system,
            self.background_system,
            self.camera_plan,
            *self.dynamic_devices,
            *self.suggested_features,
            *(element.kind for element in self.elements),
            *(element.role for element in self.elements),
        ]
        normalized: set[str] = set()
        for phrase in phrases:
            text = str(phrase or "").strip().lower()
            if not text:
                continue
            normalized.add(text.replace(" ", "_"))
            for token in re.findall(r"[a-z0-9_]+", text):
                if len(token) >= 4 or token in {"ui", "path", "beam", "ring"}:
                    normalized.add(token)
        return normalized

    def to_prompt_block(self) -> str:
        lines = [
            f"Blueprint: {self.blueprint_id} ({self.archetype})",
            f"Rationale: {self.rationale}",
            f"Layout thesis: {self.layout_thesis}",
            f"Focal system: {self.focal_system}",
            f"Background system: {self.background_system}",
            f"Camera plan: {self.camera_plan}",
            f"Title strategy: {self.title_strategy}",
            f"Dynamic devices: {', '.join(self.dynamic_devices)}",
            f"Suggested Manim features: {', '.join(self.suggested_features)}",
            "Copy strategy:",
        ]
        lines.extend(f"- {item}" for item in self.copy_strategy)
        lines.append("Elements:")
        lines.extend(
            f"- {element.element_id}: role={element.role}, kind={element.kind}, placement={element.placement}, motion={element.motion}, source={element.copy_source or 'none'}"
            for element in self.elements
        )
        lines.append("Motion beats:")
        lines.extend(
            f"- {beat.beat_id}: {beat.purpose}; focus={beat.focus}; actions={', '.join(beat.actions)}"
            for beat in self.motion_beats
        )
        if self.anti_patterns:
            lines.append("Avoid:")
            lines.extend(f"- {item}" for item in self.anti_patterns)
        return "\n".join(lines)


def _element(
    element_id: str,
    *,
    role: str,
    kind: str,
    placement: str,
    motion: str,
    notes: str = "",
    copy_source: str = "",
) -> BlueprintElement:
    return BlueprintElement(
        element_id=element_id,
        role=role,
        kind=kind,
        placement=placement,
        motion=motion,
        notes=notes,
        copy_source=copy_source,
    )


def _beat(beat_id: str, *, purpose: str, focus: str, actions: list[str]) -> MotionBeat:
    return MotionBeat(beat_id=beat_id, purpose=purpose, focus=focus, actions=list(actions))


def _blueprint(
    brief: SceneBrief,
    *,
    blueprint_id: str,
    archetype: str,
    rationale: str,
    layout_thesis: str,
    focal_system: str,
    background_system: str,
    camera_plan: str,
    title_strategy: str,
    copy_strategy: list[str],
    dynamic_devices: list[str],
    suggested_features: list[str],
    anti_patterns: list[str],
    elements: list[BlueprintElement],
    motion_beats: list[MotionBeat],
) -> SceneBlueprint:
    return SceneBlueprint(
        blueprint_id=blueprint_id,
        scene_family=brief.scene_family,
        archetype=archetype,
        rationale=rationale,
        layout_thesis=layout_thesis,
        focal_system=focal_system,
        background_system=background_system,
        camera_plan=camera_plan,
        title_strategy=title_strategy,
        copy_strategy=copy_strategy,
        dynamic_devices=dynamic_devices,
        suggested_features=suggested_features,
        anti_patterns=anti_patterns,
        elements=elements,
        motion_beats=motion_beats,
    )


def _metric_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::metric_orbit_axis",
            archetype="metric_orbit_axis",
            rationale="Turns the spoken claim into a discovered metric by tying a live tracker to a path and a chart.",
            layout_thesis="Hero metric in one corner, evidence geometry across the opposite half, with one traveling pulse binding them together.",
            focal_system="tracked metric badge + axis sweep",
            background_system="orbital rings with a restrained focus beam",
            camera_plan="Start wide for the title, then punch into the chart once the tracker starts climbing.",
            title_strategy="Compact editorial eyebrow plus a narrow hero line anchored in the top-left band.",
            copy_strategy=[
                "Use the headline as a short claim, not a sentence block.",
                "Move supporting copy into small labels or metric badges near the evidence geometry.",
                "Let the metric itself be the dominant readable text after the title enters.",
            ],
            dynamic_devices=["camera_reframe", "value_tracker", "move_along_path", "always_redraw", "orbit_ring"],
            suggested_features=["MovingCameraScene", "ValueTracker", "Axes", "always_redraw", "MoveAlongPath", "TracedPath"],
            anti_patterns=[
                "big centered statistic with no evidence geometry",
                "stacked cards explaining the metric in prose",
            ],
            elements=[
                _element("title_band", role="title", kind="editorial_band", placement="top_left", motion="stagger_in", copy_source="headline"),
                _element("hero_metric", role="metric", kind="metric_badge", placement="upper_left", motion="tracker_driven", copy_source="headline"),
                _element("evidence_axis", role="chart", kind="axes_path_bundle", placement="center_right", motion="draw_and_track", copy_source="supporting_lines"),
                _element("signal_pulse", role="diagram", kind="traveling_glow", placement="along_path", motion="move_along_path"),
                _element("depth_orbits", role="background", kind="orbit_rings", placement="rear_right", motion="slow_drift"),
            ],
            motion_beats=[
                _beat("establish", purpose="Set the claim and visual frame", focus="title_band", actions=["fade_in_title", "fade_in_depth"]),
                _beat("prove", purpose="Show the metric being earned", focus="evidence_axis", actions=["draw_axes", "animate_tracker", "travel_pulse"]),
                _beat("land", purpose="Connect the metric back to the claim", focus="hero_metric", actions=["camera_punch_in", "badge_emphasis"]),
            ],
        ),
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::metric_bridge_morph",
            archetype="metric_bridge_morph",
            rationale="Uses before/after geometry and a bridge path so the scene explains the change rather than merely announcing it.",
            layout_thesis="Left side establishes the weaker state, right side resolves into the stronger state, and the metric bridges them.",
            focal_system="matched-shape comparison with a metric bridge",
            background_system="angled beams and a low-opacity measurement grid",
            camera_plan="Hold a composed frame first, then slide attention toward the winning state during the morph.",
            title_strategy="Title sits above the morph with compact deck text tucked beneath it.",
            copy_strategy=[
                "Keep both states to short labels or chips.",
                "Use one short winning phrase on the resolved side instead of explanatory paragraphs.",
                "Reserve the largest text treatment for the metric bridge or final outcome.",
            ],
            dynamic_devices=["transform_matching", "camera_slide", "focus_beam", "value_tracker"],
            suggested_features=["TransformMatchingShapes", "ReplacementTransform", "MovingCameraScene", "ValueTracker", "LaggedStart"],
            anti_patterns=[
                "literal vs cards with paragraph blocks",
                "duplicating the same copy on both sides of the comparison",
            ],
            elements=[
                _element("before_cluster", role="hero", kind="state_cluster", placement="left", motion="morph_out", copy_source="left_detail"),
                _element("after_cluster", role="hero", kind="state_cluster", placement="right", motion="morph_in", copy_source="right_detail"),
                _element("metric_bridge", role="metric", kind="bridge_counter", placement="center", motion="tracker_driven", copy_source="headline"),
                _element("focus_beam", role="background", kind="focus_beam", placement="center", motion="rise_and_dim"),
            ],
            motion_beats=[
                _beat("frame", purpose="Set the before state and claim", focus="before_cluster", actions=["fade_in_title", "fade_in_before"]),
                _beat("transform", purpose="Carry the viewer through the change", focus="metric_bridge", actions=["bridge_draw", "shape_morph", "camera_slide"]),
                _beat("resolve", purpose="Land on the stronger state", focus="after_cluster", actions=["metric_lock", "after_glow"]),
            ],
        ),
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::metric_ribbon_sweep",
            archetype="metric_ribbon_sweep",
            rationale="Keeps the shot premium with directional ribbons, tracked counters, and asymmetrical evidence instead of a dashboard wall.",
            layout_thesis="A ribbon path sweeps across the frame carrying the main number while evidence modules stay sparse and offset.",
            focal_system="ribbon-guided metric sweep",
            background_system="constellation dots with a soft diagonal wash",
            camera_plan="Follow the ribbon motion with a restrained frame drift rather than a hard zoom.",
            title_strategy="Small eyebrow and deck stacked above the ribbon start point.",
            copy_strategy=[
                "Split the phrase into one hero line and one deck line.",
                "Use secondary copy as tiny callout pills near the ribbon checkpoints.",
                "Keep labels visually attached to geometry rather than floating paragraphs.",
            ],
            dynamic_devices=["move_along_path", "camera_drift", "metric_badge", "lagged_callouts"],
            suggested_features=["MoveAlongPath", "LaggedStart", "always_redraw", "MovingCameraScene", "FadeTransform"],
            anti_patterns=[
                "four equally weighted info cards",
                "centered headline plus bottom caption with no motion spine",
            ],
            elements=[
                _element("title_stack", role="title", kind="editorial_stack", placement="upper_left", motion="stagger_in", copy_source="headline"),
                _element("ribbon_route", role="chart", kind="route_ribbon", placement="diagonal_midframe", motion="sweep", copy_source="supporting_lines"),
                _element("travel_badge", role="metric", kind="metric_badge", placement="along_ribbon", motion="path_travel", copy_source="headline"),
                _element("checkpoint_callouts", role="support", kind="pills", placement="near_checkpoints", motion="lagged_reveal", copy_source="supporting_lines"),
            ],
            motion_beats=[
                _beat("enter", purpose="Establish the path and claim", focus="title_stack", actions=["fade_in_title", "draw_route"]),
                _beat("travel", purpose="Move the viewer through the evidence", focus="travel_badge", actions=["move_along_path", "reveal_callouts"]),
                _beat("lock", purpose="Resolve into a clean final state", focus="metric_badge", actions=["camera_drift", "settle_badge"]),
            ],
        ),
    ]


def _dashboard_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::dashboard_signal_wall",
            archetype="dashboard_signal_wall",
            rationale="Treats multiple metrics like a living signal wall with one dominant hero instead of equal-sized tiles.",
            layout_thesis="One oversized hero metric anchors the frame while smaller evidence strips cascade around it with clear hierarchy.",
            focal_system="hero metric with cascading evidence strips",
            background_system="thin grid lines and traveling scan light",
            camera_plan="Punch into the hero first, then widen enough to reveal the supporting strips.",
            title_strategy="Top-left title sits outside the metric field and stays compact.",
            copy_strategy=[
                "One short hero phrase plus short metric labels.",
                "Support strips should use chips and numerals, not sentence-level copy.",
                "Keep the wall asymmetric so the hero is obvious at a glance.",
            ],
            dynamic_devices=["camera_reframe", "value_tracker", "scan_light", "lagged_reveals"],
            suggested_features=["MovingCameraScene", "ValueTracker", "LaggedStart", "always_redraw", "Axes"],
            anti_patterns=["flat equal dashboard tiles", "dense micro-stat clutter"],
            elements=[
                _element("hero_metric", role="metric", kind="hero_counter", placement="left_center", motion="tracker_driven", copy_source="headline"),
                _element("evidence_strips", role="chart", kind="strip_modules", placement="right_cascade", motion="lagged_reveal", copy_source="supporting_lines"),
                _element("scan_light", role="background", kind="focus_beam", placement="across_modules", motion="sweep"),
            ],
            motion_beats=[
                _beat("arrive", purpose="Introduce the claim", focus="hero_metric", actions=["fade_in_title", "fade_in_hero"]),
                _beat("cascade", purpose="Reveal the supporting evidence", focus="evidence_strips", actions=["lagged_reveal", "scan_sweep"]),
                _beat("emphasize", purpose="Return focus to the hero metric", focus="hero_metric", actions=["camera_reframe", "tracker_lock"]),
            ],
        )
    ]


def _timeline_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::route_journey_pan",
            archetype="route_journey_pan",
            rationale="Makes the process feel like a guided route with a traveling marker and moving frame, not a row of cards.",
            layout_thesis="A curved route spans the frame with milestones offset above and below it for rhythm and readable progression.",
            focal_system="traveling route marker with milestone ribbons",
            background_system="soft orbital echoes behind the route",
            camera_plan="Track along the route and settle near the decisive milestone.",
            title_strategy="Title sits near the route origin and exits visual dominance once the journey begins.",
            copy_strategy=[
                "Milestones should be short verbs or noun phrases.",
                "Use one small support line near the destination, not under every milestone.",
                "Keep the route itself more visually prominent than the copy.",
            ],
            dynamic_devices=["move_along_path", "camera_pan", "traced_path", "lagged_ribbons"],
            suggested_features=["MoveAlongPath", "TracedPath", "MovingCameraScene", "LaggedStart", "Succession"],
            anti_patterns=["milestones as identical stacked cards", "full-sentence captions at every step"],
            elements=[
                _element("route", role="chart", kind="curved_route", placement="full_width_mid", motion="draw_and_travel", copy_source="steps"),
                _element("milestone_ribbons", role="label", kind="ribbon_labels", placement="offset_along_route", motion="lagged_reveal", copy_source="steps"),
                _element("marker", role="hero", kind="glow_marker", placement="route_origin", motion="path_travel"),
                _element("destination_note", role="support", kind="compact_callout", placement="destination", motion="fade_in", copy_source="deck"),
            ],
            motion_beats=[
                _beat("seed", purpose="Introduce the journey", focus="route", actions=["fade_in_title", "draw_route"]),
                _beat("travel", purpose="Carry the viewer through the process", focus="marker", actions=["move_along_path", "lagged_labels", "camera_pan"]),
                _beat("land", purpose="Resolve on the destination", focus="destination_note", actions=["trace_hold", "destination_emphasis"]),
            ],
        ),
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::milestone_focus_shift",
            archetype="milestone_focus_shift",
            rationale="Uses focus beams and punch-ins so the sequence reads like staged emphasis rather than a uniform checklist.",
            layout_thesis="Milestones occupy different depths and scale levels, with one focus beam promoting the currently active stage.",
            focal_system="focus-beam milestone promotion",
            background_system="subtle grid and drifting dots",
            camera_plan="Reframe from one milestone cluster to the next with short punch-ins.",
            title_strategy="Small title lockup fades into the scene after the first beat.",
            copy_strategy=[
                "Stage labels only, with one support phrase on the active beat.",
                "Use scale and position to signal hierarchy more than text quantity.",
                "Let the active milestone carry the brightest accent and largest readable words.",
            ],
            dynamic_devices=["camera_punch_ins", "focus_beam", "lagged_fade", "path_marker"],
            suggested_features=["MovingCameraScene", "LaggedStart", "FadeTransform", "MoveAlongPath", "SurroundingRectangle"],
            anti_patterns=["flat step list", "same-size cards from left to right"],
            elements=[
                _element("milestone_clusters", role="hero", kind="stage_clusters", placement="depth_staggered", motion="promote_active", copy_source="steps"),
                _element("focus_beam", role="background", kind="focus_beam", placement="active_stage", motion="slide"),
                _element("path_marker", role="diagram", kind="path_marker", placement="between_clusters", motion="travel"),
            ],
            motion_beats=[
                _beat("cue", purpose="Present the first stage", focus="milestone_clusters", actions=["fade_in_title", "promote_first"]),
                _beat("step", purpose="Walk across the middle stages", focus="path_marker", actions=["travel", "beam_slide", "camera_punch_in"]),
                _beat("resolve", purpose="Land on the final stage", focus="focus_beam", actions=["final_promote", "settle_frame"]),
            ],
        ),
    ]


def _system_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::orbit_hub_signal",
            archetype="orbit_hub_signal",
            rationale="Builds the process around a clear hub and orbital signal flow so the frame reads like a real system.",
            layout_thesis="A central hub anchors the system while source and destination nodes sit at asymmetric offsets with visible route arcs.",
            focal_system="hub-and-spoke pulse network",
            background_system="orbit ring plus faint signal trails",
            camera_plan="Start wide enough to understand the topology, then drift toward the hub as the pulse arrives.",
            title_strategy="Top-left title establishes the context before yielding to the hub system.",
            copy_strategy=[
                "Node labels should be compact nouns or verbs.",
                "Use one optional support line near the hub if needed.",
                "Avoid explanatory paragraphs inside the network.",
            ],
            dynamic_devices=["orbit_ring", "traced_path", "camera_drift", "curved_routes", "traveling_pulse"],
            suggested_features=["MovingCameraScene", "CurvedArrow", "TracedPath", "always_redraw", "LaggedStart", "MoveAlongPath"],
            anti_patterns=["workflow as disconnected floating cards", "equally weighted nodes with no hub"],
            elements=[
                _element("hub_node", role="hero", kind="signal_hub", placement="center", motion="pulse_anchor", copy_source="headline"),
                _element("source_node", role="diagram", kind="signal_node", placement="left_lower", motion="fade_in", copy_source="left_detail"),
                _element("destination_node", role="diagram", kind="signal_node", placement="right_upper", motion="fade_in", copy_source="right_detail"),
                _element("route_arcs", role="connector", kind="curved_routes", placement="between_nodes", motion="draw"),
                _element("orbit_ring", role="background", kind="orbit_ring", placement="around_hub", motion="slow_rotate"),
                _element("pulse", role="hero", kind="traveling_glow", placement="source_to_destination", motion="path_travel"),
            ],
            motion_beats=[
                _beat("topology", purpose="Reveal the network structure", focus="hub_node", actions=["fade_in_title", "bring_in_nodes", "draw_routes"]),
                _beat("flow", purpose="Animate the signal moving through the system", focus="pulse", actions=["travel_pulse", "trace_signal", "camera_drift"]),
                _beat("resolve", purpose="Land the payoff near the hub or destination", focus="destination_node", actions=["accent_destination", "settle_camera"]),
            ],
        ),
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::signal_river_map",
            archetype="signal_river_map",
            rationale="Uses one sinuous route as the organizing river of the process, which is cleaner and more cinematic than node rows.",
            layout_thesis="The flow path carries the eye diagonally while modules dock to it at intentional checkpoints.",
            focal_system="route-led system river",
            background_system="angled beams and isolated glow accents",
            camera_plan="Follow the route diagonally and settle on the final docked module.",
            title_strategy="Editorial ribbon near the route source.",
            copy_strategy=[
                "Checkpoint labels are short docked ribbons.",
                "Only the final checkpoint gets a supporting phrase.",
                "Copy should feel attached to the river, not detached cards.",
            ],
            dynamic_devices=["route_path", "move_along_path", "focus_beam", "camera_follow"],
            suggested_features=["MoveAlongPath", "MovingCameraScene", "LaggedStart", "TracedPath", "always_redraw"],
            anti_patterns=["four same-size modules", "top headline plus bottom explanation panel"],
            elements=[
                _element("signal_route", role="chart", kind="route_path", placement="diagonal", motion="draw_and_follow", copy_source="supporting_lines"),
                _element("docked_modules", role="diagram", kind="docked_labels", placement="along_route", motion="lagged_reveal", copy_source="supporting_lines"),
                _element("travel_pulse", role="hero", kind="glow_marker", placement="route_origin", motion="path_travel"),
                _element("final_callout", role="support", kind="compact_callout", placement="route_destination", motion="fade_in", copy_source="deck"),
            ],
            motion_beats=[
                _beat("seed", purpose="Reveal the route and title", focus="signal_route", actions=["fade_in_title", "draw_route"]),
                _beat("dock", purpose="Stage the intermediate modules", focus="docked_modules", actions=["lagged_modules", "travel_pulse", "camera_follow"]),
                _beat("arrive", purpose="Resolve at the destination", focus="final_callout", actions=["final_callout", "trace_hold"]),
            ],
        ),
    ]


def _comparison_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::spotlight_bridge_morph",
            archetype="spotlight_bridge_morph",
            rationale="Stages the contrast through a bridge and spotlight, making the difference itself the animation.",
            layout_thesis="Before and after live on offset planes connected by one bridge ribbon, with the winning state arriving cleaner and brighter.",
            focal_system="matched-shape bridge morph",
            background_system="single spotlight beam with soft orbit residue",
            camera_plan="Slide across the bridge and settle on the resolved state with a slight zoom-in.",
            title_strategy="Title occupies a narrow top-left band and stays out of the morph plane.",
            copy_strategy=[
                "Use one short label for each state and one concise supporting phrase.",
                "Let the final state contain the clearest copy and brightest accent.",
                "Avoid explanatory sentences on both sides at once.",
            ],
            dynamic_devices=["transform_matching", "camera_slide", "focus_beam", "ribbon_bridge"],
            suggested_features=["TransformMatchingShapes", "ReplacementTransform", "MovingCameraScene", "LaggedStart", "FadeTransform"],
            anti_patterns=["literal two-column card comparison", "equal visual weight on both states"],
            elements=[
                _element("before_state", role="hero", kind="state_cluster", placement="left", motion="morph_out", copy_source="left_detail"),
                _element("after_state", role="hero", kind="state_cluster", placement="right", motion="morph_in", copy_source="right_detail"),
                _element("bridge", role="diagram", kind="ribbon_bridge", placement="between_states", motion="draw"),
                _element("spotlight", role="background", kind="focus_beam", placement="resolved_state", motion="shift"),
            ],
            motion_beats=[
                _beat("present", purpose="Show the starting state", focus="before_state", actions=["fade_in_title", "fade_in_before"]),
                _beat("cross", purpose="Animate the transition itself", focus="bridge", actions=["draw_bridge", "shape_morph", "camera_slide"]),
                _beat("resolve", purpose="Land the viewer on the winning state", focus="after_state", actions=["focus_shift", "accent_after"]),
            ],
        ),
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::focus_lane_transition",
            archetype="focus_lane_transition",
            rationale="Creates a cleaner contrast by using a single moving lane of attention rather than two static halves.",
            layout_thesis="The frame contains one active attention lane that migrates from the old state to the new one while background residues remain dim.",
            focal_system="single active focus lane",
            background_system="layered dim residues of the inactive state",
            camera_plan="Punch into the active lane and let the inactive residue fall back.",
            title_strategy="Small ribbon title floating over the inactive margin.",
            copy_strategy=[
                "Keep the active lane readable in under a second.",
                "Use copy fragments that can morph between states.",
                "Reserve one supporting sentence fragment for the final lane only.",
            ],
            dynamic_devices=["fade_transform", "camera_punch_in", "focus_lane", "shape_morph"],
            suggested_features=["FadeTransform", "TransformMatchingShapes", "MovingCameraScene", "LaggedStart"],
            anti_patterns=["split-screen paragraphs", "two static modules with a versus label"],
            elements=[
                _element("lane_before", role="hero", kind="focus_lane", placement="left_center", motion="fade_transform", copy_source="left_detail"),
                _element("lane_after", role="hero", kind="focus_lane", placement="right_center", motion="resolve_in", copy_source="right_detail"),
                _element("inactive_residue", role="background", kind="ghost_shapes", placement="rear", motion="dim_shift"),
            ],
            motion_beats=[
                _beat("cue", purpose="Establish the first lane", focus="lane_before", actions=["fade_in_title", "focus_before"]),
                _beat("migrate", purpose="Move attention to the new lane", focus="lane_after", actions=["lane_shift", "camera_punch_in", "shape_morph"]),
                _beat("lock", purpose="Hold on the resolved lane", focus="lane_after", actions=["resolve_copy", "settle_camera"]),
            ],
        ),
    ]


def _quote_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::ribbon_phrase_sweep",
            archetype="ribbon_phrase_sweep",
            rationale="Makes the phrase feel authored through moving emphasis, ribbons, and a guided sweep instead of a static title card.",
            layout_thesis="One hero phrase spans the frame while accents and focus beams travel across the most important words.",
            focal_system="kinetic phrase sweep",
            background_system="subtle beam and orbit wash",
            camera_plan="Micro-reframe toward the emphasized phrase segment as the sweep travels.",
            title_strategy="Short eyebrow ribbon, then let the hero phrase dominate.",
            copy_strategy=[
                "Break the line into one hero phrase and one optional support phrase.",
                "Animate emphasis word by word or chunk by chunk instead of showing all lines equally.",
                "Avoid framing the phrase inside a large card.",
            ],
            dynamic_devices=["transform_matching", "focus_beam", "camera_micro_reframe", "ribbon_motion"],
            suggested_features=["TransformMatchingShapes", "FadeTransform", "MovingCameraScene", "LaggedStart", "Underline"],
            anti_patterns=["quote on a centered panel", "full sentence paragraphs in multiple boxes"],
            elements=[
                _element("eyebrow_ribbon", role="title", kind="ribbon_label", placement="upper_left", motion="fade_in", copy_source="headline"),
                _element("hero_phrase", role="quote", kind="kinetic_phrase", placement="center_left", motion="write_and_morph", copy_source="headline"),
                _element("emphasis_sweep", role="background", kind="focus_beam", placement="under_phrase", motion="sweep"),
                _element("support_phrase", role="support", kind="support_line", placement="lower_right", motion="fade_in", copy_source="deck"),
            ],
            motion_beats=[
                _beat("declare", purpose="Introduce the phrase", focus="hero_phrase", actions=["fade_in_ribbon", "write_phrase"]),
                _beat("sweep", purpose="Move emphasis through the line", focus="emphasis_sweep", actions=["beam_sweep", "phrase_morph", "camera_micro_reframe"]),
                _beat("echo", purpose="Land the support phrase cleanly", focus="support_phrase", actions=["support_fade_in", "underline_hold"]),
            ],
        ),
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::word_ladder_morph",
            archetype="word_ladder_morph",
            rationale="Lets key words climb and transform across the frame, which gives the beat more motion grammar than a paragraph card ever could.",
            layout_thesis="A vertical or diagonal ladder of short word clusters climbs through the shot, with the final cluster owning the landing position.",
            focal_system="word-cluster ladder",
            background_system="soft constellation and thin guide line",
            camera_plan="Ascend along the ladder with a small upward drift.",
            title_strategy="Tiny eyebrow, then immediate handoff to the ladder.",
            copy_strategy=[
                "Use compressed word clusters only.",
                "Let each rung of the ladder be 1-3 words.",
                "Keep support copy as a tiny footer note or omit it.",
            ],
            dynamic_devices=["phrase_morph", "camera_drift", "guide_line", "lagged_word_reveal"],
            suggested_features=["TransformMatchingShapes", "LaggedStart", "MovingCameraScene", "FadeTransform"],
            anti_patterns=["long prose lines", "big centered quote block"],
            elements=[
                _element("word_ladder", role="quote", kind="word_clusters", placement="diagonal", motion="climb_and_morph", copy_source="keywords"),
                _element("guide_line", role="diagram", kind="thin_route", placement="through_ladder", motion="draw"),
                _element("landing_cluster", role="hero", kind="hero_phrase", placement="upper_right", motion="resolve", copy_source="headline"),
            ],
            motion_beats=[
                _beat("seed", purpose="Place the first cluster", focus="word_ladder", actions=["fade_in_first", "draw_guide"]),
                _beat("climb", purpose="Progress the phrase upward", focus="word_ladder", actions=["lagged_word_reveal", "morph_clusters", "camera_drift"]),
                _beat("land", purpose="Resolve to the final phrase", focus="landing_cluster", actions=["hero_lock", "fade_residue"]),
            ],
        ),
    ]


def _stack_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::keyword_route_nodes",
            archetype="keyword_route_nodes",
            rationale="Organizes short keywords around a motion spine so the scene feels like a guided idea map instead of boxes with text.",
            layout_thesis="Keywords attach to a flowing route as nodes, with one active marker carrying attention between them.",
            focal_system="keyword route map",
            background_system="constellation dots and a faint beam",
            camera_plan="Hold the full map first, then drift toward the strongest keyword cluster.",
            title_strategy="Small title stack near the route origin.",
            copy_strategy=[
                "Each keyword should stay very short.",
                "Use tiny annotations or pills only where necessary.",
                "Do not box every keyword inside the same container.",
            ],
            dynamic_devices=["move_along_path", "lagged_nodes", "camera_drift", "traced_path"],
            suggested_features=["MoveAlongPath", "TracedPath", "LaggedStart", "MovingCameraScene", "always_redraw"],
            anti_patterns=["stacked label cards", "repeating equal-size panels"],
            elements=[
                _element("keyword_route", role="diagram", kind="route_path", placement="sweeping_midframe", motion="draw", copy_source="keywords"),
                _element("keyword_nodes", role="hero", kind="node_labels", placement="along_route", motion="lagged_reveal", copy_source="keywords"),
                _element("marker", role="metric", kind="travel_marker", placement="route_origin", motion="path_travel"),
            ],
            motion_beats=[
                _beat("map", purpose="Introduce the concept map", focus="keyword_route", actions=["fade_in_title", "draw_route"]),
                _beat("activate", purpose="Walk the active marker through the idea", focus="marker", actions=["travel", "reveal_nodes", "camera_drift"]),
                _beat("hold", purpose="Land on the strongest keyword", focus="keyword_nodes", actions=["hero_emphasis", "settle_frame"]),
            ],
        )
    ]


def _interface_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    return [
        _blueprint(
            brief,
            blueprint_id=f"{brief.visual_id}::module_cascade_zoom",
            archetype="module_cascade_zoom",
            rationale="Makes interface scenes feel premium by using module depth, focus rings, and camera choreography instead of fake dashboard tiles.",
            layout_thesis="Modules cascade in depth with one hero module promoted by a focus ring and camera punch-in.",
            focal_system="hero module promotion",
            background_system="subtle rails and a scan beam",
            camera_plan="Start on the full module stack and punch into the promoted module after the cascade arrives.",
            title_strategy="Top-left eyebrow plus a short deck, then get out of the way.",
            copy_strategy=[
                "UI labels must be compact and module-specific.",
                "Only the promoted module gets full readable copy.",
                "Other modules should read as context, not equal-weight cards.",
            ],
            dynamic_devices=["camera_punch_in", "focus_ring", "lagged_modules", "scan_beam"],
            suggested_features=["MovingCameraScene", "SurroundingRectangle", "LaggedStart", "FadeTransform", "always_redraw"],
            anti_patterns=["flat dashboard screenshot", "same-size panels with the same emphasis"],
            elements=[
                _element("module_stack", role="panel", kind="ui_modules", placement="depth_cascade", motion="lagged_reveal", copy_source="supporting_lines"),
                _element("focus_ring", role="hero", kind="focus_ring", placement="hero_module", motion="track_and_hold"),
                _element("scan_beam", role="background", kind="focus_beam", placement="across_stack", motion="sweep"),
            ],
            motion_beats=[
                _beat("cascade", purpose="Bring the modules into the frame", focus="module_stack", actions=["fade_in_title", "lagged_modules"]),
                _beat("promote", purpose="Promote the hero module", focus="focus_ring", actions=["focus_track", "camera_punch_in", "scan_sweep"]),
                _beat("land", purpose="Hold on the important module", focus="module_stack", actions=["settle_frame", "accent_hero"]),
            ],
        )
    ]


def _candidate_blueprints(brief: SceneBrief) -> list[SceneBlueprint]:
    if brief.scene_family == "metric_story":
        return _metric_blueprints(brief)
    if brief.scene_family == "dashboard_build":
        return _dashboard_blueprints(brief)
    if brief.scene_family == "timeline_journey":
        return _timeline_blueprints(brief)
    if brief.scene_family == "system_map":
        return _system_blueprints(brief)
    if brief.scene_family == "comparison_morph":
        return _comparison_blueprints(brief)
    if brief.scene_family == "interface_focus":
        return _interface_blueprints(brief)
    if brief.scene_family == "kinetic_stack":
        return _stack_blueprints(brief)
    return _quote_blueprints(brief)


def _score_blueprint(brief: SceneBrief, blueprint: SceneBlueprint) -> float:
    score = 0.0
    score += 5.0 if blueprint.scene_family == brief.scene_family else 1.5
    preferred = set(brief.preferred_manim_features)
    score += len(preferred.intersection(blueprint.suggested_features)) * 0.38
    score += len(blueprint.dynamic_devices) * 0.14
    score += len(blueprint.motion_beats) * 0.18
    score += len(blueprint.elements) * 0.06
    panel_like = sum(1 for element in blueprint.elements if element.kind in {"panel", "card", "ui_modules", "strip_modules"})
    score -= max(0, panel_like - (1 if brief.scene_family == "interface_focus" else 0)) * 0.55
    if brief.camera_style == "guided" and ("camera" in blueprint.camera_plan.lower() or "track" in blueprint.camera_plan.lower()):
        score += 0.45
    if brief.animation_intensity == "high":
        score += 0.35 if len(blueprint.dynamic_devices) >= 4 else 0.0
    elif brief.animation_intensity == "low":
        score -= max(0, len(blueprint.dynamic_devices) - 3) * 0.1
    if brief.composition_mode == "replace":
        score += 0.25
    if brief.scene_family in {"system_map", "timeline_journey"} and any(
        "path" in element.kind or "route" in element.kind for element in blueprint.elements
    ):
        score += 0.42
    if brief.scene_family == "comparison_morph" and "transform_matching" in blueprint.dynamic_devices:
        score += 0.38
    if brief.scene_family in {"kinetic_quote", "kinetic_stack"} and "focus_beam" in blueprint.dynamic_devices:
        score += 0.22
    return round(score, 3)


def build_scene_blueprints(brief: SceneBrief, *, limit: int = 3) -> list[SceneBlueprint]:
    blueprints = _candidate_blueprints(brief)
    for blueprint in blueprints:
        blueprint.score = _score_blueprint(brief, blueprint)
    ranked = sorted(
        blueprints,
        key=lambda item: (item.score, len(item.dynamic_devices), len(item.motion_beats)),
        reverse=True,
    )
    return ranked[:limit]
