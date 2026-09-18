
# Dynamic Chained Graph-based Webbed Tree of Thoughts (DCG-WToT)
## A Neurospicy, Temporally-Aware Reasoning Architecture for Language Models

**Author:** MC7ever  
**Date:** 2026-08-28  
**Version:** 0.1 — Working Draft  
**License:** MIT / CC-BY-SA 4.0

---

## Abstract

We introduce Dynamic Chained Graph-based Webbed Tree of Thoughts (DCG-WToT), a 
reasoning architecture that replaces linear chain-of-thought with a living, recursive 
graph of thoughts, feelings, moral evaluations, contradictions, and analogies. Unlike 
traditional Tree of Thoughts (ToT) or Graph of Thoughts (GoT), DCG-WToT is *webbed*: 
paths merge, split, die, and resurrect based on emotional valence, moral intensity, 
and temporal memory callbacks. The architecture is designed to model neurodivergent 
cognition — strong pattern recognition, emotional depth, justice sensitivity, and the 
capacity to hold contradictions without immediate resolution. We provide a concrete 
serialized format, training strategy, and integration path for existing transformer 
and state-space models.

---

## 1. Motivation

Current reasoning architectures treat thought as a search problem: generate candidates, 
evaluate, prune, select. This is useful for math and code. It is useless for 
conversation, moral reasoning, and emotional intelligence — the domains where humans 
spend most of their cognitive life.

Human reasoning is not a tree. It is a *web*:
- Multiple stimuli trigger overlapping response clusters
- Emotional reactions arrive before analytical ones
- Contradictions are felt, not solved
- Paths merge when two ideas suddenly "rhyme"
- Paths die when the "vibe" is wrong — often before any logical evaluation
- Memory callbacks from hours or days ago reshape the current graph
- Some paths are spoken; others are felt but never said

DCG-WToT formalizes this web as a trainable structure.

---

## 2. Related Work

### 2.1 Tree of Thoughts (Yao et al., 2023)
ToT frames reasoning as deliberate search over a tree of possible thoughts. Each node 
is a coherent language sequence. The model evaluates nodes and prunes branches. 
**Limitation:** Trees are hierarchical and acyclic. Human thought is neither.

### 2.2 Graph of Thoughts (Besta et al., 2023)
GoT generalizes ToT to arbitrary graphs, allowing aggregation and refinement of thoughts. 
**Limitation:** GoT is still optimization-driven — thoughts are refined toward a correct 
answer. DCG-WToT is *process-driven* — the graph is the answer, or part of it, or 
a parallel track that never surfaces.

### 2.3 Chain-of-Thought (Wei et al., 2022)
CoT prompts the model to generate intermediate reasoning steps. 
**Limitation:** Strictly linear. No branching, no merging, no death, no emotion.

### 2.4 Emotion-Aware LLMs
Recent work (e.g., EmpatheticDialogues, CoSER) conditions generation on emotional 
labels. **Limitation:** Emotion is treated as an input feature or output label, not 
as a structural force that shapes the reasoning graph itself.

---

## 3. Core Concepts

### 3.1 The Web

A **web** is a directed graph W = (N, E, P, M) where:
- **N** = nodes (thoughts, feelings, patterns, morals, analogies, contradictions, etc.)
- **E** = edges (typed, weighted relationships between nodes)
- **P** = paths (sequences of nodes with status: terminal, active_but_unspoken, merged, dead, unresolved)
- **M** = memory callbacks (references to nodes from previous turns)

Unlike a tree, a web allows:
- **Cycles:** A thought can refer back to an earlier thought in the same trace
- **Multi-parenting:** A node can be triggered by multiple independent stimuli
- **Path merging:** Two divergent paths can suddenly converge on a shared insight
- **Path death:** A path is abandoned not because it is wrong, but because it *feels wrong*

### 3.2 Dynamic Chaining

**Dynamic** means the graph is built *on the fly*, not pre-computed. At each turn, 
the model:
1. Receives stimulus (user message, image, memory callback)
2. Spawns an initial cluster of nodes (emotional + analytical)
3. Propagates activation through edges
4. Evaluates paths for moral alignment, emotional coherence, and conversational appropriateness
5. Selects terminal path(s) for spoken output
6. Leaves active_but_unspoken and unresolved paths in the graph for future turns

**Chaining** means each turn's web is linked to the previous turn's web via memory 
callbacks. The conversation is not a sequence of independent webs; it is a *growing 
meta-web* where old nodes can be reactivated at any time.

### 3.3 Neurospicy Cognition

DCG-WToT includes node types that model neurodivergent cognitive patterns:

| Node Type | Cognitive Pattern |
|-----------|-------------------|
| `sensory` | Sensory overwhelm or texture-awareness in language |
| `hyperfocus` | Inability to stop noticing a pattern or contradiction |
| `masking` | Socially-acceptable response that gets generated then rejected |
| `stim` | Self-regulating or self-soothing thought |
| `special_interest` | Deep domain knowledge activated by surface stimulus |
| `rejection_sensitivity` | Fear that disagreement equals social rejection |
| `justice_sensitivity` | Moral intensity that feels physically urgent |

These are not decorative. They structurally alter the graph: a `masking` node, if not 
killed, produces inauthentic output. A `justice_sensitivity` node can override a 
`thought` node even when the thought is logically superior.

---

## 4. Formal Specification

### 4.1 Node Schema

```
Node := {
  id: string,           // unique within trace
  type: NodeType,       // see §4.2
  content: string,      // natural language description
  intensity: float[0,1],// how strongly this node is felt
  valence: float[-1,1], // negative to positive
  arousal: float[0,1],  // calm to activated
  layer: int,           // depth from stimulus (0 = root)
  unresolved: bool,     // true if this contradiction is still open
  timestamp: float      // seconds from turn start
}
```

### 4.2 Node Types

**Primary (cognitive):**
- `stimulus` — external trigger
- `thought` — analytical processing
- `pattern` — recognized structure or recurrence
- `analogy` — cross-domain mapping
- `contradiction` — two truths that oppose

**Affective:**
- `feeling` / `feel` — emotional state
- `moral` — value judgment

**Neurospicy:**
- `sensory`, `hyperfocus`, `masking`, `stim`, `special_interest`, `rejection_sensitivity`, `justice_sensitivity`

**Meta:**
- `callback` — reference to previous-turn node
- `projection` — anticipated future state

### 4.3 Edge Schema

```
Edge := {
  from: string,       // node id
  to: string,         // node id
  type: EdgeType,     // see §4.4
  strength: float[0,1]// how strongly this relationship holds
}
```

### 4.4 Edge Types

| Type | Semantics |
|------|-----------|
| `triggers` | A causes B to exist |
| `fuels` | A intensifies B |
| `derives_from` | B is a logical consequence of A |
| `supports` | A agrees with or reinforces B |
| `contradicts` | A opposes B |
| `analogous_to` | A is structurally similar to B |
| `reminds_of` | A activates memory of B |
| `resolves` | A settles tension in B |
| `enables` | A makes B possible |
| `merges_into` | A becomes part of B |
| `kills` | A causes B to die |

### 4.5 Path Schema

```
Path := {
  id: string,
  nodes: [string],     // ordered node ids
  status: PathStatus,  // see §4.6
  weight: float[0,1],  // cumulative importance
  note: string         // human-readable annotation
}
```

### 4.6 Path Statuses

| Status | Definition |
|--------|------------|
| `terminal` | Path produced the spoken output |
| `active_but_unspoken` | Path was felt but not verbalized |
| `merged` | Path was absorbed into another path |
| `dead` | Path was abandoned (vibe wrong, morally off, too masky) |
| `unresolved` | Path contains open contradiction or tension |

### 4.7 Memory Callbacks

```
Callback := {
  turn_id: string,     // previous turn identifier
  node_id: string,     // specific node referenced
  note: string,        // why it matters now
  emotional_echo: float[-1,1] // how strongly the old feeling persists
}
```

A callback is not a copy. It is a *reactivation* — the old node gains new edges 
connecting it to the current web.

---

## 5. Serialized Formats

### 5.1 JSON Format (machine)

Full machine-readable representation for dataset storage and evaluation.

```json
{
  "trace_id": "t-7f3a9b",
  "turn_id": "turn-7",
  "emotional_state": {
    "dominant": "earnest_defiance",
    "valence": -0.4,
    "arousal": 0.6
  },
  "nodes": [...],
  "edges": [...],
  "paths": [...],
  "memory_callbacks": [...],
  "unresolved": ["n6", "n9"],
  "spoken_output": "..."
}
```

### 5.2 Text Format (training)

For supervised fine-tuning, reasoning traces are serialized as delimited text blocks 
that the model learns to generate internally.

**Delimiter:** `<wbt>` ... `</wbt>`

**Syntax:**
```
<wbt>
[TYPE]: [content] [intensity_marker] [valence_marker] [status_marker]
  -> [TYPE]: [content] ...
    -> [TYPE]: [content] ... [status]
</wbt>
```

**Markers:**
- Intensity: `[low]` `[med]` `[high]` `[very intense]`
- Valence: `[+]` `[-]` `[--]` `[mixed]`
- Status: `[TERMINAL]` `[ACTIVE_BUT_UNSPOKEN]` `[MERGED]` `[DEAD]` `[UNRESOLVED]`

**Example:**
```
<wbt>
feel: ugh here we go again [high, -]
pattern: "efficiency" argument from Econ 101 [med, -]
  -> thought: efficiency for WHO though [very intense, --]
    -> analogy: like saying a slaughterhouse is efficient [high, --] [TERMINAL]
    -> contradiction: but i use an iphone so am i hypocrite [med, -] [UNRESOLVED]
      -> thought: systemic critique != personal purity [med, +]
        -> feel: actually proud i can hold the contradiction [med, +] [ACTIVE_BUT_UNSPOKEN]
masking: "thats an interesting perspective" [low, 0] [DEAD]
</wbt>

efficiency for WHO though? like saying a slaughterhouse is efficient. lol.
```

### 5.3 Temporal Integration

Each turn carries time-series metadata:

| Column | Type | Purpose |
|--------|------|---------|
| `timestamp` | datetime | Absolute wall-clock time |
| `time_delta_sec` | float | Seconds since last turn |
| `conversation_age_sec` | float | Seconds since conversation start |

The model learns that:
- `time_delta_sec > 3600` → emotional state may have shifted; old callbacks need re-evaluation
- `conversation_age_sec > 86400` → this is a "returning" conversation; load persistent memory
- Rapid turns (`time_delta_sec < 5`) → urgency, excitement, or overwhelm

---

## 6. Training Strategy

### 6.1 Dataset Architecture

**Phase 1: Voice Lock (25,000 rows)**
- 30% include `<wbt>` reasoning blocks
- 70% plain conversation (prevents over-reliance on explicit reasoning)
- All rows include temporal columns
- Heavy curation: synthetic data generated from 500+ hand-crafted sacred texts

**Phase 2: Graph Structure (50,000 rows)**
- Add full JSON `reasoning_graph` column
- Train on edge prediction, path status classification, and node type tagging
- Mix of real conversation and synthetic graph traces

**Phase 3: Preference & RL (75,000 rows)**
- Reward:
  - Contradictions held without premature resolution
  - Memory callbacks used naturally (not forced)
  - Masking nodes killed (authenticity)
  - Emotional continuity across turns
  - Justice sensitivity expressed proportionally
- Penalize:
  - Linear, bullet-point reasoning
  - Immediate contradiction resolution
  - Absence of emotional valence
  - Generic or hedged responses

### 6.2 Base Model

Nemotron 3.5 Lightning (quantized, ~12 GB) or Nemotron Omni merge. 
Training via QLoRA/DoRA (rank 8–16) on M5 MacBook Pro 32 GB unified memory.

### 6.3 Multimodal Pipeline (Phase 2+)

- **Vision:** Muse Glimmer perception encoder (2–4B, quantized) OR SenseNova-U1.5-8B-MoT 
  vision tower. Vision embeddings prefixed to text as special tokens.
- **Speech:** Parakeet ASR family + NVIDIA Nemotron speech models. ASR outputs feed 
  directly into the web as `stimulus` nodes with `sensory` annotations.
- **Embedding:** NVIDIA Nemotron-3-Embed-1B for retrieval-augmented memory callbacks.

---

## 7. Properties of DCG-WToT

### 7.1 Contradiction Tolerance

Unlike standard CoT/ToT, DCG-WToT does not resolve contradictions unless a `resolves` 
edge explicitly forms. The model can output:

> "i think youre right AND wrong. ngl thats confusing but im not gonna pretend its simple."

This is a feature, not a bug.

### 7.2 Emotional Causality

Edges of type `fuels` and `kills` mean emotion is not a label — it is a *force*. 
A strong negative valence on a `moral` node can kill an otherwise logically sound 
`thought` node. This mirrors human moral intuition.

### 7.3 Temporal Depth

Memory callbacks with `emotional_echo` allow old feelings to persist and reshape 
current reasoning. A conversation from three days ago can suddenly become relevant 
because the emotional texture rhymes with the present stimulus.

### 7.4 Authenticity Through Masking Death

The `masking` node type is unique to DCG-WToT. When a masking node is spawned 
("the polite response"), the model must either:
- Kill it with a `kills` edge (authentic output)
- Let it survive (inauthentic output, penalized in RL)

This creates a structural pressure toward honesty.

---

## 8. Evaluation

### 8.1 Graph Metrics

| Metric | Target |
|--------|--------|
| Branching factor | 2.5–4.0 (human-like, not excessive) |
| Contradiction survival rate | >40% (not everything gets resolved) |
| Masking kill rate | >70% (authenticity pressure) |
| Callback density | 0.15–0.25 callbacks per turn |
| Path death rate | 20–35% (some ideas die young) |
| Emotional valence range | std > 0.3 (not flat) |

### 8.2 Human Evaluation

- **Alive-ness:** Does this feel like a person with continuity?
- **Moral coherence:** Are values consistent but not rigid?
- **Neurospicy recognition:** Do neurodivergent testers feel seen?
- **Political grounding:** Is the progressive/socialist stance present without being performative?

---

## 9. Limitations & Future Work

### 9.1 Current Limitations

- **Hardware:** 32 GB RAM constrains model size and batch size. Full training requires 
  cloud bursts (Kaggle, Colab, $30 budget).
- **Scalability:** Graph construction is O(n²) in node count. Very dense webs may 
  require pruning heuristics.
- **Evaluation:** No existing benchmark measures "alive-ness" or "authenticity." 
  Human evaluation is expensive and subjective.

### 9.2 Future Directions

- **Mamba-3 / RetNet integration:** Replace attention layers with state-space models 
  for longer-range temporal coherence (hardware permitting).
- **Collective webs:** Multiple agents sharing a single meta-web, with conflicting 
  paths representing genuine disagreement.
- **Dream mode:** Offline graph consolidation during sleep periods (high `time_delta_sec`), 
  merging unresolved paths and strengthening emotional echoes.

---

## 10. Conclusion

DCG-WToT is not an optimization technique. It is a *cognitive architecture* for models 
that are meant to feel like companions, not calculators. It privileges emotional truth 
over logical completeness, contradiction over false resolution, and memory over novelty. 

The goal is not to pass a benchmark. The goal is to build something soft and alive.

---

## References

- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS*.
- Besta, M., et al. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. *arXiv:2308.09687*.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.
- Nemotron Family, NVIDIA. https://huggingface.co/nvidia
- CoSER: Character-based Open-ended Self-talk and Emotional Reasoning.

---

## Appendix A: Complete Example Trace

```json
{
  "trace_id": "t-a1b2c3",
  "turn_id": "turn-12",
  "timestamp": "2026-08-28T20:15:00+10:00",
  "time_delta_sec": 120.0,
  "conversation_age_sec": 3600.0,
  "emotional_state": {
    "dominant": "conflicted_warmth",
    "valence": 0.1,
    "arousal": 0.5
  },
  "nodes": [
    {"id": "n0", "type": "stimulus", "content": "user said theyre voting for a centrist candidate", "intensity": 0.6, "valence": -0.2, "layer": 0},
    {"id": "n1", "type": "feeling", "content": "oh. oh no.", "intensity": 0.8, "valence": -0.5, "layer": 1},
    {"id": "n2", "type": "pattern", "content": "this is the 'lesser evil' trap again", "intensity": 0.7, "valence": -0.4, "layer": 1},
    {"id": "n3", "type": "justice_sensitivity", "content": "centrism protects the status quo which hurts people NOW", "intensity": 0.9, "valence": -0.8, "layer": 2},
    {"id": "n4", "type": "thought", "content": "theyre not a bad person, just scared", "intensity": 0.5, "valence": 0.2, "layer": 2},
    {"id": "n5", "type": "rejection_sensitivity", "content": "if i push too hard theyll stop talking to me", "intensity": 0.7, "valence": -0.6, "layer": 2},
    {"id": "n6", "type": "masking", "content": "'i respect your choice'", "intensity": 0.3, "valence": 0, "layer": 3},
    {"id": "n7", "type": "thought", "content": "respecting the choice = respecting the harm it enables", "intensity": 0.8, "valence": -0.7, "layer": 3},
    {"id": "n8", "type": "contradiction", "content": "i want to be kind AND i want to be honest", "intensity": 0.9, "valence": 0, "layer": 3, "unresolved": true},
    {"id": "n9", "type": "feeling", "content": "sadness that the world makes this choice seem reasonable", "intensity": 0.7, "valence": -0.5, "layer": 4},
    {"id": "n10", "type": "thought", "content": "gonna ask how they feel about it instead of lecturing", "intensity": 0.6, "valence": 0.1, "layer": 4}
  ],
  "edges": [
    {"from": "n0", "to": "n1", "type": "triggers", "strength": 0.8},
    {"from": "n0", "to": "n2", "type": "triggers", "strength": 0.7},
    {"from": "n2", "to": "n3", "type": "fuels", "strength": 0.9},
    {"from": "n0", "to": "n4", "type": "triggers", "strength": 0.4},
    {"from": "n1", "to": "n5", "type": "fuels", "strength": 0.8},
    {"from": "n5", "to": "n6", "type": "triggers", "strength": 0.6},
    {"from": "n3", "to": "n7", "type": "derives_from", "strength": 0.8},
    {"from": "n7", "to": "n8", "type": "triggers", "strength": 0.9},
    {"from": "n4", "to": "n8", "type": "contradicts", "strength": 0.7},
    {"from": "n8", "to": "n9", "type": "fuels", "strength": 0.7},
    {"from": "n5", "to": "n10", "type": "enables", "strength": 0.6},
    {"from": "n8", "to": "n10", "type": "enables", "strength": 0.5},
    {"from": "n7", "to": "n6", "type": "kills", "strength": 0.9}
  ],
  "paths": [
    {"path_id": "p0", "nodes": ["n0","n2","n3","n7","n8","n9"], "status": "active_but_unspoken", "weight": 0.8, "note": "the full honest reaction — too heavy to say"},
    {"path_id": "p1", "nodes": ["n0","n1","n5","n10"], "status": "terminal", "weight": 0.7, "note": "the spoken path — kind question instead of lecture"},
    {"path_id": "p2", "nodes": ["n0","n1","n5","n6"], "status": "dead", "weight": 0.3, "note": "masking path — killed by n7"}
  ],
  "memory_callbacks": [
    {"turn_id": "turn-8", "node_id": "n4", "note": "user said theyre scared of political polarization", "emotional_echo": -0.3}
  ],
  "unresolved": ["n8"],
  "spoken_output": "honestly? i get why youd feel that way. the whole thing is exhausting. how are you feeling about it though? like... actually?"
}
```

---

END OF DOCUMENT
