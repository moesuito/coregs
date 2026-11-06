# Game profiles

CoreGS renders by the GS hardware rules. Some games, however, use those rules in a way that a
renderer can recognise and optimise for: a full-screen pass that always reads one table, a
channel-shuffle run that always ends in a palette lookup, a repeated readback of the same buffer.
Those optimisations are only correct for the game that produces them, so they are **not** in the
generic core. They live in a **game profile**: a set of rules that a host installs explicitly.

The generic profile (`coregs::GameProfile{}`) has every rule off. With it, CoreGS renders any GS
stream by the hardware rules alone.

## The rules

`include/coregs/profile.h` defines the whole mechanism. A profile is data, not a subclass: a name,
a set of flags and (for two rules) a predicate the renderer calls with a description of what it is
about to do.

| Rule | Default | What it assumes | Risk if the game does not match |
| --- | --- | --- | --- |
| `hide_light_flare_table` + `is_light_flare_table_read` | off | The read the predicate names is a visibility table whose entries the game only uses to scale an effect, so zeros are a valid answer. | A read the game needs is answered with zeros: missing effects or wrong logic. |
| `skip_effect_pass` + `is_effect_pass` | off | The pass the predicate names is a screen-space effect the player can trade for speed. | A visible effect disappears (typically darker or missing light). |
| `effect_pass_early_out` | off | For the named pass, a fragment that cannot contribute can leave before it samples or stores, without changing the picture. | A wrong early-out changes edge pixels. |
| `skip_alpha_only_fills` + `alpha_only_fill_min_width` | off | Fills that write only the alpha lane exist for an effect the profile also drops. | Wrong alpha on the next blended pass. |
| `tone_runs` | off | A channel-shuffle run into one target implements a tone curve, and one palette pass over the target is equivalent to the run. | A different use of channel shuffles renders incorrectly. |
| `palette_segments` | off | A T8 sprite batch that changes palette mid-batch can be kept open and drawn in segments, one palette row each. | Palette changes inside a batch are drawn with the wrong colours. |
| `effect_targets_at_1x` | off | Small off-screen effect targets may stay at 1x while the picture is scaled. | Blurrier reflections or shadows (never wrong geometry). |
| `known_shader_variants` | empty | The listed draw states are the ones the game will ask for; precompiling them avoids first-run stalls. | Only a stall; never a wrong picture. |

The two predicates receive plain structs, not renderer internals:

```cpp
struct GameProfile::ReadbackView {
  std::uint32_t psm;      // GS pixel format of the read
  std::uint32_t width;    // pixels
  std::uint32_t height;
  std::uint32_t qwords;   // bytes / 16
};

struct GameProfile::EffectPassView {
  std::uint32_t frame_width, frame_height;  // native target size
  std::uint64_t covered_area;               // native pixels the pass covers
  bool textured, blends;
  std::uint32_t blend_a, blend_b, blend_c, blend_d;
  bool depth_never, depth_always;
  std::uint32_t format, fbmsk;
};
```

A predicate must be a pure function of that view (and of the profile's own settings). It is called
on the GS thread, once per candidate pass or read.

## Installing a profile

Install it before the first frame, or between frames with the renderer idle:

```cpp
coregs::gt4::Settings settings;             // all the GT4 rules' own switches
coregs::GameProfile profile = coregs::gt4::MakeProfile(settings);
renderer.set_profile(profile);
```

`set_profile` copies the profile; the caller may free its strings and arrays afterwards. Set it
before `init()` if you want the shader precompilation list to be used; the per-draw rules can be
changed later.

## The Gran Turismo 4 profile

`profiles/gt4/` is the reference example, MIT and part of this repository. GT4 is the game CoreGS
was developed and validated against, so its profile is also the documented answer to "what does a
real profile look like?".

With the default `coregs::gt4::Settings` it reproduces the faithful GT4 renderer:

| Setting | Default | Effect |
| --- | --- | --- |
| `hide_light_flare_table` | `false` | When on, GT4's 64×8 CT32 light-flare visibility read is answered with zeros: the readback wait and the flare/glare draws disappear. |
| `skip_road_sheen` | `false` | When on, the full-screen road sheen pass is dropped: the road is darker. |
| `road_sheen_early_out` | `true` | Exact early-out for that pass (a fragment with no destination alpha contributes nothing). |
| `tone_runs` | `true` | GT4's brightness/fade ramp is a run of channel-shuffle draws; one palette pass replaces it. |
| `palette_segments` | `false` | Experimental; a measured GT4 scene did not gain, so it stays off. |
| `effect_targets_at_1x` | `false` | Small reflection/shadow targets stay at 1x when the game is scaled. |
| `precompile_known_variants` | `true` | The draw states GT4 asks for, in `known_shader_variants.inc`, are compiled into the driver cache before the first race. |

The two predicates are in `gt4_profile.cpp` and are short enough to read as the specification of
the rule:

```cpp
bool FlareTable(const GameProfile::ReadbackView &read) {
  return read.psm == 0u && read.width == 64u && read.height == 8u && read.qwords == 128u;
}

bool RoadSheen(const GameProfile::EffectPassView &pass) {
  if (!pass.textured || !pass.blends) return false;
  if (pass.blend_a != 0u || pass.blend_b != 2u || pass.blend_c != 1u || pass.blend_d != 1u) return false;
  if (!pass.depth_never && !pass.depth_always) return false;
  if (pass.frame_width < 640u) return false;
  const std::uint64_t frame_area = static_cast<std::uint64_t>(pass.frame_width) * pass.frame_height;
  return pass.covered_area * 5u >= frame_area * 2u;
}
```

`profiles/gt4/patterns.h` adds a second, independent example: a classifier for GT4's tone patterns
that a host can use to choose settings for a scene. The renderer never calls it. It shows how a
profile can keep game knowledge on its own side of the interface.

### What the GT4 profile does *not* do

It does not change blending, depth, formats, the display path or readback semantics. Those are
hardware rules. If a GT4 scene renders wrong, the fix belongs in the core (with a fixture), not in
the profile.

## Adding a profile for another game

1. Create `profiles/<game>/` with a header and a `.cpp` that build a `coregs::GameProfile`.
   Start with every rule off.
2. Reproduce the bug or the performance problem with the generic profile. Record the scene, the
   capture and the measurement in the pull request.
3. Turn on **one** rule. Prove that it is equivalent (same picture and same GS memory in the
   fixture, or the documented effect trade) or that it is what the game actually needs.
4. Add a fixture under `profiles/<game>/` that runs a synthetic or recorded stream through the
   generic profile and through yours. The generic run must stay correct; the profile run must show
   the intended change.
5. Document every rule in the profile's header: what the game does, why the rule is safe, and what
   the player loses when the rule is on.

A profile is a compatibility statement. If you cannot explain in one sentence why the rule is
correct for that game, it is not ready.

## When not to add a profile

If the game renders wrong with every rule off, the core has a bug: fix the core. A profile must
never be used to paper over a wrong format, a wrong blend or a missing readback. The generic path is
the contract; profiles are the exception, on the record.
