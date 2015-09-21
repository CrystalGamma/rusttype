# Goals

* Correct typesetting for all scripts
  * Bidi text
  * Vertical text
* Highspeed rendering for TrueType outlines
  * No hinting support planned
* Simple, high-level API
  * powerful enough to implement a text editor
    * Need mouse selection support, caret positioning - as precise as possible

# Non-Goals

* Hinting
* Multi-line Layout
  * Line breaks
  * Hyphenation

# Typesetting phases
## Bidi mangling
Outputs `Streaks`: substrings with direction.
Font independent.

Configuration:
* Handling of text in non-native directions
  * translate-forward, translate-reverse, rotate-cw, rotate-ccw, rotate-forward, rotate-reverse, *-weak

Rationale: Separated in order to allow the application to draw its own backdrops and boxes

## Shaping & Positioning
Takes `Render Streaks`: Streaks may be broken up further if the application creates inline boxes.
Open Questions: Do any fonts do shaping/positioning across Bidi boundaries?
(I assume there are no scripts with both RTL and LTR characters)


### Lookup
Per `Render Streak`
Look up characters in fonts.
Produces `Bursts` (strings of glyph indices belonging to a single font),
usually 1 per Streak, but more if characters were only found in different fonts.

Challenges:
* What to do if some characters could not be found in preferred font,
  but the fallback also contains some characters between the ones not found in preferred font?
  Take those from the fallback too? How many?


Configuration:
* Font preference

### Shaping
Per `Burst`:
Transforms the glyph index strings according to font rules.
Produces new `Burst`.

Configuration:
* Font feature options
  * alternative Glyphs
  * writing system
* Upright rotated text: Shape everything as isolated

Challenges:
* Input Character -> Glyph mapping (necessary for selection/caret)
  * Maybe use metadata from Graphite?
  * need heuristics for other cases

### Positioning
Per `Burst`:
Calculate position for the glyphs in the `Burst`.
Calculate the bounding box.

Challenges: Upright rotated text. Do we need 

