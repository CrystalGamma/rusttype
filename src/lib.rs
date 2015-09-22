#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {Forward, Reverse}
use self::Direction::*;

impl Direction {
	fn reverse(self) -> Direction {
		match self {
			Forward => Reverse,
			Reverse => Forward
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GlyphOrient {Perpendicular, Parallel}
use self::GlyphOrient::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirBehavior(Direction, GlyphOrient);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirPreference {Horiz(DirBehavior), Vert(DirBehavior), BiOrient(DirBehavior, DirBehavior)}	// enable scripts without preference?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rot180 {Normal, Rotated}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreakTypesetting(DirBehavior, Rot180);

impl StreakTypesetting {
	fn effective_direction(&self) -> Direction {
		match self.1 {
			Rot180::Normal => self.0 . 0,
			Rot180::Rotated => self.0 . 0 .reverse()
		}
	}
}

/// ISO 15924 alphabetical code
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScriptCode([u8; 4]);

pub static LATIN: ScriptCode = ScriptCode(['L' as u8, 'a' as u8, 't' as u8, 'n' as u8]);	// "Latn"

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MainAxis {Ltr, Rtl, Ttb, Btt}

pub struct NonNativeConf;

pub struct Run<'a>(pub &'a str, pub ScriptCode);

pub fn make_default_streak(run: &Run, main_axis: MainAxis) {
	use self::MainAxis::*;
	Streak(run.0, run.1, StreakTypesetting(DirBehavior(Forward, Perpendicular), if run.1 == LATIN {
		match main_axis {
			Ltr | Ttb => Normal,
			Rtl | Btt => Rotated
		}
	} else {
		unimplemented!();
	}))
}

pub fn make_runs(s: &str) -> Vec<Run> {vec![Run(s, LATIN)]}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Streak<'a>(&'a str, ScriptCode, StreakTypesetting);

impl<'a> Streak<'a> {
	/// what order do the parts have if you split the streak
	pub fn direction(&self) -> Direction {self.2.effective_direction()}
}

pub fn bidi_algorithm(streaks: &mut [Streak]) {}

pub struct FontPreference(Font);

impl FontPreference {
	fn lookup(&self) -> Vec<Burst> {
		Vec::new()
	}
}

pub struct Font;

struct GlyphIndex(u16);

struct Burst<'a>(Vec<GlyphIndex>, &'a Font, StreakTypesetting);

pub struct TypeSetting<'a>(Vec<Burst<'a>>);

fn typeset<'a>(streak: Streak, fonts: &'a FontPreference) -> TypeSetting<'a> {
	TypeSetting(Vec::new())
}