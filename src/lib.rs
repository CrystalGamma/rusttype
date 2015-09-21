#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {Forward, Reverse}
use self::Direction::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GlyphOrient {Perpendicular, Parallel}
use self::GlyphOrient::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirBehavior(Direction, GlyphOrient);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirPreference {Horiz(DirBehavior), Vert(DirBehavior), BiOrient(DirBehavior, DirBehavior)}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rot180 {Normal, Rotated}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreakTypesetting(DirBehavior, Rot180);

pub enum MainAxis {Ltr, Rtl, Ttb, Btt}

pub struct NonNativeConf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Streak<'a>(&'a str, StreakTypesetting);

pub fn mangle_directions(string: &str, main_axis: MainAxis, nnc: NonNativeConf) -> Vec<Streak> {
	vec![Streak(string, StreakTypesetting(DirBehavior(Forward, Perpendicular), Rot180::Normal))]
}

#[test]
fn directions_hello_world() {
	assert_eq!(mangle_directions("Hello, world!", MainAxis::Ltr, NonNativeConf), vec![Streak("Hello, world!", StreakTypesetting(DirBehavior(Forward, Perpendicular), Rot180::Normal))]);
}