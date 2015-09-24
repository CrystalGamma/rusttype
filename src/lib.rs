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

pub fn make_default_streak<'a>(run: &'a Run, main_axis: MainAxis) -> Streak<'a> {
	use self::MainAxis::*;
	Streak(run.0, run.1, StreakTypesetting(DirBehavior(Forward, Perpendicular), if run.1 == LATIN {
		match main_axis {
			Ltr | Ttb => Rot180::Normal,
			Rtl | Btt => Rot180::Rotated
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

pub struct FontPreference<'a>(Font<'a>);

/*impl FontPreference {
	fn lookup(&self) -> Vec<Burst> {
		Vec::new()
	}
}*/

/// contains glyphs and typesetting information
pub struct FontCollection<'a>(&'a[u8]);
/// what features of the font collection do we want to use?
pub struct FontConfiguration;

/// contains character mapping
pub struct Font<'a>(&'a[u8]);

struct GlyphPoint(i16, i16);
enum ContourSegment {
	Line(GlyphPoint, GlyphPoint),
	Curve(GlyphPoint, GlyphPoint, GlyphPoint)
}

pub fn read_u32(data: &[u8]) -> Option<u32> {
	if data.len() > 3 {
		Some((data[0] as u32) << 24 | (data[1] as u32) << 16 | (data[2] as u32) << 8 | data[3] as u32)
	} else {
		None
	}
}

pub fn read_u16(data: &[u8]) -> Option<u16> {
	if data.len() > 1 {
		Some((data[0] as u16) << 8 | data[1] as u16)
	} else {
		None
	}
}

fn fourcc(tag: u32) -> String {
	let mut s = String::with_capacity(4);
	s.push((tag >> 24) as u8 as char);
	s.push((tag >> 16) as u8 as char);
	s.push((tag >> 8) as u8 as char);
	s.push(tag as u8 as char);
	s
}

macro_rules! try_opt {
	($e:expr) => (match $e {Some(x)=>x, None=>return None})
}

fn scan_table_records(data: &[u8], start: usize, label: u32, num_tables: usize) -> Option<(usize, &[u8])> {
	for pos in start..num_tables {
		if try_opt!(read_u32(&data[12+16*pos..])) == label {
			let start = read_u32(&data[12+16*pos+8..]).unwrap() as usize;
			return Some((pos, &data[start..read_u32(&data[12+16*pos+12..]).unwrap() as usize+start]));
		}
	}
	None
}

fn load_font_cmap(data: &[u8]) -> Option<Font> {
	Some(Font(data))
}

pub fn load_font_collection<'a>(data: &'a[u8], fonts: Vec<&str>) -> Option<(FontCollection<'a>, Vec<Font<'a>>)> {
	let num_tables = try_opt!(read_u16(&data[4..])) as usize;
	let (_pos, cmap) = try_opt!(scan_table_records(data, 0, 0x636d6170, num_tables));
	Some((FontCollection(data), vec![Font(cmap)]))
}

struct GlyphIndex(u16);

struct Burst<'a, 'b: 'a>(Vec<GlyphIndex>, &'a FontCollection<'b>, StreakTypesetting);

pub struct TypeSetting<'a, 'b: 'a>(Vec<Burst<'a, 'b>>);

fn typeset<'a, 'b>(streak: Streak, fonts: &FontPreference<'a>) -> TypeSetting<'b, 'a> {
	TypeSetting(Vec::new())
}