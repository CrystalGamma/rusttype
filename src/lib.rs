use std::ops::Range;
use std::cmp::min;

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
pub enum GlyphOrient {Perpendicular, Parallel}
use self::GlyphOrient::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirBehavior(pub Direction, pub GlyphOrient);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirPreference {Horiz(DirBehavior), Vert(DirBehavior), BiOrient(DirBehavior, DirBehavior)}	// enable scripts without preference?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rot180 {Normal, Rotated}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemTypesetting(pub DirBehavior, pub Rot180);

impl ItemTypesetting {
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

pub fn make_default_streak<'a>(run: &'a Run, main_axis: MainAxis) -> Item<'a> {
	use self::MainAxis::*;
	Item {
		text: run.0,
		script: run.1,
		dir: ItemTypesetting(DirBehavior(Forward, Perpendicular), if run.1 == LATIN {
			match main_axis {
				Ltr | Ttb => Rot180::Normal,
				Rtl | Btt => Rot180::Rotated
			}
		} else {
			unimplemented!();
		})
	}
}

pub fn make_runs(s: &str) -> Vec<Run> {vec![Run(s, LATIN)]}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item<'text>{
	pub text: &'text str,
	pub script: ScriptCode,
	pub dir: ItemTypesetting
}

impl<'a> Item<'a> {
	/// what order do the parts have if you split the streak
	pub fn direction(&self) -> Direction {self.dir.effective_direction()}
}

pub fn bidi_algorithm(streaks: &mut [Item]) {}

pub struct FontPreference<'a>(Font<'a>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FWord(i16);

macro_rules! try_opt {
	($e:expr) => (match $e {Some(x)=>x, None=>return None})
}

/// contains glyphs and typesetting information
#[derive(PartialEq, Clone, Copy)]
pub struct FontCollection<'a>{
	tables: TableDirectory<'a>
}
#[derive(PartialEq, Clone, Copy)]
pub struct TableDirectory<'a>{
	data: &'a[u8],
	num_tables: u16
}
impl<'a> TableDirectory<'a> {
	fn new(data: &'a[u8]) -> Option<TableDirectory<'a>> {
		if data.len() < 12 {return None}
		let version = read_u32(data);
		if version != Some(0x10000) && version != Some(0x4f54544f) {return None}
		let num_tables = try_opt!(read_u16(&data[4..]));
		if data.len() < num_tables as usize*16 + 12 {
			None
		} else {
			Some(TableDirectory {
				data: data,
				num_tables: num_tables
			})
		}
	}
	fn find(&self, start: u16, label: u32) -> Option<(u16, &[u8])> {
		for pos_ in start..self.num_tables {
			let pos = pos_ as usize;
			let candidate = try_opt!(read_u32(&self.data[12+16*pos..]));
			if candidate == label {
				let start = read_u32(&self.data[12+16*pos+8..]).unwrap() as usize;
				return Some((pos as u16, &self.data[start..read_u32(&self.data[12+16*pos+12..]).unwrap() as usize+start]));
			} else if candidate > label {
				return None
			}
		}
		None
	}
}

/// what features of the font collection do we want to use?
pub struct FontConfiguration;

/// contains character mapping
#[derive(PartialEq, Clone, Copy)]
pub struct Font<'a>{
	pub cmap: CMap<'a>,
	glyph_src: &'a FontCollection<'a>
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

fn find_best_cmap(cmap: &[u8]) -> Option<&[u8]> {
	let mut bmp = None;
	for encoding in 0..read_u16(&cmap[2..]).unwrap() as usize {
		let enc_header = &(&cmap[4+8*encoding..])[..8];
		let (plat, enc) = (read_u16(enc_header).unwrap(), read_u16(&enc_header[2..]).unwrap());
		match (plat, enc) {
			(0, 3) | (3, 1) => {bmp=Some(&cmap[try_opt!(read_u32(&enc_header[4..])) as usize..]);},
			(0, 4) | (3, 10) => return Some(&cmap[try_opt!(read_u32(&enc_header[4..])) as usize..]),
			_ => {}	// unknown encoding
		}
	}
	bmp
}

#[derive(PartialEq, Clone, Copy)]
pub struct CMap<'otf>(Encoding<'otf>);

impl<'otf> CMap<'otf> {pub fn lookup(&self, c: char) -> Option<GlyphIndex> {self.0.lookup(c)}}

#[derive(PartialEq, Clone, Copy)]
enum Encoding<'a> {
	Fmt4(CmapFmt4<'a>)
}

impl<'a> Encoding<'a> {
	pub fn lookup(&self, c: char) -> Option<GlyphIndex> {
		match *self {
			Encoding::Fmt4 (CmapFmt4 {end, start, delta, crazy_indexing_part: range_offset}) => {
				if c as u32 > 0xffff {return Some(GlyphIndex(0))}
				let mut range = 0..end.len()/2;
				while range.start != range.end {
					let pivot = ((range.end - range.start) & !1) + range.start*2;
					let pivot_val = read_u16(&end[pivot..]).unwrap();
					range = if pivot_val < c as u16 {
						pivot/2+1..range.end
					} else {
						range.start..pivot/2
					};
				}
				let seg_offset = range.start*2;
				let block_start = read_u16(&start[seg_offset..]).unwrap();
				if block_start > c as u16 {return Some(GlyphIndex(0))}
				let block_index = c as u16 - block_start;
				return Some(GlyphIndex((read_u16(&delta[seg_offset..]).unwrap()).wrapping_add({
					let offset = read_u16(&range_offset[seg_offset..]).unwrap();
					if offset == 0 {
						block_index
					} else {	// this path is untested because the spec is really weird and I've never seen it used
						let res = read_u16(&range_offset[seg_offset+(offset as usize &!1)+block_index as usize..]).unwrap();
						if res == 0 {
							return Some(GlyphIndex(0))
						} else {
							res
						}
					}
				})))
			}
		}
	}
}

#[derive(PartialEq, Clone, Copy)]
struct CmapFmt4<'a> {
	end: &'a[u8],
	start: &'a[u8],
	delta: &'a[u8],
	crazy_indexing_part: &'a[u8]
}

fn load_enc_table(mut enc: &[u8]) -> Option<Encoding> {
	let format = try_opt!(read_u16(enc));
	match format {
		4 => {
			let len = try_opt!(read_u16(&enc[2..])) as usize;
			if len > enc.len() {return None}
			enc = &enc[..len];
			let segsX2 = try_opt!(read_u16(&enc[6..])) as usize;
			if segsX2 < 2 || segsX2 % 2 != 0 || 4*segsX2 + 16 < len {return None}
			let end = &enc[14..14+segsX2];
			if read_u16(&end[segsX2-2..]).unwrap() != 0xffff {return None}
			Some(Encoding::Fmt4(CmapFmt4{
				end: end,
				start: &enc[16+segsX2..16+2*segsX2],
				delta: &enc[16+2*segsX2..16+3*segsX2],
				crazy_indexing_part: &enc[16+3*segsX2..]
			}))
		},
		_ => None	// not implemented
	}
}

pub fn load_font_collection(data: &[u8]) -> Option<FontCollection> {
	let tables = try_opt!(TableDirectory::new(data));
	Some(FontCollection {
		tables: tables
	})
}

static CMAP_TAG: u32 = 0x636d6170;
static HHEA_TAG: u32 = 0x68686561;

pub fn load_font<'a>(collection: &'a FontCollection<'a>, font: &str) -> Option<Font<'a>> {
	let (_pos, cmap) = try_opt!(collection.tables.find(0, CMAP_TAG));
	let best_enc = try_opt!(find_best_cmap(cmap));
	let enc = try_opt!(load_enc_table(best_enc));
	Some(Font {cmap: CMap(enc), glyph_src: collection})
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphIndex(u16);

#[derive(Debug, Clone, Copy)]
pub struct MappedGlyph<'text> {
	pub range: &'text str,
	pub glyph: GlyphIndex,
	pub dir: Direction
}

pub trait CharMapper {
	/// given the text of the cluster, the character index relative to cluster start and the position within that character, return the byte offset into the cluster, where the cursor is
	fn map_cursor(&self, text: &str, character: usize, pos: Fractional) -> usize;
	fn map_range(&self, range: Range<usize>) -> Vec<Range<usize>>;
}

pub trait Shaper {
	type CharMapper: CharMapper;
	/// segment the string into clusters, shape the clusters and look up the glyphs (with given callback)
	///
	/// first result is the characters, second the pairs of (first character index of cluster, string offset of cluster start)
	fn shape<T, F: FnMut(char) -> Option<T>>(&self, text: &str, lookup: &mut F) -> Option<(Vec<T>, Self::CharMapper)>;
}

static MAX_UTF8_LEN: usize = 4;

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct Fractional(u32);
static FRACTIONAL_CENTER: Fractional = Fractional(0x80000000);

pub struct BasicShaper;
impl Shaper for BasicShaper {
	type CharMapper = BasicCharMapper;
	fn shape<T, F: FnMut(char) -> Option<T>>(&self, text: &str, lookup: &mut F) -> Option<(Vec<T>, BasicCharMapper)> {
		let (mut chars, mut clusters) = (Vec::with_capacity(text.len()/MAX_UTF8_LEN), Vec::with_capacity(text.len()/MAX_UTF8_LEN));
		for (i, c) in text.char_indices() {
			chars.push(try_opt!(lookup(c)));
			let last_cluster = clusters.last().cloned().unwrap_or((0, 0));
			let char_size = c.len_utf8();
			if char_size > 1 {
				clusters.push((chars.len(), i+char_size));
			}
		}
		let num_chars = chars.len();
		Some((chars, BasicCharMapper(num_chars, clusters)))
	}
}

/// Simple cluster mapping (cluster = 0.. ASCII chars (len=1) + 0..1 non-ASCII)
/// Tuples are cluster boundaries as (char idx, string offset), a first (0, 0) and the last (num chars, str len) are implied
/// Rationale: compared to cluster=1char, this saves memory for text containing ASCII (no heap alloc for pure ASCII) while being cheap to compute
#[derive(Debug)]
pub struct BasicCharMapper(usize, Vec<(usize, usize)>);
impl CharMapper for BasicCharMapper {
	fn map_cursor(&self, text: &str, character: usize, pos: Fractional) -> usize {unimplemented!()}
	fn map_range(&self, range: Range<usize>) -> Vec<Range<usize>> {
		println!("{:?}", range);
		let translate = |pos:usize| {
			let (cluster_offset, cluster_start, cluster_end) = match self.1.binary_search_by(|&(_, cstart)| cstart.cmp(&pos)) {
				Err(0) => (0, 0, self.1.first().map_or(self.0, |&(x, _)| x)),
				Ok(x) => (self.1[x].1, self.1[x].0, self.1.get(x+1).map_or(self.0, |&(_, x)| x)),
				Err(x) => (self.1[x-1].1, self.1[x-1].0, self.1.get(x).map_or(self.0, |&(_, x)| x))
			};
			println!("{:?}", (cluster_offset, cluster_start, cluster_end));
			min(pos-cluster_offset, cluster_end-cluster_start) + cluster_start
		};
		vec![translate(range.start)..translate(range.end)]
	}
}

#[test]
fn test_basic_shaper() {
	let data = &[
		"",
		"Abc",
		"Test123",
		"Übung",	// multibyte char at the beginning
		"Weiß"	// multibyte char at the end
	];
	for case in data {
		let (chars, mapper) = BasicShaper.shape(case, &mut|x:char| Some(x)).unwrap();
		assert_eq!(chars, case.chars().collect::<Vec<char>>());
		println!("{:?}, {:?}", case, mapper);
		for (cidx, (i, c)) in case.char_indices().enumerate() {
			assert_eq!(&mapper.map_range(i..case.len()), &[cidx..chars.len()]);
		}
		for (cidx, (i, c)) in case.char_indices().rev().enumerate() {
			assert_eq!(&mapper.map_range(0..i), &[0..chars.len()-1-cidx]);
		}
	}
}

pub struct PositionedGlyph {
	glyph: GlyphIndex,
	start: DesignUnits,
	pos_main: DesignUnits,
	pos_cross: DesignUnits
}

struct Layout {
	glyphs: Vec<PositionedGlyph>,
	length: DesignUnits
}

struct DesignUnits(u32);

enum JustificationMode {
	Default,
	Shortest,
	Longest,
	Target {
		length: DesignUnits
	}
}
