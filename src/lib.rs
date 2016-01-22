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

/// contains glyphs and typesetting information
pub struct FontCollection<'a>(&'a[u8]);
/// what features of the font collection do we want to use?
pub struct FontConfiguration;

/// contains character mapping
pub struct Font<'a>{
	pub cmap: CMap<'a>,
	glyph_src: &'a FontCollection<'a>
}

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

pub struct CMap<'otf>(Encoding<'otf>);

impl<'otf> CMap<'otf> {pub fn lookup(&self, c: char) -> Option<GlyphIndex> {self.0.lookup(c)}}

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
	Some(FontCollection(data))
}

pub fn load_font<'a>(collection: &'a FontCollection<'a>, font: &str) -> Option<Font<'a>> {
	let data = collection.0;
	let num_tables = try_opt!(read_u16(&data[4..])) as usize;
	let (_pos, cmap) = try_opt!(scan_table_records(data, 0, 0x636d6170, num_tables));
	let best_enc = try_opt!(find_best_cmap(cmap));
	let enc = try_opt!(load_enc_table(best_enc));
	Some(Font {cmap: CMap(enc), glyph_src: collection})
}

#[derive(Debug)]
pub struct GlyphIndex(u16);

#[derive(Debug)]
pub struct GlyphMap<'text> {
	pub range: &'text str,
	pub dir: Direction,
	pub glyph_off: usize,
	pub span: u16	// between which caret positions of a ligature?
}

pub struct Burst<'text, 'a, 'b: 'a> {
	pub glyphs: Vec<GlyphIndex>,
	pub font: &'a Font<'b>,
	pub dir: ItemTypesetting,
	pub glyph_map: Vec<GlyphMap<'text>>
}

pub fn shape<'text, 'a, 'b: 'a>(item: Item<'text>, font: &'a Font<'b>) -> Option<Burst<'text, 'a, 'b>> {
	let (mut glyphs, mut glyph_map) = (Vec::with_capacity(item.text.len()), Vec::with_capacity(item.text.len()));	// 1 char ⇒ 1 glyph (at this stage) is at least one byte long
	let mut rest = item.text;
	while rest.len() > 0 {
		let c = rest.chars().next().unwrap();
		let idx = try_opt!(font.cmap.lookup(c));
		glyphs.push(idx);
		let size = c.len_utf8();
		glyph_map.push(GlyphMap {
			range: &rest[..size],
			dir: Forward,
			glyph_off: glyphs.len() - 1,
			span: 0
		});
		rest = &rest[size..];
	}
	Some(Burst {glyphs: glyphs, glyph_map: glyph_map, font: font, dir: item.dir})
}