use std::ops::Range;
use std::cmp::{min, Ordering};
use std::error::Error;

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
	pub tables: TableDirectory<'a>,
	pub gsub: Option<GSub<'a>>
}
#[derive(PartialEq, Clone, Copy)]
pub struct TableDirectory<'a>{
	data: &'a[u8],
	num_tables: u16
}
impl<'a> TableDirectory<'a> {
	fn new(data: &[u8]) -> Result<TableDirectory, CorruptFont> {
		if data.len() < 12 {return Err(CorruptFont(data, TableTooShort))}
		let version = read_u32(data);
		if version != Some(0x10000) && version != Some(0x4f54544f) {return Err(CorruptFont(data, ReservedFeature))}
		let num_tables = read_u16(&data[4..]).unwrap();
		if data.len() < num_tables as usize*16 + 12 {
			Err(CorruptFont(data, TableTooShort))
		} else {
			Ok(TableDirectory {
				data: data,
				num_tables: num_tables
			})
		}
	}
	fn find(&self, start: u16, label: u32) -> Option<(u16, &'a[u8])> {
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

pub fn fourcc(tag: u32) -> String {
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
				return Some(GlyphIndex((read_u16(&delta[seg_offset..]).unwrap()).wrapping_add({
					let offset = read_u16(&range_offset[seg_offset..]).unwrap();
					if offset == 0 {
						println!("delta: {} start: {}", read_u16(&delta[seg_offset..]).unwrap(), block_start);
						c as u16
					} else {	// this path is untested because the spec is really weird and I've never seen it used
						let res = read_u16(&range_offset[seg_offset+(offset as usize &!1)+(c as usize - block_start as usize)..]).unwrap();
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

fn load_enc_table(mut enc: &[u8]) -> Result<Encoding, CorruptFont> {
	if enc.len() < 4 {return Err(CorruptFont(enc, TableTooShort))}
	let format = read_u16(enc).unwrap();
	match format {
		4 => {
			let len = read_u16(&enc[2..]).unwrap() as usize;
			if len > enc.len() && len < 14 {return Err(CorruptFont(enc, TableTooShort))}
			enc = &enc[..len];
			let segsX2 = read_u16(&enc[6..]).unwrap() as usize;
			if segsX2 % 2 != 0 {return Err(CorruptFont(enc, OddSegsX2))}
			if segsX2 < 2 || 4*segsX2 + 16 > len {return Err(CorruptFont(enc, CmapInvalidSegmentCount))}
			let end = &enc[14..14+segsX2];
			if read_u16(&end[segsX2-2..]).unwrap() != 0xffff {return Err(CorruptFont(enc, CmapMissingGuard))}
			Ok(Encoding::Fmt4(CmapFmt4{
				end: end,
				start: &enc[16+segsX2..16+2*segsX2],
				delta: &enc[16+2*segsX2..16+3*segsX2],
				crazy_indexing_part: &enc[16+3*segsX2..]
			}))
		},
		_ => Err(CorruptFont(enc, Unimplemented))
	}
}

pub fn load_font_collection(data: &[u8]) -> Result<FontCollection, CorruptFont> {
	let tables = try!(TableDirectory::new(data));
	println!("#glyphs: {:?}", tables.find(0, 0x6d617870).and_then(|x|read_u16(&x.1[4..])));
	let gsub = if let Some(x) = tables.find(0, GSUB_TAG) {
		Some(try!(GSub::new(x.1)))
	} else {None};
	Ok(FontCollection {
		gsub: gsub,
		tables: tables
	})
}

static CMAP_TAG: u32 = 0x636d6170;
static HHEA_TAG: u32 = 0x68686561;
static GSUB_TAG: u32 = 0x47535542;

pub fn load_font<'a>(collection: &'a FontCollection<'a>, font: &str) -> Result<Font<'a>, CorruptFont<'a>> {
	let (_pos, cmap) = try!(collection.tables.find(0, CMAP_TAG).ok_or(CorruptFont(collection.tables.data, NoCmap)));
	let best_enc = try!(find_best_cmap(cmap).ok_or(CorruptFont(cmap, NoCmap)));
	let enc = try!(load_enc_table(best_enc));
	Ok(Font {cmap: CMap(enc), glyph_src: collection})
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
		let translate = |pos:usize| {
			let (cluster_offset, cluster_start, cluster_end) = match self.1.binary_search_by(|&(_, cstart)| cstart.cmp(&pos)) {
				Err(0) => (0, 0, self.1.first().map_or(self.0, |&(x, _)| x)),
				Ok(x) => (self.1[x].1, self.1[x].0, self.1.get(x+1).map_or(self.0, |&(_, x)| x)),
				Err(x) => (self.1[x-1].1, self.1[x-1].0, self.1.get(x).map_or(self.0, |&(_, x)| x))
			};
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
		for (cidx, (i, c)) in case.char_indices().enumerate() {
			assert_eq!(&mapper.map_range(i..case.len()), &[cidx..chars.len()]);
		}
		for (cidx, (i, c)) in case.char_indices().rev().enumerate() {
			assert_eq!(&mapper.map_range(0..i), &[0..chars.len()-1-cidx]);
		}
	}
}

trait GlyphMapper {
	fn new() -> Self;
	fn map_char_to_glyph(&self, char_idx: usize, pos: Fractional) -> Option<(usize, Fractional)>;
	fn map_glyph_to_char(&self, glyph_idx: usize, pos: Fractional) -> Option<(usize, Fractional)>;
}
trait MonotonicSubstitution {
	type GlyphMapper: GlyphMapper;
	/// performs the substitution. Returns the number of characters affected.
	fn substitute_mutate(&self, forward: &mut Vec<GlyphIndex>, backward: &mut Vec<GlyphIndex>, map_fwd: &mut Self::GlyphMapper, map_bwd: &mut Self::GlyphMapper) -> Option<usize>;
	fn substitute(&self, mut glyphs: Vec<GlyphIndex>) -> Option<(Vec<GlyphIndex>, Self::GlyphMapper)> {
		let (mut m1, mut m2) = (Self::GlyphMapper::new(), Self::GlyphMapper::new());
		try_opt!(self.substitute_mutate(&mut glyphs, &mut Vec::new(), &mut m1, &mut m2));
		Some((glyphs, m1))
	}
}

#[derive(Clone, Copy, Debug)]
enum FontCorruption {
	Unimplemented,
	ReservedFeature,
	TableTooShort,
	OffsetOutOfBounds,
	IncorrectDfltScript,
	CmapInvalidSegmentCount,
	OddSegsX2,
	CmapMissingGuard,
	NoCmap,
	UnknownTableVersion
}
use FontCorruption::*;

#[derive(Clone, Copy, Debug)]
pub struct CorruptFont<'a>(&'a[u8], FontCorruption);

impl<'a> Error for CorruptFont<'a> {
	fn description(&self) -> &str {match self.1 {
		Unimplemented => "The font uses a feature that is not implemented",
		ReservedFeature => "A reserved field differed from the default value",
		TableTooShort => "Unexpected end of table",
		OffsetOutOfBounds => "An Offset pointed outside of the respective table",
		IncorrectDfltScript => "'DFLT' script with missing DefaultLangSys or LangSysCount ≠ 0",
		CmapInvalidSegmentCount => "The segment count in the character mapping is invalid",
		OddSegsX2 => "The doubled segment count in the character mapping is not an even number",
		CmapMissingGuard => "The character mapping is missing a guard value",
		NoCmap => "No character mapping found",
		UnknownTableVersion => "The font uses a table version that is not recognised"
	}}
}
impl<'a> ::std::fmt::Display for CorruptFont<'a> {
	fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
		write!(f, "{}", self.description())
	}
}

pub trait SearchStrategy<Idx> {
	/// do the search. The argument to the callback may never leave the given range.
	/// returns Ok(idx) when found, Err(Ok(idx)) when not found and Err(Err((idx, error))) when the callback returned Err(error) at pos idx
	fn search<E, F: FnMut(&Idx) -> Result<Ordering, E>>(&self, range: Range<Idx>, func: &mut F) -> Result<Idx, Result<Idx, (Idx, E)>>;
}

pub struct LinearSearch;
pub struct BinarySearch;
macro_rules! impl_strategies {($typ: ty) => (
	impl SearchStrategy<$typ> for LinearSearch {
		fn search<E, F: FnMut(&$typ) -> Result<Ordering, E>>(&self, range: Range<$typ>, func: &mut F) -> Result<$typ, Result<$typ, ($typ, E)>> {
			for i in range.clone() {	// for some reason Range does not derive Copy
				match func(&i) {
					Ok(Ordering::Equal) => return Ok(i),
					Ok(Ordering::Greater) => return Err(Ok(i)),
					Err(e) => return Err(Err((i, e))),
					_ => {}
				}
			}
			Err(Ok(range.end))
		}
	}
	impl SearchStrategy<$typ> for BinarySearch {
		fn search<E, F: FnMut(&$typ) -> Result<Ordering, E>>(&self, mut range: Range<$typ>, func: &mut F) -> Result<$typ, Result<$typ, ($typ, E)>> {
			assert!(range.start <= range.end);
			while range.start != range.end {
				let pivot = (range.end - range.start) / 2 + range.start;
				match func(&pivot) {
					Ok(Ordering::Equal) => return Ok(pivot),
					Ok(Ordering::Greater) => {range.end = pivot},
					Err(e) => return Err(Err((pivot, e))),
					_ => {range.start = pivot + 1}
				}
			}
			Err(Ok(range.start))
		}
	}
	impl SearchStrategy<$typ> for AutoSearch {
		fn search<E, F: FnMut(&$typ) -> Result<Ordering, E>>(&self, range: Range<$typ>, func: &mut F) -> Result<$typ, Result<$typ, ($typ, E)>> {
			match *self {
				AutoSearch::Bin => BinarySearch.search(range, func),
				AutoSearch::Lin => LinearSearch.search(range, func)
			}
		}
	}
)}
impl_strategies!(u8);
impl_strategies!(u16);
impl_strategies!(u32);
impl_strategies!(usize);

pub enum AutoSearch {Lin, Bin}
impl AutoSearch {fn new(list_size: usize) -> AutoSearch {if list_size > 4096 {AutoSearch::Bin} else {AutoSearch::Lin}}}

#[derive(Debug)]
pub struct FeatureIndex(pub u16);

pub struct LangSysTable<'a>(&'a[u8]);

impl<'a> LangSysTable<'a> {
	pub fn new(data: &'a[u8]) -> Result<LangSysTable<'a>, CorruptFont<'a>> {
		if data.len() < 6 {return Err(CorruptFont(data, TableTooShort))}
		if read_u16(data).unwrap() != 0 {return Err(CorruptFont(data, ReservedFeature))}
		let num_features = read_u16(&data[4..]).unwrap();
		if data.len() - 6 < num_features as usize*2 {return Err(CorruptFont(data, TableTooShort))}
		Ok(LangSysTable(&data[2..num_features as usize*2 + 6]))
	}
	pub fn num_features(&self) -> u16 {(self.0.len() / 2 - 2) as u16}
	pub fn required_feature(&self) -> Option<FeatureIndex> {
		let res = read_u16(self.0).unwrap();
		if res == 0xffff {None} else {Some(FeatureIndex(res))}
	}
	pub fn get_feature(&self, idx: u16) -> Option<FeatureIndex> {read_u16(&self.0[2 + idx as usize*2..]).map(|x| FeatureIndex(x))}
}

static DEFAULT_LANGSYS_TABLE: [u8; 4] = [255, 255, 0, 0];

impl<'a> Default for LangSysTable<'a> {
	fn default() -> LangSysTable<'a> {LangSysTable(&DEFAULT_LANGSYS_TABLE)}
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LangSys;
pub type ScriptTable<'a> = TagOffsetList<'a, LangSys>;
impl<'a> TagListTable<'a> for LangSys {
	type Table = LangSysTable<'a>;
	fn bias() -> usize {2}
	fn new(data: &'a[u8]) -> Result<Self::Table, CorruptFont<'a>> {LangSysTable::new(data)}
}
impl Tagged for LangSys {}

impl<'a> ScriptTable<'a> {
	pub fn new(data: &'a[u8]) -> Result<ScriptTable<'a>, CorruptFont<'a>> {ScriptTable::new_list(data)}
	pub fn default_lang_sys(&self) -> Result<LangSysTable<'a>, CorruptFont<'a>> {
		let offset = read_u16(self.0).unwrap() as usize;
		println!("LS offset {:x}", offset);
		if offset == 0 {
			Ok(Default::default())
		} else {
			if self.0.len() < offset {return Err(CorruptFont(self.0, OffsetOutOfBounds))}
			LangSysTable::new(&self.0[offset..])
		}
	}
	pub fn validate_dflt(&self) -> Result<(), CorruptFont<'a>> {
		if read_u16(self.0).unwrap() != 0 && self.num_tables() == 0 {
			Ok(())
		} else {
			Err(CorruptFont(self.0, IncorrectDfltScript))
		}
	}
}

use std::marker::PhantomData;

pub trait TagListTable<'a> {
	type Table;
	fn bias() -> usize {0}
	fn new(data: &'a[u8]) -> Result<Self::Table, CorruptFont<'a>>;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TagOffsetList<'a, Table: TagListTable<'a>>(&'a[u8], PhantomData<Table>);

pub trait Tagged {}
#[derive(Clone, Copy)]
pub struct Tag<Table: Tagged>(pub u32, PhantomData<Table>);
impl<T: Tagged> Tag<T> {pub fn new(v: u32) -> Tag<T> {Tag(v, PhantomData)}}

impl<'a, Table: TagListTable<'a> + Tagged> TagOffsetList<'a, Table> {
	fn new_list(data: &'a[u8]) -> Result<TagOffsetList<'a, Table>, CorruptFont<'a>> {
		if data.len() < 2 + Table::bias() {return Err(CorruptFont(data, TableTooShort))}
		let res = TagOffsetList(data, PhantomData);
		if data.len() < res.num_tables() as usize*6 + 2 + Table::bias() {return Err(CorruptFont(data, TableTooShort))}
		Ok(res)
	}
	pub fn num_tables(&self) -> u16 {read_u16(&self.0[Table::bias()..]).unwrap()}
	pub fn tag(&self, idx: u16) -> Option<Tag<Table>> {read_u32(&self.0[idx as usize*6+2+Table::bias()..]).map(|x|Tag(x, PhantomData))}
	pub fn table(&self, script_index: u16) -> Result<Table::Table, CorruptFont<'a>> {
		let offset_pos = &self.0[script_index as usize*6 + 6 + Table::bias()..];
		let offset = read_u16(offset_pos).unwrap() as usize;
		if self.0.len() < offset {return Err(CorruptFont(offset_pos, OffsetOutOfBounds))}
		println!("offset {:x}", offset);
		Table::new(&self.0[offset..])
	}
}
impl<'a, Table: TagListTable<'a> + Tagged> IntoIterator for TagOffsetList<'a, Table> {
	type Item = (Tag<Table>, Table::Table);
	type IntoIter = TagOffsetIterator<'a, Table>;
	fn into_iter(self) -> TagOffsetIterator<'a, Table> {TagOffsetIterator(self, 0, PhantomData)}
}

#[derive(Clone, Copy)]
pub struct TagOffsetIterator<'a, Table: TagListTable<'a>>(TagOffsetList<'a, Table>, u16, PhantomData<Table>);
impl<'a, Table: TagListTable<'a> + Tagged> Iterator for TagOffsetIterator<'a, Table> {
	type Item = (Tag<Table>, Table::Table);
	fn next(&mut self) -> Option<(Tag<Table>, Table::Table)> {
		if self.1 >= self.0.num_tables() {
			None
		} else {
			self.1 += 1;
			Some((self.0.tag(self.1 - 1).unwrap(), self.0.table(self.1 - 1).unwrap()))
		}
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Script;
pub type ScriptList<'a> = TagOffsetList<'a, Script>;
impl<'a> TagListTable<'a> for Script {type Table = ScriptTable<'a>; fn new(data: &'a[u8]) -> Result<Self::Table, CorruptFont<'a>> {ScriptTable::new(data)}}
impl Tagged for Script {}
impl<'a> ScriptList<'a> {
	pub fn new(data: &'a[u8]) -> Result<ScriptList<'a>, CorruptFont<'a>> {ScriptList::new_list(data)}
	pub fn features_for(&self, selector: Option<(Tag<Script>, Option<Tag<LangSys>>)>) -> Result<LangSysTable<'a>, CorruptFont<'a>> {
		let search = AutoSearch::new(self.num_tables() as usize*6);
		if let Some((script, lang_sys_opt)) = selector {
			match search.search(0..self.num_tables(), &mut move|&i| Ok(self.tag(i).unwrap().0.cmp(&script.0))) {
				Ok(idx) => {
					let script_table = try!(self.table(idx));
					if let Some(lang_sys) = lang_sys_opt {
						unimplemented!()
					} else {
						return script_table.default_lang_sys()
					}
				},
				Err(Ok(_)) => {println!("default");return Ok(Default::default())},
				Err(Err((_, e))) => return Err(e)
			}
		}
		match search.search(0..self.num_tables(), &mut move|&i| Ok(self.tag(i).unwrap().0.cmp(&DFLT_TAG.0))) {
			Ok(i) => {
				let script_table = try!(self.table(i));
				try!(script_table.validate_dflt());
				script_table.default_lang_sys()
			},
			Err(Ok(_)) => Ok(Default::default()),
			Err(Err((_, e))) => Err(e)
		}
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Feature;
pub type FeatureList<'a> = TagOffsetList<'a, Feature>;
impl<'a> TagListTable<'a> for Feature {type Table = FeatureTable<'a>; fn new(data: &'a[u8]) -> Result<Self::Table, CorruptFont<'a>> {FeatureTable::new(data)}}
impl Tagged for Feature {}

impl<'a> FeatureList<'a> {fn new(data: &'a[u8]) -> Result<FeatureList<'a>, CorruptFont<'a>> {FeatureList::new_list(data)}}

pub struct FeatureTable<'a>(&'a[u8]);
impl<'a> FeatureTable<'a> {
	fn new(data: &'a[u8]) -> Result<FeatureTable<'a>, CorruptFont<'a>> {
		if data.len() < 4 {return Err(CorruptFont(data, TableTooShort))}
		if read_u16(data).unwrap() != 0 {return Err(CorruptFont(data, ReservedFeature))}
		let len = read_u16(&data[2..]).unwrap();
		if len as usize*2+4 > data.len() {return Err(CorruptFont(data, TableTooShort))}
		Ok(FeatureTable(&data[4..len as usize*2+4]))
	}
}

static DFLT_TAG: Tag<Script> = Tag(0x44464c54, PhantomData);


#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GSub<'a> {
	pub script_list: ScriptList<'a>,
	pub feature_list: FeatureList<'a>,
	pub lookup_list: ()
}

impl<'a> GSub<'a> {
	fn new(data: &'a[u8]) -> Result<GSub<'a>, CorruptFont<'a>> {
		if data.len() < 10 {return Err(CorruptFont(data, TableTooShort))}
		if read_u32(data) != Some(0x00010000) {return Err(CorruptFont(data, UnknownTableVersion))}
		let scr_off = read_u16(&data[4..]).unwrap() as usize;
		if data.len() < scr_off + 2 {return Err(CorruptFont(data, TableTooShort))}
		let feat_off = read_u16(&data[6..]).unwrap() as usize;
		Ok(GSub {
			script_list: try!(ScriptList::new(&data[scr_off..])),
			feature_list: try!(FeatureList::new(&data[scr_off..])),
			lookup_list: ()
		})
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
