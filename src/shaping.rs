use std::ops::Range;
use std::cmp::min;
use super::{Fractional, Shaper, CharMapper};

static MAX_UTF8_LEN: usize = 4;

pub struct BasicShaper;
impl Shaper for BasicShaper {
	type CharMapper = BasicCharMapper;
	fn shape<T, F: FnMut(char) -> Option<T>>(&self, text: &str, lookup: &mut F) -> Option<(Vec<T>, BasicCharMapper)> {
		let (mut chars, mut clusters) = (Vec::with_capacity(text.len()/MAX_UTF8_LEN), Vec::with_capacity(text.len()/MAX_UTF8_LEN));
		for (i, c) in text.char_indices() {
			chars.push(try_opt!(lookup(c)));
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
