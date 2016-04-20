use std::ops::Range;
use std::cmp::{min, Ordering};

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
impl AutoSearch {pub fn new(list_size: usize) -> AutoSearch {if list_size > 4096 {AutoSearch::Bin} else {AutoSearch::Lin}}}
