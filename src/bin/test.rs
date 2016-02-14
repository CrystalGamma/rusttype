extern crate rusttype;
use rusttype::*;

fn main() {
	let font_file = include_bytes!("Oxygen-Sans.ttf");
	let collection = load_font_collection(font_file).unwrap();
	let font = load_font(&collection, "").unwrap();
	println!("A: {:?} b: {:?} c: {:?}", font.cmap.lookup('A').unwrap(), font.cmap.lookup('b').unwrap(), font.cmap.lookup('c').unwrap());
	println!("{:?}", BasicShaper.shape("Abc", &mut|x|Some(x)));
}