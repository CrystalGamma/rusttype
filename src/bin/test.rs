extern crate rusttype;
use rusttype::{load_font, load_font_collection};

fn main() {
	let font_file = include_bytes!("Oxygen-Sans.ttf");
	let collection = load_font_collection(font_file).unwrap();
	let font = load_font(&collection, "").unwrap();
	println!("A: {:?} b: {:?} c: {:?}", font.0.lookup('A').unwrap(), font.0.lookup('b').unwrap(), font.0.lookup('c').unwrap())
}