extern crate rusttype;
use rusttype::*;

fn main() {
	let font_file = include_bytes!("SourceSansPro-Regular.ttf");
	let collection = load_font_collection(font_file).unwrap();
	let font = load_font(&collection, "").unwrap();
	println!("A: {:?} b: {:?} c: {:?}", font.cmap.lookup('A').unwrap(), font.cmap.lookup('b').unwrap(), font.cmap.lookup('c').unwrap());
	println!("{:?}", BasicShaper.shape("Abc", &mut|x|Some(x)));
	let gsub = collection.gsub.unwrap();
	let out = |tag: Tag<LangSys>, langsys: LangSysTable|println!("	{} features: {:?} mandatory, others: {:?}", fourcc(tag.0), langsys.required_feature(), langsys.features().collect::<Vec<_>>());
	for (i, (tag, script)) in gsub.script_list.into_iter().enumerate() {
		println!("{}: {:x} {}", i, tag.0, fourcc(tag.0));
		let script = script.unwrap();
		out(Tag::new(0x20202020), script.default_lang_sys().unwrap());
		for (tag, langsys) in script {
			out(tag, langsys.unwrap());
		}
	}
	let latin = gsub.script_list.features_for(Some((Tag::new(0x6c61746e), None))).unwrap();
}
