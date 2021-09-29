use std::env;

mod huffman;

fn main() {
    let args: Vec<String> = env::args().collect();

    let ht = huffman::HuffmanTree::new(&[1,2,3,10,4,6,9]);

    println!("{:?}", ht.path(0));
    println!("{:?}", ht.path(1));
    println!("{:?}", ht.path(2));
    println!("{:?}", ht.path(3));
    println!("{:?}", ht.path(4));
    println!("{:?}", ht.path(5));
    println!("{:?}", ht.nodes);

    println!("Hello, world! {}", args[0]);
}
