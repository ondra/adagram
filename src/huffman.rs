use std::collections::BinaryHeap;
use std::cmp::{Ord,Ordering};
use std::vec::Vec;


#[derive(Debug)]
pub struct HuffmanTree {
    pub nodes: Vec<Node>,
    pub len: usize,
}

#[derive(Debug)]
#[derive(Eq)]
pub struct Node {
    parent: u32,
    lhs: bool,
    freq: u64,
    id: u32,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering { self.freq.cmp(&other.freq) }
    //fn cmp(&self, other: &Self) -> Ordering { other.freq.cmp(&self.freq) }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.freq.cmp(&other.freq)) }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool { self.freq == other.freq }
}


impl HuffmanTree {
    pub fn new(freqs: &[u64]) -> HuffmanTree {
        //let mut nodes = Vec::<Node>::with_capacity(freqs.len());
        let mut id = 0;
        let mut heap = BinaryHeap::<std::cmp::Reverse<Node>>::with_capacity(freqs.len());
        for freq in freqs {
            heap.push(std::cmp::Reverse(Node{parent: u32::MAX, freq: *freq, id: id, lhs: false}));
            id += 1;
        }

        let mut out = Vec::<Node>::with_capacity(freqs.len());
        let mut put = |id: u32, val| {
            if id as usize >= out.len() {
                out.resize_with(id as usize+1,
                    || { Node{parent: u32::MAX, freq: u64::MAX, id: u32::MAX, lhs: false} })
            }
            if out[id as usize].freq != u64::MAX {
                println!("err {:?}", out[id as usize]); 
            }
            out[id as usize] = val;
        };
        
        while heap.len() > 1 {
            let mut e1 = heap.pop().unwrap().0;
            let mut e2 = heap.pop().unwrap().0;

            heap.push(std::cmp::Reverse(Node{ parent: u32::MAX, freq: e1.freq + e2.freq, id: id, lhs: false }));

            e1.parent = id; e1.lhs = true;
            e2.parent = id; e2.lhs = false;
            
            put(e1.id, e1);
            put(e2.id, e2);
            
            id += 1;
        }

        let root = heap.pop().unwrap().0;
        put(root.id, root);
        
        HuffmanTree{nodes: out, len: freqs.len() }
    }

    // get directions through the Huffman tree from node _id_ to the root node
    pub fn softmax_path(&self, id: u32) -> (Vec<bool>, Vec<u32>) {
        let mut path = Vec::<bool>::new();
        let mut nodes = Vec::<u32>::new();
        let root_node_id = self.nodes.last().unwrap().id;
        
        let mut cur_id = id;
        while cur_id != root_node_id {
            let cur_node = &self.nodes[cur_id as usize];
            path.push(cur_node.lhs);
            nodes.push(cur_node.parent - self.len as u32); 
            cur_id = cur_node.parent;
        }
        
        path.reverse(); nodes.reverse();
        (path, nodes)
    }

    pub fn _convert(&self) {
        for id in 0..self.len {
            dbg!(id, self.softmax_path(id as u32));
        }
    }
}
