struct Kahan {
    sum: f64,
    c: f64,
}

impl Kahan {
    fn new() -> Kahan { Kahan { sum: 0, c: 0 } }
    fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }
}
