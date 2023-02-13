#[derive(Debug)]
pub struct RunningStats {
    m_n: usize,
    m_oldm: f64,
    m_newm: f64,
    m_olds: f64,
    m_news: f64,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningStats {
    pub fn new() -> RunningStats {
        RunningStats { m_n: 0, m_oldm: 0., m_newm: 0., m_olds: 0., m_news: 0. }
    }
    pub fn clear(&mut self) {
        self.m_n = 0;
    }

    pub fn push(&mut self, x: f64) {
        self.m_n += 1;

        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if self.m_n == 1 {
            self.m_oldm = x;
            self.m_newm = x;
            self.m_olds = 0.0;
        } else {
            self.m_newm = self.m_oldm + (x - self.m_oldm)/self.m_n as f64;
            self.m_news = self.m_olds + (x - self.m_oldm)*(x - self.m_newm);

            // set up for next iteration
            self.m_oldm = self.m_newm;
            self.m_olds = self.m_news;
        }
    }

    pub fn n(&self) -> usize { self.m_n }
    pub fn mean(&self) -> f64 {
        if self.m_n >= 1 { self.m_newm } else { 0.0 }
    }
    pub fn var(&self) -> f64 {
        if self.m_n > 1 { self.m_news/(self.m_n - 1) as f64 } else { 0.0 }
    }
    pub fn stddev(&self) -> f64 {
        self.var().sqrt()
    }
}
