use spfunc::gamma::digamma;

type F = f64;


// fn digamma(x: F) -> F { x.ln() - 0.5*x }
// ^ bad, maybe use https://math.stackexchange.com/a/1446110

fn mean_beta(a: F, b: F) -> F { a / (a + b) }
fn meanlog_beta(a: F, b: F) -> F { digamma(a) - digamma(a + b) }
fn mean_mirror(a: F, b: F) -> F { mean_beta(b, a) }
fn meanlog_mirror(a: F, b: F) -> F { meanlog_beta(b, a) }
