//! Comparison with KrylovKit.jl on identical inputs.
//!
//! `fixtures/krylovkit/<family>.txt` is produced by
//! `fixtures/krylovkit/generate.jl` from `<family>.jl` (KrylovKit pinned in
//! its Manifest; the version and tree hash are on the first line of each
//! fixture file). Both
//! sides use CGS2, the same matrices, start vectors and options.

mod common;

use std::collections::HashMap;

use common::*;
use krylov::{Lanczos, Which, Workspace, eigsolve, exponentiate};
use num_complex::Complex64;

const FIXTURES: [&str; 2] = [
    include_str!("fixtures/krylovkit/eigsolve.txt"),
    include_str!("fixtures/krylovkit/exponentiate.txt"),
];

fn cases() -> Vec<HashMap<String, String>> {
    let mut out = Vec::new();
    let mut cur = HashMap::new();
    for line in FIXTURES
        .iter()
        .flat_map(|f| f.lines())
        .filter(|l| !l.starts_with('#'))
    {
        if line == "end" {
            out.push(std::mem::take(&mut cur));
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        cur.insert(k.to_string(), v.to_string());
    }
    out
}

fn num<T: std::str::FromStr>(c: &HashMap<String, String>, k: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    c[k].parse().unwrap()
}

fn floats(c: &HashMap<String, String>, k: &str) -> Vec<f64> {
    c[k].split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}

fn alg(c: &HashMap<String, String>) -> Lanczos {
    Lanczos::new()
        .krylovdim(num(c, "krylovdim"))
        .tol(num(c, "tol"))
        .eager(num::<u8>(c, "eager") == 1)
        .maxiter(100)
}

fn inputs(c: &HashMap<String, String>) -> (nalgebra::DMatrix<Complex64>, Vec<Complex64>) {
    let complex = num::<u8>(c, "complex") == 1;
    let n: usize = num(c, "n");
    let mut g = SplitMix(num(c, "seed"));
    let a = hermitian(&mut g, n, complex);
    let x = random_vec(&mut g, n, complex);
    (a, x)
}

fn eig_case<S: TestScalar>(c: &HashMap<String, String>) {
    let (a, x) = inputs(c);
    let x0 = to_dvec::<S>(&x);
    let which = match c["which"].as_str() {
        "SR" => Which::SmallestReal,
        "LR" => Which::LargestReal,
        "LM" => Which::LargestMagnitude,
        w => panic!("unknown selector {w}"),
    };
    let alg = alg(c);
    let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim()).unwrap();
    let howmany = num(c, "howmany");
    let (vals, _, info) =
        eigsolve(&mut DenseOp::new(&a), &x0, howmany, which, &alg, &mut ws).unwrap();
    let expect = floats(c, "values");
    let name = &c["case"];
    eprintln!(
        "{name}: krylov numiter {} numops {} converged {} | KrylovKit {} {} {}",
        info.numiter, info.numops, info.converged, c["numiter"], c["numops"], c["converged"]
    );
    assert_eq!(
        vals.len(),
        expect.len(),
        "{name}: number of returned values"
    );
    assert_eq!(info.converged, num::<usize>(c, "converged"), "{name}");
    for (v, e) in vals.iter().zip(&expect) {
        assert!(
            (to_f64(*v) - e).abs() <= 1e-9 * e.abs().max(1.0),
            "{name}: {v:?} vs {e}"
        );
    }
    let tol: f64 = num(c, "tol");
    for r in &info.normres {
        assert!(to_f64(*r) <= tol || info.converged < vals.len());
    }
    let (ni, no): (usize, usize) = (num(c, "numiter"), num(c, "numops"));
    assert!(
        info.numiter.abs_diff(ni) <= 1,
        "{name}: numiter {} vs {ni}",
        info.numiter
    );
    assert!(
        info.numops.abs_diff(no) * 10 <= no,
        "{name}: numops {} vs {no}",
        info.numops
    );
}

fn exp_case<S: TestScalar>(c: &HashMap<String, String>) {
    let (a, x) = inputs(c);
    let x0 = to_dvec::<S>(&x);
    let t = floats(c, "t");
    let t = Complex64::new(t[0], t[1]);
    let alg = alg(c);
    let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim()).unwrap();
    let (y, info) =
        exponentiate(&mut DenseOp::new(&a), S::from_c64(t), &x0, &alg, &mut ws).unwrap();
    let expect: Vec<Complex64> = floats(c, "y_re")
        .into_iter()
        .zip(floats(c, "y_im"))
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    let name = &c["case"];
    eprintln!(
        "{name}: krylov numiter {} numops {} err {:?} | KrylovKit {} {} {}",
        info.numiter, info.numops, info.normres, c["numiter"], c["numops"], c["normres"]
    );
    let rel = diff_norm(&to_c64(&y), &expect) / norm(&expect);
    assert!(rel < 1e-10, "{name}: relative difference {rel}");
    assert_eq!(info.converged, num::<usize>(c, "converged"), "{name}");
    let (ni, no): (usize, usize) = (num(c, "numiter"), num(c, "numops"));
    assert!(
        info.numiter.abs_diff(ni) * 10 <= ni.max(10),
        "{name}: numiter {} vs {ni}",
        info.numiter
    );
    assert!(
        info.numops.abs_diff(no) * 10 <= no,
        "{name}: numops {} vs {no}",
        info.numops
    );
}

#[test]
fn matches_krylovkit_fixtures() {
    let cases = cases();
    assert!(cases.len() >= 7);
    for c in &cases {
        let complex = num::<u8>(c, "complex") == 1;
        match (c["kind"].as_str(), complex) {
            ("eig", false) => eig_case::<f64>(c),
            ("eig", true) => eig_case::<Complex64>(c),
            ("exp", false) => exp_case::<f64>(c),
            ("exp", true) => exp_case::<Complex64>(c),
            (k, _) => panic!("unknown kind {k}"),
        }
    }
}
