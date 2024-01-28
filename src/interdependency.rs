use std::collections::{HashMap, HashSet};

use crate::traits::{Contains, Dependency, TokenFilter, Tokenize};
use itertools::Itertools;

use rayon::prelude::*;

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct PairSet<'a>(&'a str, &'a str);

impl<'a> PairSet<'a> {
    pub fn with(t1: &'a str, t2: &'a str) -> Self {
        if t1 > t2 {
            PairSet(t1, t2)
        } else {
            PairSet(t2, t1)
        }
    }
}

#[derive(Debug)]
pub struct Interdependency<'a> {
    soc: HashMap<&'a str, usize>,
    poc: HashMap<PairSet<'a>, usize>,
}

impl<'a> Interdependency<'a> {
    pub fn new<Message, Tokenizer, Filter>(
        msgs: &'a [Message],
        tokenizer: &Tokenizer,
        tf: &Filter,
    ) -> Self
    where
        Message: AsRef<str> + Sync,
        Tokenizer: Tokenize + Sync,
        Filter: TokenFilter + Sync,
    {
        // Self {
        let (soc, poc) = msgs
            .iter()
            .par_bridge()
            .fold_with(
                (HashMap::new(), HashMap::new()),
                |(mut soc, mut poc), msg| {
                    let toks = tokenizer
                        .tokenize(msg.as_ref())
                        .into_iter()
                        .unique()
                        .filter(|tok| tf.token_filter(tok))
                        .map(|tok| tok.as_str())
                        .collect::<HashSet<_>>();

                    // Insert single occurances
                    for tok in &toks {
                        soc.entry(*tok)
                            .and_modify(|count| *count += 1)
                            .or_insert(1_usize);
                    }

                    // Insert double occurances
                    for (tok1, tok2) in toks.iter().tuple_combinations() {
                        poc.entry(PairSet::with(tok1, tok2))
                            .and_modify(|count| *count += 1)
                            .or_insert(1_usize);
                    }
                    (soc, poc)
                },
            )
            .reduce_with(|(mut soc1, mut poc1), (soc2, poc2)| {
                for (tok, count2) in soc2 {
                    soc1.entry(tok)
                        .and_modify(|count1| *count1 += count2)
                        .or_insert(count2);
                }
                for (pair, count2) in poc2 {
                    poc1.entry(pair)
                        .and_modify(|count1| *count1 += count2)
                        .or_insert(count2);
                }
                (soc1, poc1)
            })
            .unwrap();

        Self { soc, poc }
    }
}

impl<'a> Dependency<&'a str> for Interdependency<'a> {
    fn dependency(&self, eve: &'a str, con: &'a str) -> f32 {
        let double = *self
            .poc
            .get(&PairSet::with(eve, con))
            .unwrap_or_else(|| panic!("Pair {:?} not found in occurances", [eve, con]));
        let single = *self
            .soc
            .get(eve)
            .unwrap_or_else(|| panic!("Word '{}' not found in occurances", eve));
        (double as f32) / (single as f32)
    }
}

impl<'a> Contains<&'a str> for Interdependency<'a> {
    fn contains(&self, item: &'a str) -> bool {
        self.soc.contains_key(item)
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::Token;

    use super::*;

    #[test]
    fn test_all() {
        let msg = ["a x1 b", "a x2 b", "a x3 c", "a x4 c"];
        let tokenizer = MockTokenizer;
        let filter = MockFilter;
        let idep = Interdependency::new(&msg, &tokenizer, &filter);
        let expected_soc = HashMap::from([
            ("a", 4),
            ("b", 2),
            ("c", 2),
            ("x1", 1),
            ("x2", 1),
            ("x3", 1),
            ("x4", 1),
        ]);
        let expected_poc = HashMap::from([
            (PairSet::with("a", "x1"), 1),
            (PairSet::with("a", "x2"), 1),
            (PairSet::with("a", "x3"), 1),
            (PairSet::with("a", "x4"), 1),

            (PairSet::with("a", "b"), 2),
            (PairSet::with("a", "c"), 2),

            (PairSet::with("b", "x1"), 1),
            (PairSet::with("b", "x2"), 1),

            (PairSet::with("c", "x3"), 1),
            (PairSet::with("c", "x4"), 1),
        ]);
        assert_eq!(expected_soc, idep.soc);
        assert_eq!(expected_poc, idep.poc);

        assert_eq!(idep.dependency("a", "x1"), 0.25);
        assert_eq!(idep.dependency("a", "x2"), 0.25);
        assert_eq!(idep.dependency("a", "x3"), 0.25);
        assert_eq!(idep.dependency("a", "x4"), 0.25);

        assert_eq!(idep.dependency("x1", "a"), 1.0);
        assert_eq!(idep.dependency("x2", "a"), 1.0);
        assert_eq!(idep.dependency("x3", "a"), 1.0);
        assert_eq!(idep.dependency("x4", "a"), 1.0);

        assert_eq!(idep.dependency("b", "x1"), 0.5);
        assert_eq!(idep.dependency("b", "x2"), 0.5);

        assert_eq!(idep.dependency("x1", "b"), 1.0);
        assert_eq!(idep.dependency("x2", "b"), 1.0);

        assert_eq!(idep.dependency("c", "x3"), 0.5);
        assert_eq!(idep.dependency("c", "x4"), 0.5);

        assert_eq!(idep.dependency("x3", "c"), 1.0);
        assert_eq!(idep.dependency("x4", "c"), 1.0);

        assert_eq!(idep.dependency("a", "b"), 0.5);
        assert_eq!(idep.dependency("a", "c"), 0.5);

        assert_eq!(idep.dependency("b", "a"), 1.0);
        assert_eq!(idep.dependency("c", "a"), 1.0);

        assert!(idep.contains("a"));
        assert!(idep.contains("b"));
        assert!(idep.contains("c"));
        assert!(idep.contains("x1"));
        assert!(idep.contains("x2"));
        assert!(idep.contains("x3"));
        assert!(idep.contains("x4"));

        assert!(!idep.contains("z"));
        assert!(!idep.contains("x5"));
    }

    
    struct MockTokenizer;
    impl Tokenize for MockTokenizer {
        fn tokenize<'a>(&self, msg: &'a str) -> Vec<Token<'a>> {
            msg.split(' ')
                .map(|slice| Token::with(slice, &HashSet::default()))
                .collect_vec()
        }
    }

    struct MockFilter;
    impl TokenFilter for MockFilter {
        fn token_filter(&self, _tok: &Token) -> bool {
            true
        }
    }
}
