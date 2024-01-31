use std::{
    collections::{BTreeSet, HashMap, HashSet},
    marker::PhantomData,
};

use rayon::prelude::*;

use regex::Regex;

use crate::{
    graph::{build_graph, key_node_values},
    interdependency::Interdependency,
    template::{parameter_masks, shared_slices, templates},
    token_filter::InterdependencyFilter,
    tokenizer::{Token, Tokenizer},
    traits::{Contains, Dependency, Tokenize},
};

type Clusters = Vec<Option<usize>>;
type Templates = Vec<HashSet<String>>;
type Masks = HashMap<String, String>;

pub struct NoCompute;
pub struct Compute;

/// Tipping (Token Interdependency Parsing) log parser
#[derive(Debug, Clone)]
pub struct Parser<Templates = NoCompute, Masks = NoCompute> {
    threshold: f32,
    special_whites: Vec<Regex>,
    special_blacks: Vec<Regex>,
    symbols: HashSet<char>,
    filter_alphabetic: bool,
    filter_numeric: bool,
    filter_impure: bool,
    compute_templates: PhantomData<Templates>,
    compute_mask: PhantomData<Masks>,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}
impl Parser {
    /// Initiates a new Parser with default parameters
    pub fn new() -> Self {
        Parser {
            threshold: 0.5,
            special_whites: Default::default(),
            special_blacks: Default::default(),
            symbols: Default::default(),
            filter_alphabetic: true,
            filter_numeric: false,
            filter_impure: false,
            compute_templates: Default::default(),
            compute_mask: Default::default(),
        }
    }

    /// Sets `value` as threshold. The threshold determines if an interdependency
    /// link should be considered during key node computation or not. The
    /// threshod should be `0 <= threshold <= 1`.
    #[must_use]
    pub fn with_threshold(mut self, value: f32) -> Self {
        assert!(0.0 <= value);
        assert!(value <= 1.0);
        self.threshold = value;
        self
    }

    /// Sets `value` as spcial white regexes. White regexes are never parameterized.
    #[must_use]
    pub fn with_special_whites(mut self, value: Vec<Regex>) -> Self {
        self.special_whites = value;
        self
    }

    /// Sets `value` as special black regexes. Black regexes are always parameterized.
    #[must_use]
    pub fn with_special_blacks(mut self, value: Vec<Regex>) -> Self {
        self.special_blacks = value;
        self
    }

    /// Sets `value` as symbols. Symbolds are characters that are used alongside the
    /// white spaces to split string during tokenization.
    #[must_use]
    pub fn with_symbols(mut self, value: HashSet<char>) -> Self {
        self.symbols = value;
        self
    }

    /// The `value` determines if alphabetic tokens should be used during key node computation.
    #[must_use]
    pub fn with_filter_alphabetic(mut self, value: bool) -> Self {
        self.filter_alphabetic = value;
        self
    }

    /// The `value` determines if numeric tokens should be used during key node computation.
    #[must_use]
    pub fn with_filter_numeric(mut self, value: bool) -> Self {
        self.filter_numeric = value;
        self
    }

    /// The `value` determines if impure tokens should be used during key node computation.
    #[must_use]
    pub fn with_filter_impure(mut self, value: bool) -> Self {
        self.filter_impure = value;
        self
    }
}

impl<T> Parser<NoCompute, T> {
    // Add templates computation to the output
    #[must_use]
    pub fn compute_templates(self) -> Parser<Compute, T> {
        Parser::<Compute, T> {
            threshold: self.threshold,
            special_whites: self.special_whites,
            special_blacks: self.special_blacks,
            symbols: self.symbols,
            filter_alphabetic: self.filter_alphabetic,
            filter_numeric: self.filter_numeric,
            filter_impure: self.filter_impure,
            compute_templates: Default::default(),
            compute_mask: Default::default(),
        }
    }
}

impl<T> Parser<T, NoCompute> {
    // Add parameter mask computation to the output
    #[must_use]
    pub fn compute_masks(self) -> Parser<T, Compute> {
        Parser::<T, Compute> {
            threshold: self.threshold,
            special_whites: self.special_whites,
            special_blacks: self.special_blacks,
            symbols: self.symbols,
            filter_alphabetic: self.filter_alphabetic,
            filter_numeric: self.filter_numeric,
            filter_impure: self.filter_impure,
            compute_templates: Default::default(),
            compute_mask: Default::default(),
        }
    }
}

impl Parser<NoCompute, NoCompute> {
    /// Parses the input `messages` and returns `Clusters`.
    ///
    /// - `Clusters`: A `Vec<Option<usize>>` representing potential cluster IDs. Each `Option<usize>`
    ///   corresponds to the cluster ID of the message at the same index, or `None` if the message
    ///   couldn't be parsed.
    ///
    pub fn parse<Message: AsRef<str> + Sync>(self, messages: &[Message]) -> Clusters {
        let tokenizer = Tokenizer::new(self.special_whites, self.special_blacks, self.symbols);
        let filter = InterdependencyFilter::with(
            self.filter_alphabetic,
            self.filter_numeric,
            self.filter_impure,
        );
        let idep = Interdependency::new(messages, &tokenizer, &filter);
        let cmap = group_by_key_tokens(messages, &tokenizer, &idep, self.threshold);
        let mut clus = vec![None; messages.len()];
        cmap.into_iter()
            .filter(|(key_toks, _)| !key_toks.is_empty())
            .enumerate()
            .for_each(|(cid, (_, indices))| {
                for idx in indices {
                    clus[idx] = Some(cid);
                }
            });
        clus
    }
}

impl Parser<Compute, NoCompute> {
    /// Parses the input `messages` and returns `Clusters`, and `Templates`.
    ///
    /// - `Clusters`: A `Vec<Option<usize>>` representing potential cluster IDs. Each `Option<usize>`
    ///   corresponds to the cluster ID of the message at the same index, or `None` if the message
    ///   couldn't be parsed.
    ///
    /// - `Templates`: A `Vec<HashSet<String>>` where each set of templates is aligned with the
    ///   corresponding cluster ID in the `Clusters` vector.
    ///
    pub fn parse<Message: AsRef<str> + Sync>(self, messages: &[Message]) -> (Clusters, Templates) {
        let tokenizer = Tokenizer::new(self.special_whites, self.special_blacks, self.symbols);
        let filter = InterdependencyFilter::with(
            self.filter_alphabetic,
            self.filter_numeric,
            self.filter_impure,
        );
        let idep = Interdependency::new(messages, &tokenizer, &filter);
        let cmap = group_by_key_tokens(messages, &tokenizer, &idep, self.threshold);
        let mut clus = vec![None; messages.len()];
        let mut temps = vec![HashSet::default(); cmap.len()];
        cmap.into_iter()
            .filter(|(key_toks, _)| !key_toks.is_empty())
            .enumerate()
            .for_each(|(cid, (_, indices))| {
                let stok = shared_slices(
                    indices.iter().cloned().map(|idx| messages[idx].as_ref()),
                    &tokenizer,
                    self.filter_alphabetic,
                    self.filter_numeric,
                    self.filter_impure,
                );
                temps[cid] = templates(
                    indices.iter().cloned().map(|idx| messages[idx].as_ref()),
                    &tokenizer,
                    &stok,
                );
                for idx in indices {
                    clus[idx] = Some(cid);
                }
            });

        (clus, temps)
    }
}

impl Parser<NoCompute, Compute> {
    /// Parses the input `messages` and returns `Clusters`,  `Masks`.
    ///
    /// - `Clusters`: A `Vec<Option<usize>>` representing potential cluster IDs. Each `Option<usize>`
    ///   corresponds to the cluster ID of the message at the same index, or `None` if the message
    ///   couldn't be parsed.
    ///
    /// - `Masks`: A table mapping each message to its parameter masks.
    ///
    pub fn parse<Message: AsRef<str> + Sync>(self, messages: &[Message]) -> (Clusters, Masks) {
        let tokenizer = Tokenizer::new(self.special_whites, self.special_blacks, self.symbols);
        let filter = InterdependencyFilter::with(
            self.filter_alphabetic,
            self.filter_numeric,
            self.filter_impure,
        );
        let idep = Interdependency::new(messages, &tokenizer, &filter);
        let cmap = group_by_key_tokens(messages, &tokenizer, &idep, self.threshold);
        let mut clus = vec![None; messages.len()];
        let mut masks = HashMap::new();
        cmap.into_iter()
            .filter(|(key_toks, _)| !key_toks.is_empty())
            .enumerate()
            .for_each(|(cid, (_, indices))| {
                let stok = shared_slices(
                    indices.iter().cloned().map(|idx| messages[idx].as_ref()),
                    &tokenizer,
                    self.filter_alphabetic,
                    self.filter_numeric,
                    self.filter_impure,
                );
                masks.extend(parameter_masks(
                    indices.iter().cloned().map(|idx| messages[idx].as_ref()),
                    &tokenizer,
                    &stok,
                ));
                for idx in indices {
                    clus[idx] = Some(cid);
                }
            });

        (clus, masks)
    }
}

impl Parser<Compute, Compute> {
    /// Parses the input `messages` and returns `Clusters`, `Templates`, and `Masks`.
    ///
    /// - `Clusters`: A `Vec<Option<usize>>` representing potential cluster IDs. Each `Option<usize>`
    ///   corresponds to the cluster ID of the message at the same index, or `None` if the message
    ///   couldn't be parsed.
    ///
    /// - `Templates`: A `Vec<HashSet<String>>` where each set of templates is aligned with the
    ///   corresponding cluster ID in the `Clusters` vector.
    ///
    /// - `Masks`: A table mapping each message to its parameter masks.
    ///
    pub fn parse<Message: AsRef<str> + Sync>(
        self,
        messages: &[Message],
    ) -> (Clusters, Templates, Masks) {
        let tokenizer = Tokenizer::new(self.special_whites, self.special_blacks, self.symbols);
        let filter = InterdependencyFilter::with(
            self.filter_alphabetic,
            self.filter_numeric,
            self.filter_impure,
        );
        let idep = Interdependency::new(messages, &tokenizer, &filter);
        let groups = group_by_key_tokens(messages, &tokenizer, &idep, self.threshold);
        let mut clus = vec![None; messages.len()];
        let mut temps = vec![HashSet::default(); groups.len()];
        let mut masks = HashMap::new();
        groups
            .into_iter()
            .filter(|(key_toks, _)| !key_toks.is_empty())
            .enumerate()
            .for_each(|(cid, (_, indices))| {
                let stok = shared_slices(
                    indices.iter().cloned().map(|idx| messages[idx].as_ref()),
                    &tokenizer,
                    self.filter_alphabetic,
                    self.filter_numeric,
                    self.filter_impure,
                );
                temps[cid] = templates(
                    indices.iter().cloned().map(|idx| messages[idx].as_ref()),
                    &tokenizer,
                    &stok,
                );
                masks.extend(parameter_masks(
                    indices.iter().cloned().map(|idx| messages[idx].as_ref()),
                    &tokenizer,
                    &stok,
                ));
                for idx in indices {
                    clus[idx] = Some(cid);
                }
            });

        (clus, temps, masks)
    }
}

fn group_by_key_tokens<'a, T: AsRef<str> + Sync>(
    messages: &'a [T],
    tokenizer: &Tokenizer,
    idep: &'a Interdependency<'a>,
    threshold: f32,
) -> HashMap<BTreeSet<Token<'a>>, BTreeSet<usize>> {
    messages
        .iter()
        .enumerate()
        .par_bridge()
        .map(|(idx, msg)| {
            (idx, {
                let tokens = tokenizer.tokenize(msg.as_ref());
                let graph = build_graph(
                    tokens
                        .iter()
                        .copied()
                        .filter(|tok| idep.contains(tok.as_str())),
                    |tok1, tok2| idep.dependency(tok1.as_str(), tok2.as_str()) > threshold,
                );
                let mut key_toks = key_node_values(graph);
                for tok in tokens {
                    match tok {
                        Token::SpecialWhite(_) => {
                            key_toks.insert(tok);
                        }
                        Token::SpecialBlack(_) => {
                            key_toks.remove(&tok);
                        }
                        _ => (),
                    }
                }
                key_toks
            })
        })
        .fold_with(
            HashMap::<BTreeSet<Token<'a>>, BTreeSet<usize>>::new(),
            |mut map, (idx, key_tokens)| {
                map.entry(key_tokens)
                    .and_modify(|indices| {
                        indices.insert(idx);
                    })
                    .or_insert([idx].into());
                map
            },
        )
        .reduce_with(|mut m1, m2| {
            m2.into_iter().for_each(|(k, v2)| {
                if let Some(v1) = m1.get_mut(&k) {
                    v1.extend(v2);
                } else {
                    m1.insert(k, v2);
                }
            });
            m1
        })
        .unwrap_or_default()
}
