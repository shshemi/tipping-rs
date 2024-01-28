use crate::tokenizer::Token;

pub trait Tokenize {
    fn tokenize<'a>(&self, msg: &'a str) -> Vec<Token<'a>>;
}

pub trait TokenFilter {
    fn token_filter(&self, tok: &Token) -> bool;
}

pub trait Dependency<T> {
    fn dependency(&self, event: T, condition: T) -> f32;
}

pub trait Contains<T> {
    fn contains(&self, item: T) -> bool;
}