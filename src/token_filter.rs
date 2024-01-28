use crate::traits::TokenFilter;
use crate::tokenizer::Token;


pub struct InterdependencyFilter{
    alphabetic: bool,
    numeric: bool,
    impure: bool,
}

impl InterdependencyFilter {
    pub fn with(alphabetic: bool, numeric: bool, impure: bool) -> Self {
        Self {
            alphabetic,
            numeric,
            impure,
        }
    }
}

impl TokenFilter for InterdependencyFilter {
    fn token_filter(&self, tok: &Token) -> bool {
        match tok {
            Token::Alphabetic(_) => self.alphabetic,
            Token::Numeric(_) => self.numeric,
            Token::Impure(_) => self.impure,
            Token::Symbolic(_) => false,
            Token::Whitespace(_) => false,
            Token::SpecialBlack(_) => false,
            Token::SpecialWhite(_) => true,
        }
    }
}